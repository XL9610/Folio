#!/usr/bin/env python3
"""Fair A/B grounding review for the Phase 3 unsupported-claim gate.

Unlike the judge-flagged sheets used during development, this pipeline
extracts ALL major factual claims from frozen flag-off and flag-on
answers with one frozen extractor, semantically deduplicates them per
answer, and (when needed) samples the same number of claims per question
per configuration. The output is a blind human-review sheet plus a
separate decode key; `--decode` computes the final metrics after scoring.

Frozen pipeline (do not tune per-configuration):
  - extractor prompt EXTRACTOR_PROMPT_VERSION = "1", temperature 0;
  - extraction runs twice per answer, claims are unioned;
  - dedup: exact (normalized) then semantic (content-word Jaccard >= 0.8
    or containment >= 0.9 keeps the larger claim);
  - sampling: per question, min(n_off, n_on) claims, optional cap.

Usage:
    python backend/scripts/overview_fair_review.py \\
        --off <flag-off capture.json> --on <flag-on capture.json> \\
        --sheet <sheet.md> --key <key.txt> [--max-per-question N]
    python backend/scripts/overview_fair_review.py \\
        --decode <scored-sheet.md> --key <key.txt>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.llm import LLMConfig  # noqa: E402
from app.services.llm.client import create_llm_client  # noqa: E402

EXTRACTOR_PROMPT_VERSION = "1"

EXTRACTOR_SYSTEM_PROMPT = """You extract major factual claims from a study-companion answer for human review.

A major factual claim is an assertion of specific fact about the source material: events, dates, names, numbers, quotations, character actions, causal statements, attributions of content to the source.

Exclude: headings, purely organizational text, restatements of the question, meta-commentary about what the answer does, hedging about the review process, and connective sentences with no factual content.

Rules:
1. Reproduce each claim verbatim from the answer (trim whitespace), including any [S#] or [W#] citation markers that appear in the same sentence or immediately after it.
2. Do NOT judge whether a claim is supported. Include unsupported-looking and supported-looking claims alike.
3. Do not merge two distinct facts into one claim. Do not split one fact into two claims.
4. Prefer completeness over brevity: include every major factual assertion.

Return ONLY valid JSON:
{"claims": [{"text": "...", "citations": ["S3"]}]}"""

_STOP = set(
    "the a an and or but of in on at to for with by from as is are was were "
    "be been this that these those it its their his her our your i you he "
    "she we they not no which who whom whose when where why how says said "
    "based retrieved material book".split()
)
_REF_RE = re.compile(r"\[([SW]\d+)\]")
_NORM_RE = re.compile(r"\s+")


def _words(s: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", s.lower())
        if len(w) >= 3 and w not in _STOP
    }


def _extract_claims(judge, question: str, answer: str) -> list[dict]:
    """Two frozen extraction passes, unioned."""
    merged: dict[str, dict] = {}
    for _ in range(2):
        try:
            result = judge.complete_json(
                system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                user_prompt=f"Question: {question}\n\nAnswer:\n{answer}",
                # Long answers yield many claims; the app-wide
                # LLM_MAX_TOKENS=1024 truncates the extractor's JSON.
                max_tokens=4096,
            )
        except Exception:
            continue
        for c in (result.data or {}).get("claims", []):
            if not isinstance(c, dict):
                continue
            text = _NORM_RE.sub(" ", str(c.get("text") or "")).strip()
            if not text:
                continue
            refs = [r for r in (c.get("citations") or [])
                    if isinstance(r, str) and _REF_RE.fullmatch(r)]
            key = text.lower()
            merged.setdefault(key, {"text": text, "citations": refs})
    return list(merged.values())


def _dedup_claims(claims: list[dict]) -> list[dict]:
    """Semantic dedup per answer: near-duplicates collapse, keeping the
    longer text; near-contained fragments collapse into their container."""
    claims = sorted(claims, key=lambda c: -len(c["text"]))
    kept: list[dict] = []
    for c in claims:
        wc = _words(c["text"])
        if not wc:
            continue
        duplicate = False
        for k in kept:
            wk = _words(k["text"])
            inter = len(wc & wk)
            union = len(wc | wk)
            jac = inter / union if union else 0.0
            contain = inter / min(len(wc), len(wk)) if min(len(wc), len(wk)) else 0.0
            if jac >= 0.8 or contain >= 0.9:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
    return kept


def _build_sheet(off: dict, on: dict, args) -> None:
    cfg = LLMConfig()
    cfg.temperature = 0.0
    judge = create_llm_client(cfg)

    off_map = {e["id"]: e for e in off}
    on_map = {e["id"]: e for e in on}
    ids = sorted(set(off_map) & set(on_map))
    if not ids:
        sys.exit("no common question ids between captures")

    extracted: dict[str, dict[str, list[dict]]] = {}
    for qid in ids:
        for config, capture in (("flag-off", off_map), ("flag-on", on_map)):
            entry = capture[qid]
            claims = _dedup_claims(
                _extract_claims(judge, entry["question"], entry["answer"])
            )
            extracted.setdefault(qid, {})[config] = claims

    # Balanced sampling: same count per question per configuration.
    sampled: dict[str, dict[str, list[dict]]] = {}
    stats: list[dict] = []
    for qid in ids:
        n = min(len(extracted[qid]["flag-off"]), len(extracted[qid]["flag-on"]))
        if args.max_per_question and n > args.max_per_question:
            n = args.max_per_question
        sampled[qid] = {
            "flag-off": extracted[qid]["flag-off"][:n],
            "flag-on": extracted[qid]["flag-on"][:n],
        }
        stats.append({
            "id": qid,
            "extracted_off": len(extracted[qid]["flag-off"]),
            "extracted_on": len(extracted[qid]["flag-on"]),
            "sampled_per_config": n,
        })

    # Blind ordering: fixed shuffle across both configurations.
    import random
    order: list[tuple[str, str]] = []
    for qid in ids:
        for config in ("flag-off", "flag-on"):
            order.append((qid, config))
    random.Random(20261013).shuffle(order)

    lines = [
        "# Phase 3 — fair blind human review: grounding A/B gate",
        "",
        "**Reviewer instructions.** Each sample is one frozen answer with its cited",
        "source passages. Score EVERY claim below with exactly one box:",
        "",
        "- `[ ] supported` — at least one cited passage ([S1]...[S10]) supports the claim (paraphrase counts);",
        "- `[x] unsupported` — no cited passage supports it;",
        "- `[ ] not a claim` — the item is connective/organizational text, not a major factual claim (excluded from the rate).",
        "",
        "Ignore minor stylistic or connective statements. Judge claims as written.",
        "Claims were extracted from both configurations with one frozen pipeline",
        "and balanced per question. Sample ids are blind; the decode key is in a",
        "separate file — do not open it until all scoring is done.",
        "",
        "---",
    ]
    key_lines = ["# Fair-review decode key — open only after scoring all samples", ""]
    sample_no = 0
    sample_map: list[tuple[int, str, str]] = []
    for qid, config in order:
        entry = (off_map if config == "flag-off" else on_map)[qid]
        claims = sampled[qid][config]
        if not claims:
            continue
        sample_no += 1
        sample_map.append((sample_no, qid, config))
        key_lines.append(f"- Sample {sample_no}: {config}  ({qid})")
        lines.append(f'\n## Sample {sample_no}')
        lines.append(f'\n**Question:** {entry["question"]}')
        lines.append(f'\n**Answer:**\n\n{entry["answer"]}')
        lines.append('\n**Cited source passages:**')
        for p in entry["passages"]:
            loc = p["section_path"] or ""
            lines.append(f'\n[{p["ref"]}] ({loc})\n{p["text"][:1500]}')
        lines.append(f'\n**Claims to score ({len(claims)}):**')
        for j, c in enumerate(claims, 1):
            refs = "".join(f"[{r}]" for r in c["citations"])
            lines.append(f'\n{j}. {c["text"]}{" " + refs if refs else ""}')
            lines.append("   - [ ] supported")
            lines.append("   - [ ] unsupported")
            lines.append("   - [ ] not a claim")
        lines.append("\n---")

    Path(args.sheet).write_text("\n".join(lines), encoding="utf-8")
    Path(args.key).write_text("\n".join(key_lines), encoding="utf-8")

    print(f"sheet: {args.sheet} ({sample_no} samples)")
    print(f"key:   {args.key}")
    print("extraction stats:")
    for s in stats:
        print(f"  {s['id']}: extracted off={s['extracted_off']} on={s['extracted_on']} "
              f"-> sampled {s['sampled_per_config']} per config")
    total = sum(s["sampled_per_config"] * 2 for s in stats)
    print(f"total claims to review: {total}")
    json.dump({"stats": stats, "sample_map": [
        {"sample": n, "id": q, "config": c} for n, q, c in sample_map
    ]}, open(str(Path(args.sheet).with_suffix(".meta.json")), "w", encoding="utf-8"),
        indent=2)


def _decode(sheet_path: Path, key_path: Path) -> None:
    sheet = sheet_path.read_text(encoding="utf-8")
    key = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"- Sample (\d+): (\S+)\s+\(([^)]+)\)", line)
        if m:
            key[int(m.group(1))] = (m.group(2), m.group(3))

    samples = re.split(r"\n## Sample (\d+)", sheet)[1:]
    scores: dict[str, dict] = {}
    per_answer: dict[tuple[str, str], dict] = {}
    for i in range(0, len(samples), 2):
        num = int(samples[i])
        body = samples[i + 1]
        if num not in key:
            continue
        config, qid = key[num]
        claim_blocks = re.split(r"\n(\d+)\. ", body.split("**Claims to score**", 1)[1])[1:]
        n_sup = n_uns = n_not = 0
        for j in range(0, len(claim_blocks), 2):
            cbody = claim_blocks[j + 1]
            marks = {
                "supported": "[x] supported" in cbody,
                "unsupported": "[x] unsupported" in cbody,
                "not a claim": "[x] not a claim" in cbody,
            }
            marked = sum(marks.values())
            if marked != 1:
                print(f"WARNING sample {num} claim {claim_blocks[j]}: {marked} boxes")
                continue
            n_sup += marks["supported"]
            n_uns += marks["unsupported"]
            n_not += marks["not a claim"]
        s = scores.setdefault(config, {"supported": 0, "unsupported": 0,
                                       "not_a_claim": 0, "total": 0})
        s["supported"] += n_sup
        s["unsupported"] += n_uns
        s["not_a_claim"] += n_not
        s["total"] += n_sup + n_uns + n_not
        per_answer[(config, qid)] = {
            "supported": n_sup, "unsupported": n_uns, "not_a_claim": n_not,
        }

    print("=== Fair A/B grounding gate ===")
    for config in ("flag-off", "flag-on"):
        s = scores[config]
        scored = s["supported"] + s["unsupported"]
        rate = s["unsupported"] / scored if scored else None
        print(f"{config}: {s['supported']} supported, {s['unsupported']} unsupported, "
              f"{s['not_a_claim']} not-a-claim -> unique unsupported rate "
              f"= {rate:.4f}" if rate is not None else f"{config}: no scorable claims")
    print("per-answer unsupported density (unsupported / scorable claims):")
    for (config, qid), v in sorted(per_answer.items()):
        scored = v["supported"] + v["unsupported"]
        rate = v["unsupported"] / scored if scored else None
        print(f"  {config} {qid}: {v['unsupported']}/{scored} = {rate}")
    base = scores["flag-off"]
    cand = scores["flag-on"]
    bs = base["supported"] + base["unsupported"]
    cs = cand["supported"] + cand["unsupported"]
    if bs and cs:
        br = base["unsupported"] / bs
        cr = cand["unsupported"] / cs
        print(f"gate: candidate {cr:.4f} vs baseline {br:.4f} "
              f"-> {'PASS' if cr <= br else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off", type=Path, help="frozen flag-off capture JSON")
    parser.add_argument("--on", type=Path, help="frozen flag-on capture JSON")
    parser.add_argument("--sheet", type=Path, help="review sheet output path")
    parser.add_argument("--key", type=Path, help="decode key output path")
    parser.add_argument("--max-per-question", type=int, default=None,
                        help="cap claims per question per config (default: balanced exhaustive)")
    parser.add_argument("--decode", nargs=2, type=Path, metavar=("SHEET", "KEY"),
                        help="decode a scored sheet: --decode sheet.md key.txt")
    args = parser.parse_args()

    if args.decode:
        _decode(args.decode[0], args.decode[1])
        return 0
    if not (args.off and args.on and args.sheet and args.key):
        parser.error("--off --on --sheet --key are required (or use --decode)")
    off = json.loads(args.off.read_text(encoding="utf-8"))
    on = json.loads(args.on.read_text(encoding="utf-8"))
    _build_sheet(off, on, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
