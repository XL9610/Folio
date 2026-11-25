#!/usr/bin/env python3
"""Fair A/B grounding review for the Phase 3 unsupported-claim gate.

Unlike the judge-flagged sheets used during development, this pipeline
extracts ALL major factual claims from frozen answers with one frozen
extractor and semantically deduplicates them per answer. Balanced mode
samples the same number of claims per question per configuration;
--no-balance reviews every deduped claim of every answer (each answer
weighted equally, nothing silently dropped). The output is a blind
human-review sheet plus a separate decode key; `--decode` computes the
final metrics after scoring (micro unsupported rate, per-answer density,
macro density average, claims per answer, per-configuration gate lines).

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


def _build_sheet(configs: dict[str, list], args) -> None:
    """Build one blind sheet balanced across N configurations.

    ``configs`` maps a configuration name to its frozen capture list.
    Per question, min(config claim counts) claims are sampled from every
    configuration; samples are shuffled blind with a fixed seed.
    """
    cfg = LLMConfig()
    cfg.temperature = 0.0
    judge = create_llm_client(cfg)

    by_config = {name: {e["id"]: e for e in capture}
                 for name, capture in configs.items()}
    ids = sorted(set.intersection(*(set(m) for m in by_config.values())))
    if not ids:
        sys.exit("no common question ids between captures")

    extracted: dict[str, dict[str, list[dict]]] = {}
    for qid in ids:
        for name, entries in by_config.items():
            entry = entries[qid]
            claims = _dedup_claims(
                _extract_claims(judge, entry["question"], entry["answer"])
            )
            extracted.setdefault(qid, {})[name] = claims

    # Balanced sampling: same count per question across configurations.
    # --no-balance keeps every deduped claim from every configuration
    # (all-claims review; each answer is weighted equally and nothing is
    # silently dropped).
    sampled: dict[str, dict[str, list[dict]]] = {}
    stats: list[dict] = []
    for qid in ids:
        if args.no_balance:
            sampled[qid] = {
                name: list(extracted[qid][name]) for name in configs
            }
            stats.append({
                "id": qid,
                "extracted": {name: len(extracted[qid][name]) for name in configs},
                "sampled_per_config": None,
            })
            continue
        n = min(len(extracted[qid][name]) for name in configs)
        if args.max_per_question and n > args.max_per_question:
            n = args.max_per_question
        sampled[qid] = {
            name: extracted[qid][name][:n] for name in configs
        }
        stats.append({
            "id": qid,
            "extracted": {name: len(extracted[qid][name]) for name in configs},
            "sampled_per_config": n,
        })

    # Blind ordering: fixed shuffle across configurations.
    import random
    order: list[tuple[str, str]] = []
    for qid in ids:
        for name in configs:
            order.append((qid, name))
    random.Random(20261013).shuffle(order)

    lines = [
        "# Fair blind human review: grounding gate",
        "",
        "**Reviewer instructions.** Each sample is one frozen answer with its cited",
        "source passages. Score EVERY claim below with exactly one box:",
        "",
        "- `[ ] supported` — at least one cited passage ([S1]...[S10]) supports the claim (paraphrase counts);",
        "- `[x] unsupported` — no cited passage supports it;",
        "- `[ ] not a claim` — the item is connective/organizational text, not a major factual claim (excluded from the rate).",
        "",
        "Ignore minor stylistic or connective statements. Judge claims as written.",
        "Claims were extracted from every configuration with one frozen pipeline"
        + (
            "; every deduped claim of every answer is reviewed — each answer"
            " is weighted equally."
            if args.no_balance else
            " and balanced per question."
        ),
        " Sample ids are blind; the decode key is in a",
        "separate file — do not open it until all scoring is done.",
        "",
        "---",
    ]
    key_lines = ["# Fair-review decode key — open only after scoring all samples", ""]
    sample_no = 0
    sample_map: list[tuple[int, str, str]] = []
    for qid, name in order:
        entry = by_config[name][qid]
        claims = sampled[qid][name]
        if not claims:
            continue
        sample_no += 1
        sample_map.append((sample_no, qid, name))
        key_lines.append(f"- Sample {sample_no}: {name}  ({qid})")
        lines.append(f'\n## Sample {sample_no}')
        lines.append(f'\n**Question:** {entry["question"]}')
        lines.append(f'\n**Answer:**\n\n{entry["answer"]}')
        lines.append('\n**Cited source passages:**')
        for p in entry["passages"]:
            loc = p["section_path"] or ""
            lines.append(f'\n[{p["ref"]}] ({loc})\n{p["text"][:2000]}')
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
        if s["sampled_per_config"] is None:
            print(f"  {s['id']}: all-claims {s['extracted']}")
        else:
            print(f"  {s['id']}: extracted {s['extracted']} "
                  f"-> sampled {s['sampled_per_config']} per config")
    total = sum(
        len(sampled[qid][name]) for qid in ids for name in configs
    )
    print(f"total claims to review: {total}")


def _parse_sheet(sheet: str, key: dict[int, tuple[str, str]]):
    """Parse a scored sheet into per-config totals and per-answer counts."""
    samples = re.split(r"\n## Sample (\d+)", sheet)[1:]
    scores: dict[str, dict] = {}
    per_answer: dict[tuple[str, str], dict] = {}
    for i in range(0, len(samples), 2):
        num = int(samples[i])
        body = samples[i + 1]
        if num not in key:
            continue
        config, qid = key[num]
        # Regex, not split: the header carries a count suffix, e.g.
        # "**Claims to score (20):**".
        m = re.search(
            r"\*\*Claims to score(?: \(\d+\))?:\*\*\s*\n(.*?)(?=\n---|\Z)",
            body, re.S,
        )
        if not m:
            print(f"WARNING sample {num}: no claims section found")
            continue
        # Prepend a newline so a claim at the very start of the section is
        # split correctly.
        claim_blocks = re.split(r"\n(\d+)\. ", "\n" + m.group(1))[1:]
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
            "n_scorable": n_sup + n_uns,
        }
    return scores, per_answer


def _decode(sheet_path: Path, key_path: Path, baseline: str | None) -> None:
    sheet = sheet_path.read_text(encoding="utf-8")
    key = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"- Sample (\d+): (\S+)\s+\(([^)]+)\)", line)
        if m:
            key[int(m.group(1))] = (m.group(2), m.group(3))

    scores, per_answer = _parse_sheet(sheet, key)

    print("=== Fair grounding gate ===")
    for config, s in scores.items():
        scored = s["supported"] + s["unsupported"]
        if scored:
            rate = s["unsupported"] / scored
            print(f"{config}: {s['supported']} supported, {s['unsupported']} "
                  f"unsupported, {s['not_a_claim']} not-a-claim -> micro "
                  f"unsupported rate = {rate:.4f}")
        else:
            print(f"{config}: no scorable claims")
    print("per-answer unsupported density (unsupported / scorable claims):")
    densities: dict[str, list[float]] = {}
    claims_per_answer: dict[str, list[int]] = {}
    for (config, qid), v in sorted(per_answer.items()):
        scored = v["supported"] + v["unsupported"]
        rate = v["unsupported"] / scored if scored else None
        densities.setdefault(config, []).append(rate)
        claims_per_answer.setdefault(config, []).append(scored)
        print(f"  {config} {qid}: {v['unsupported']}/{scored} = {rate}")
    print("macro density (mean of per-answer densities) and claims/answer:")
    for config in densities:
        ds = [d for d in densities[config] if d is not None]
        cs = claims_per_answer[config]
        macro = sum(ds) / len(ds) if ds else None
        print(f"  {config}: macro = {macro:.4f} (n={len(ds)})"
              if macro is not None else f"  {config}: no densities")
        print(f"  {config}: claims/answer = {cs}")
    if baseline and baseline in scores:
        base = scores[baseline]
        bs = base["supported"] + base["unsupported"]
        if not bs:
            print(f"gate: baseline {baseline} has no scorable claims")
            return
        br = base["unsupported"] / bs
        bmacro = (sum(d for d in densities[baseline] if d is not None)
                  / len([d for d in densities[baseline] if d is not None]))
        for config, cand in scores.items():
            if config == baseline:
                continue
            cs = cand["supported"] + cand["unsupported"]
            if not cs:
                print(f"gate: {config} has no scorable claims")
                continue
            cr = cand["unsupported"] / cs
            cmacro = (sum(d for d in densities[config] if d is not None)
                      / len([d for d in densities[config] if d is not None]))
            print(f"gate {config} vs {baseline}: micro {cr:.4f} vs {br:.4f} "
                  f"-> {'PASS' if cr <= br else 'FAIL'}; "
                  f"macro {cmacro:.4f} vs {bmacro:.4f} "
                  f"-> {'PASS' if cmacro <= bmacro else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", action="append", metavar="NAME=PATH",
                        help="frozen capture per configuration (repeatable)")
    parser.add_argument("--sheet", type=Path, help="review sheet output path")
    parser.add_argument("--key", type=Path, help="decode key output path")
    parser.add_argument("--max-per-question", type=int, default=None,
                        help="cap claims per question per config (balanced mode only)")
    parser.add_argument("--no-balance", action="store_true",
                        help="review ALL deduped claims of every configuration "
                             "(each answer weighted equally; nothing dropped)")
    parser.add_argument("--decode", nargs=2, type=Path, metavar=("SHEET", "KEY"),
                        help="decode a scored sheet: --decode sheet.md key.txt")
    parser.add_argument("--baseline", default=None,
                        help="baseline config name for gate lines (decode)")
    args = parser.parse_args()

    if args.decode:
        _decode(args.decode[0], args.decode[1], args.baseline)
        return 0
    if not args.captures or not args.sheet or not args.key:
        parser.error("--captures NAME=PATH (repeatable), --sheet, --key are required (or use --decode)")
    configs: dict[str, list] = {}
    for spec in args.captures:
        name, _, path = spec.partition("=")
        if not name or not path:
            parser.error(f"bad --captures spec: {spec}")
        configs[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    _build_sheet(configs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
