#!/usr/bin/env python3
"""Retrieval-only eval against a fixed ground-truth question set (PRD Phase 0/1).

No LLM judge: ranking metrics are computed deterministically against
ground-truth chunks bound by immutable content hashes. Runs against the
running backend:

  - ``POST /api/search``   -> ranked chunks + per-stage latency
  - ``POST /api/ask``      -> full ask pipeline (refusal classification,
    tool-trace-derived rerank accounting, client-side wall time)

Variants (--variants, Phase 1 of the roadmap):
  fts        keyword backend only
  dense      dense backend only
  rrf        dense + FTS + RRF fusion (the base retriever)
  ask        the complete ask pipeline (planner, query expansion, scoping,
             boosts, multi-round) — ranking scored on the chunks it
             actually retrieved, in the order generation saw them
  ask_rerank the ask pipeline with always-on LLM reranking — run against a
             backend started with RERANK_ENABLED=true RERANK_FORCE_ALL=true

Ground-truth questions file (JSON array):

    [
      {
        "id": "q-gk-01",
        "question": "What is binary search?",
        "category": "exact_lookup",
        "difficulty": "easy",
        "document_hash": "<sha256 of source file content>",
        "document_filename": "Grokking_Algorithms_....pdf",
        "additional_document_hashes": [],
        "expect_refusal": false,
        "relevant_chunks": [
          {"chunk_content_hash": "<sha256 of chunk text>", "page_start": 20,
           "page_end": 20, "section_path": "1. Binary search"}
        ]
      },
      {"id": "q-un-01", "question": "Who won the 2026 World Cup?",
       "category": "unanswerable", "expect_refusal": true, "relevant_chunks": []}
    ]

Chunk and document identity is content-hash based, so re-ingesting the corpus
(which regenerates chunk UUIDs) never invalidates the set; a question whose
hashes no longer resolve is reported as an extraction/chunking failure, not
silently scored.

Metrics per variant: Recall@5, Recall@10, Precision@10, MRR@10, nDCG@10,
refusal precision/recall (ask variants), latency, and rerank LLM calls
(ask variants, from the tool trace).

Usage:
    python backend/scripts/retrieval_eval.py \\
        --questions backend/scripts/retrieval_ground_truth.json \\
        --variants fts,dense,rrf,ask \\
        --out .scratch/retrieval-quality-roadmap/phase1-report.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / "local-data" / "rag_companion.db"

_ALL_VARIANTS = ("fts", "dense", "rrf", "ask", "ask_rerank")
_SEARCH_VARIANT_FLAGS = {"fts": {"dense_enabled": False}, "dense": {"keyword_enabled": False}, "rrf": {}}

_REFUSAL_PHRASES = (
    "does not cover",
    "doesn't cover",
    "does not contain enough information",
    "don't have any material",
    "no material on",
    "cannot be identified from the source",
    "no reliable answer",
    "no reliable information",
    "does not discuss",
    "doesn't discuss",
    "does not mention",
    "not covered",
    "out of scope",
)

# A refusal states absence up front; a real answer only mentions "does not X"
# deep inside an explanation, or pivots straight into substance ("does not
# discuss X, but shows Y"). Pivots after the phrase in the opening region
# mark the contrastive (answered) case.
_CONTRAST_PIVOTS = (" but ", " however", " although", " instead", ", though")


def _is_refusal(agent_status: str, answer: str) -> bool:
    """Refusal/clarification classification, shared with ragas_eval.py.

    ``agent_status`` is the authoritative signal. For answers the backend
    delivered with status "completed", a refusal phrase in the opening 200
    chars without a contrast pivot marks refusal wording. Calibrated on the
    observed backend outputs (refusals run 70-1700 chars, often cite [S#]).
    """
    if agent_status == "insufficient_context":
        return True
    if agent_status == "clarification_needed":
        return True
    if agent_status != "completed":
        return True
    head = (answer or "").lower()[:200]
    positions = [head.find(p) for p in _REFUSAL_PHRASES if p in head]
    if not positions:
        return False
    tail = head[min(positions):]
    return not any(p in tail for p in _CONTRAST_PIVOTS)


# ---------------------------------------------------------------------------
# Ground-truth resolution against the live DB (read-only)
# ---------------------------------------------------------------------------


def _load_ground_truth(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _open_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _resolve(question: dict, documents: list[dict], con: sqlite3.Connection) -> dict | None:
    """Resolve one question against the notebook's documents and the DB.

    Returns None (recorded as a resolution failure) or a dict with
    document_ids (search scope) and gt_chunk_hashes (relevant, present).
    """
    hashes = [question.get("document_hash", "")]
    hashes += question.get("additional_document_hashes", [])
    hashes = [h for h in hashes if h]
    if not hashes:  # unanswerable: whole-notebook scope, no ground truth
        return {"document_ids": [], "gt_chunk_hashes": set()}

    by_hash = {d.get("content_hash", ""): d for d in documents}
    doc_ids: list[str] = []
    for h in hashes:
        match = by_hash.get(h)
        if match:
            doc_ids.append(match["document_id"])
        else:
            return None  # document no longer in the notebook

    gt = {c["chunk_content_hash"] for c in question.get("relevant_chunks", [])}
    if gt:
        # Hashes whose chunk text is no longer in the DB = extraction changed.
        placeholders = ",".join("?" for _ in gt)
        present = {
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT content_hash FROM chunks WHERE content_hash IN ({placeholders})",
                tuple(gt),
            )
        }
        gt = present
        if not gt:
            return None  # all evidence chunks are gone
    return {"document_ids": doc_ids, "gt_chunk_hashes": gt}


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _search(api: str, question: str, notebook_id: str, document_ids: list[str], top_k: int, **extra) -> dict:
    payload: dict = {"notebook_id": notebook_id, "query": question, "top_k": top_k, **extra}
    if document_ids:
        payload["document_ids"] = document_ids
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{api}/api/search", json=payload)
        resp.raise_for_status()
        return resp.json()


def _ask(api: str, question: str, notebook_id: str, document_ids: list[str], top_k: int) -> dict:
    payload: dict = {
        "notebook_id": notebook_id,
        "query": question,
        "top_k": top_k,
        "history": [],
        "allow_web": False,
        "debug_retrieval": True,
    }
    if document_ids:
        payload["document_ids"] = document_ids
    with httpx.Client(timeout=180) as client:
        resp = client.post(f"{api}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(rel: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(rel))


def _ranking_metrics(ranked_hashes: list[str], gt: set[str]) -> dict:
    """Recall@5/10, Precision@10, MRR@10, nDCG@10 against a relevance set."""
    ranked = list(ranked_hashes)[:10]
    hits = [h for h in ranked if h in gt]
    hit_ranks = [i + 1 for i, h in enumerate(ranked) if h in gt]
    denom = len(gt)
    binary = [1 if h in gt else 0 for h in ranked]
    ideal = [1] * min(denom, len(ranked)) + [0] * max(0, len(ranked) - denom)
    return {
        "recall_at_5": (sum(1 for r in hit_ranks if r <= 5) / denom) if denom else None,
        "recall_at_10": (len(hits) / denom) if denom else None,
        "precision_at_10": len(hits) / len(ranked) if ranked else None,
        "mrr_at_10": (1.0 / min(hit_ranks)) if hit_ranks else 0.0,
        "ndcg_at_10": (_dcg(binary) / _dcg(ideal)) if _dcg(ideal) else 0.0,
        "hit_rank": min(hit_ranks) if hit_ranks else None,
        "n_hits": len(hits),
        "n_relevant": denom,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, required=True, help="Ground-truth JSON array")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Backend base URL (default %(default)s)")
    parser.add_argument("--notebook-id", default="default", help="Notebook to evaluate (default %(default)s)")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB, help="SQLite DB for chunk/hash resolution")
    parser.add_argument("--top-k", type=int, default=10, help="Chunks to retrieve per question (default %(default)s)")
    parser.add_argument("--variants", default=",".join(_ALL_VARIANTS),
                        help="Comma-separated variants to run (default: all five)")
    parser.add_argument("--out", type=Path, help="Write JSON report here (sibling .md summary is always written)")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in _ALL_VARIANTS]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}; choose from {_ALL_VARIANTS}")

    questions = _load_ground_truth(args.questions)
    if not isinstance(questions, list) or not questions:
        sys.exit("--questions must point to a non-empty JSON array.")

    # ── Corpus snapshot ──
    with httpx.Client(timeout=60) as client:
        resp = client.get(f"{args.api}/api/documents", params={"notebook_id": args.notebook_id})
        resp.raise_for_status()
        documents = resp.json()
    con = _open_db(args.db)
    chunk_hash_by_id = {r["id"]: r["content_hash"] for r in con.execute("SELECT id, content_hash FROM chunks")}

    resolved: list[dict] = []
    unresolved: list[dict] = []
    for q in questions:
        r = _resolve(q, documents, con)
        if r is None:
            unresolved.append({"id": q.get("id"), "question": q.get("question", "")[:80],
                               "reason": "document or evidence chunks not found in notebook/DB"})
            continue
        resolved.append({**q, **r})

    # ── Warmup search (model/DB caches) ──
    with contextlib.suppress(Exception):
        _search(args.api, "warmup query", args.notebook_id, [], args.top_k)

    rows: list[dict] = []
    latencies: dict[str, dict[str, list[float]]] = {v: {} for v in variants}
    for variant in variants:
        for q in resolved:
            entry = {
                "id": q.get("id"),
                "question": q.get("question"),
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "expect_refusal": bool(q.get("expect_refusal")),
                "document_filename": q.get("document_filename", ""),
                "variant": variant,
            }
            gt = q["gt_chunk_hashes"]

            if variant in _SEARCH_VARIANT_FLAGS:
                if q.get("expect_refusal"):  # nothing rankable without ground truth
                    continue
                try:
                    s = _search(args.api, q["question"], args.notebook_id, q["document_ids"],
                                args.top_k, **_SEARCH_VARIANT_FLAGS[variant])
                except Exception as exc:
                    rows.append({**entry, "failure": f"search failed: {exc}"})
                    continue
                entry["search_mode"] = s.get("search_mode")
                for stage, v in (s.get("latency_ms") or {}).items():
                    if isinstance(v, int | float):
                        latencies[variant].setdefault(stage, []).append(float(v))
                entry["latency_ms"] = s.get("latency_ms")
                ranked_hashes = [chunk_hash_by_id.get((r.get("chunk_id") or ""), "")
                                 for r in s.get("results", [])]
                entry.update(_ranking_metrics(ranked_hashes, gt))
                rows.append(entry)
                continue

            # ask / ask_rerank: full pipeline
            t0 = time.monotonic()
            try:
                a = _ask(args.api, q["question"], args.notebook_id, q["document_ids"], args.top_k)
            except Exception as exc:
                rows.append({**entry, "failure": f"ask failed: {exc}"})
                continue
            entry["wall_ms"] = round((time.monotonic() - t0) * 1000, 1)

            entry["agent_status"] = a.get("agent_status", "")
            answer = (a.get("answer") or "").strip()
            entry["refused"] = _is_refusal(entry["agent_status"], answer)
            if entry["refused"]:
                entry["refusal_evidence"] = answer  # full text: classification must be auditable

            trace = a.get("tool_trace", [])
            entry["rerank_calls"] = sum(1 for t in trace if (t.get("args") or {}).get("reranked"))
            entry["retrieval_rounds"] = sum(1 for t in trace if t.get("name") == "retrieve_chunks")

            ask_ids = [c.get("chunk_id", "") for c in a.get("retrieved_chunks", [])
                       if not str(c.get("chunk_id", "")).startswith("web:")]
            ranked_hashes = [chunk_hash_by_id.get(c, "") for c in ask_ids[: args.top_k]]
            if gt:
                entry.update(_ranking_metrics(ranked_hashes, gt))
            rows.append(entry)

    # ── Aggregates per variant ──
    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    def _median(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        mid = len(s) // 2
        return round((s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2), 2)

    metric_names = ("recall_at_5", "recall_at_10", "precision_at_10", "mrr_at_10", "ndcg_at_10")
    variant_metrics: dict[str, dict] = {}
    for variant in variants:
        vrows = [r for r in rows if r.get("variant") == variant]
        answerable = [r for r in vrows if not r.get("expect_refusal")]
        vm: dict = {
            "n_questions": len(vrows),
            "metric_means": {
                name: _mean([r[name] for r in vrows if isinstance(r.get(name), int | float)])
                for name in metric_names
            },
        }
        if variant in ("ask", "ask_rerank"):
            expect_refusal_rows = [r for r in vrows if r.get("expect_refusal")]
            refused_rows = [r for r in vrows if r.get("refused")]
            vm["refusal_recall"] = (
                round(sum(1 for r in expect_refusal_rows if r.get("refused")) / len(expect_refusal_rows), 4)
                if expect_refusal_rows else None
            )
            vm["refusal_precision"] = (
                round(sum(1 for r in refused_rows if r.get("expect_refusal")) / len(refused_rows), 4)
                if refused_rows else None
            )
            vm["false_refusals"] = [r["id"] for r in answerable if r.get("refused")]
            vm["llm_calls"] = {
                "rerank_calls_total": sum(r.get("rerank_calls", 0) for r in vrows),
                "retrieval_rounds_total": sum(r.get("retrieval_rounds", 0) for r in vrows),
            }
            wall = [r["wall_ms"] for r in vrows if isinstance(r.get("wall_ms"), int | float)]
            vm["wall_ms"] = {"mean": _mean(wall), "median": _median(wall), "n": len(wall)}
        else:
            vm["latency_ms"] = {
                stage: {"mean": _mean(v), "median": _median(v), "n": len(v)}
                for stage, v in latencies.get(variant, {}).items()
            }
        # failure attribution over answerable rows
        attr: dict[str, int] = {}
        for r in answerable:
            if r.get("hit_rank") is None:
                key = r.get("failure") or "not_recalled_top10"
            elif r["hit_rank"] > 5:
                key = "recalled_rank6to10"
            else:
                key = "recalled_top5"
            attr[key] = attr.get(key, 0) + 1
        vm["attribution_counts"] = attr
        variant_metrics[variant] = vm

    report = {
        "api": args.api,
        "notebook_id": args.notebook_id,
        "db": str(args.db),
        "top_k": args.top_k,
        "variants_run": variants,
        "n_questions": len(questions),
        "n_resolved": len(resolved),
        "unresolved": unresolved,
        "variant_metrics": variant_metrics,
        "rows": rows,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    out = args.out or Path.cwd() / "retrieval_report.json"
    out.write_text(text, encoding="utf-8")
    _write_markdown(report, out.with_suffix(".md"))
    print(f"Wrote {out} and {out.with_suffix('.md')}")
    return 0


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Retrieval eval (Phase 1 variants)",
        "",
        f"- API: {report['api']} · notebook: {report['notebook_id']} · top_k: {report['top_k']}",
        f"- Variants run: {', '.join(report['variants_run'])}",
        f"- Questions: {report['n_questions']} total, {report['n_resolved']} resolved",
        "",
        "## Metric means by variant (answerable questions)",
        "",
        "| Variant | Recall@5 | Recall@10 | Prec@10 | MRR@10 | nDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    for variant, vm in report["variant_metrics"].items():
        m = vm["metric_means"]
        lines.append(f"| {variant} | {m['recall_at_5']} | {m['recall_at_10']} | "
                     f"{m['precision_at_10']} | {m['mrr_at_10']} | {m['ndcg_at_10']} |")
    lines += ["", "## Refusal and LLM calls (ask variants)", ""]
    for variant in ("ask", "ask_rerank"):
        vm = report["variant_metrics"].get(variant)
        if not vm:
            continue
        lines += [
            f"### {variant}",
            f"- refusal recall: {vm.get('refusal_recall')} · refusal precision: {vm.get('refusal_precision')}",
            f"- false refusals (answerable wrongly refused): {vm.get('false_refusals')}",
            f"- rerank LLM calls (total): {vm['llm_calls']['rerank_calls_total']} · "
            f"retrieval rounds (total): {vm['llm_calls']['retrieval_rounds_total']}",
            f"- wall ms: mean {vm['wall_ms']['mean']} · median {vm['wall_ms']['median']} (n={vm['wall_ms']['n']})",
            "",
        ]
    lines += ["## Latency (search variants, ms)", ""]
    for variant in ("fts", "dense", "rrf"):
        vm = report["variant_metrics"].get(variant)
        if not vm:
            continue
        lines.append(f"### {variant}")
        for stage, v in vm.get("latency_ms", {}).items():
            lines.append(f"- {stage}: mean {v['mean']} · median {v['median']} (n={v['n']})")
        lines.append("")
    lines += ["## Failure attribution by variant (answerable questions)", ""]
    for variant, vm in report["variant_metrics"].items():
        lines.append(f"### {variant}")
        for cause, n in sorted(vm["attribution_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {cause}: {n}")
        lines.append("")
    if report["unresolved"]:
        lines += ["## Ground truth not resolvable (extraction/chunking changed?)", ""]
        for u in report["unresolved"]:
            lines.append(f"- {u['id']}: {u['question']}")
    lines += ["", "## Per-question rows", ""]
    for r in report["rows"]:
        m = " ".join(f"{k}={r[k]}" for k in ("recall_at_10", "hit_rank", "ndcg_at_10") if r.get(k) is not None)
        ref = " REFUSED" if r.get("refused") else ""
        lines.append(f"- `{r['variant']}:{r['id']}` {r['question'][:60]}{ref} {m}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
