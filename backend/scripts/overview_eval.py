#!/usr/bin/env python3
"""Overview-specific evaluation for the hierarchical overview index (Phase 3).

Deterministic metrics per overview question (no LLM):

  - exact Recall@10            (existing hash-bound metric)
  - window-aware evidence recall@10: a retrieved chunk counts when it is the
    annotated chunk, an immediate neighboring chunk (chunk_index +-1 in the
    same document), or belongs to an accepted major section recorded in
    ground truth
  - section coverage: fraction of accepted section keys represented in the
    retrieved summary context (hierarchical route: trace `section_keys`;
    6/4 route: retrieved chunks' major sections)
  - citation validity: every returned citation resolves to an existing raw
    chunk in the authorized document set
  - build coverage (per document): accepted body chunks represented by
    exactly one section card / accepted body chunks

Optional frozen LLM judge (--judge; temperature 0, model from the LLM env
config) for:

  - expected-topic coverage: fraction of prospectively annotated topics
    substantively present in the answer
  - unsupported-major-claim rate: major answer claims with no supporting
    raw source passage among their cited provenance

The judge prompt and version are frozen below; judge variance is reported by
running the judge twice per question.

Usage:
    python backend/scripts/overview_eval.py --questions <gt.json> \\
        --notebook-id <id> [--judge] --out <report.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / "local-data" / "rag_companion.db"

if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.generation.hierarchical_summary_builder import (  # noqa: E402
    build_manifest,
)
from app.services.generation.sampling import major_section_key  # noqa: E402
from app.services.llm.client import create_llm_client  # noqa: E402

JUDGE_PROMPT_VERSION = "1"

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a study-companion RAG system. Score the candidate answer against the user question and the cited source passages.

Rules:
- A topic is COVERED only when the answer substantively addresses it: a concrete example, argument, or claim drawn from the source material, not a bare mention of the topic's name.
- An unsupported claim is a major factual assertion about the source material that is NOT supported by any cited passage. Ignore minor stylistic or connective statements. Do not count claims supported by any cited passage, even partially paraphrased.
- Judge the answer as written. Do not use outside knowledge about the source.

Return ONLY valid JSON:
{
  "topic_coverage": {"<topic_id>": true, "<topic_id2>": false},
  "unsupported_claims": [
    {"claim": "..."}
  ]
}"""


def _judge_user_prompt(question: str, answer: str, topics: list[dict],
                       passages: list[dict]) -> str:
    topic_lines = "\n".join(
        f'- {t.get("topic_id", "")}: {t.get("label", "")}'
        for t in topics
    )
    passage_lines = []
    for p in passages:
        loc = p.get("section_path") or ""
        passage_lines.append(f"[{p['ref']}] ({loc})\n{p['text'][:2000]}")
    return (
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Annotated topics to score:\n{topic_lines}\n\n"
        f"Cited source passages:\n" + "\n\n".join(passage_lines)
    )


def _open_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _ask(api: str, question: str, notebook_id: str, document_ids: list[str]) -> dict:
    payload = {
        "notebook_id": notebook_id,
        "query": question,
        "top_k": 10,
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


def _resolve(question: dict, documents: list[dict], con: sqlite3.Connection):
    hashes = [question.get("document_hash", "")]
    hashes += question.get("additional_document_hashes", [])
    hashes = [h for h in hashes if h]
    by_hash = {d.get("content_hash", ""): d for d in documents}
    doc_ids = []
    for h in hashes:
        match = by_hash.get(h)
        if not match:
            return None
        doc_ids.append(match["document_id"])
    return doc_ids


def _chunk_row_by_hash(con: sqlite3.Connection, content_hash: str):
    row = con.execute(
        "SELECT * FROM chunks WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    ).fetchone()
    return dict(row) if row else None


def _section_key_of(row: dict) -> str:
    if not row:
        return ""
    return major_section_key(row.get("section_path", "") or row.get("title", ""))


def _window_hit(
    ranked_rows: list[dict],
    annotated: list[dict],
    accepted_section_keys: set[str],
    accepted_section_paths: set[str],
) -> set[str]:
    """Window-aware hit set: exact, neighbor, or accepted-section match.

    ``accepted_section_keys`` come from topic-group ground truth and match
    on the chunk's major-section key; ``accepted_section_paths`` are derived
    from legacy flat annotations and match on the exact section path.
    """
    hits: set[str] = set()
    for ann in annotated:
        arow = ann["row"]
        if not arow:
            continue
        for r in ranked_rows:
            if r["document_id"] != arow["document_id"]:
                continue
            if r["content_hash"] == arow["content_hash"]:
                hits.add(r["id"])
                continue
            if abs(r["chunk_index"] - arow["chunk_index"]) == 1:
                hits.add(r["id"])
                continue
            key = _section_key_of(r)
            if key and key in accepted_section_keys:
                hits.add(r["id"])
                continue
            path = r.get("section_path", "")
            if path and path in accepted_section_paths:
                hits.add(r["id"])
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--judge", action="store_true",
                        help="Run the frozen LLM judge (2 passes per question)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    con = _open_db(args.db)
    with httpx.Client(timeout=60) as client:
        resp = client.get(f"{args.api}/api/documents",
                          params={"notebook_id": args.notebook_id})
        resp.raise_for_status()
        documents = resp.json()

    judge_client = None
    if args.judge:
        # Frozen judge: temperature 0, model from the LLM env config.
        from app.services.llm import LLMConfig
        judge_cfg = LLMConfig()
        judge_cfg.temperature = 0.0
        judge_client = create_llm_client(judge_cfg)

    rows: list[dict] = []
    for q in questions:
        doc_ids = _resolve(q, documents, con)
        if not doc_ids:
            rows.append({"id": q.get("id"), "unresolved": True})
            continue

        # ── Annotated evidence rows ──
        annotated = []
        for c in q.get("relevant_chunks", []):
            row = _chunk_row_by_hash(con, c.get("chunk_content_hash", ""))
            annotated.append({"hash": c.get("chunk_content_hash", ""), "row": row})
        topics = q.get("topic_groups", []) or []
        accepted_section_keys: set[str] = set()
        for t in topics:
            for k in t.get("accepted_section_keys", []):
                accepted_section_keys.add(k)
        accepted_section_paths: set[str] = set()
        if not accepted_section_keys:
            accepted_section_paths = {
                a["row"].get("section_path", "")
                for a in annotated
                if a["row"] and a["row"].get("section_path")
            }

        t0 = time.monotonic()
        try:
            a = _ask(args.api, q["question"], args.notebook_id, doc_ids)
        except Exception as exc:
            rows.append({"id": q.get("id"), "failure": f"ask failed: {exc}"})
            continue
        entry: dict = {
            "id": q.get("id"),
            "question": q.get("question"),
            "wall_ms": round((time.monotonic() - t0) * 1000, 1),
        }

        trace = a.get("tool_trace", [])
        hier = next((t for t in trace if t.get("name") == "hierarchical_overview"), None)
        entry["hierarchical_status"] = (hier or {}).get("status", "")
        entry["hierarchical_args"] = (hier or {}).get("args", {})

        # ── Retrieved ranked rows (hash -> row) ──
        ask_ids = [c.get("chunk_id", "") for c in a.get("retrieved_chunks", [])
                   if not str(c.get("chunk_id", "")).startswith("web:")]
        # All passages the reader showed the model (evidence budget, e.g. 20
        # for config E), for the judge; R@10 metrics stay capped at 10.
        book_citations = [
            c for c in (a.get("citations") or [])
            if c.get("source_type", "book") == "book"
        ]
        passage_budget = len(book_citations) or 10
        all_passages: list[dict] = []
        for cid in ask_ids[:passage_budget]:
            row = con.execute("SELECT * FROM chunks WHERE id = ?", (cid,)).fetchone()
            if row:
                all_passages.append(dict(row))
        ranked_rows = all_passages[:10]

        # ── Exact recall ──
        gt_hashes = [ann["hash"] for ann in annotated if ann["hash"]]
        ranked_hashes = [r["content_hash"] for r in ranked_rows]
        denom = len(gt_hashes) or 1
        entry["exact_recall_at_10"] = sum(
            1 for h in ranked_hashes if h in gt_hashes
        ) / denom if gt_hashes else None

        # ── Window-aware evidence recall ──
        hits = _window_hit(ranked_rows, annotated, accepted_section_keys,
                           accepted_section_paths)
        entry["window_recall_at_10"] = len(hits) / denom if gt_hashes else None

        # ── Section coverage ──
        accepted_all = accepted_section_keys or accepted_section_paths
        if accepted_all:
            if accepted_section_paths:
                # Legacy flat GT: an accepted section is covered when at
                # least one retrieved evidence chunk lies in it (works for
                # window grouping too — hydrated chunks keep their original
                # section paths).
                represented = {r.get("section_path", "") for r in ranked_rows}
            elif entry["hierarchical_status"] == "ok":
                represented = set(entry["hierarchical_args"].get("section_keys", []))
            else:
                represented = {_section_key_of(r) for r in ranked_rows}
            entry["section_coverage"] = len(
                accepted_all & represented
            ) / len(accepted_all)
        else:
            entry["section_coverage"] = None

        # ── Citation validity ──
        citations = a.get("citations", []) or []
        valid = 0
        for c in citations:
            cid = c.get("chunk_id", "")
            if cid.startswith("web:"):
                valid += 1
                continue
            row = con.execute("SELECT document_id FROM chunks WHERE id = ?", (cid,)).fetchone()
            if row and row[0] in doc_ids:
                valid += 1
        entry["citation_validity"] = (valid / len(citations)) if citations else 1.0
        entry["n_citations"] = len(citations)

        # ── Frozen LLM judge ──
        # Runs for topic groups when present, and for unsupported-major-claim
        # scoring whenever --judge is given (empty topic list is valid).
        if args.judge:
            passages = [{
                "ref": f"S{i}",
                "text": r["text"],
                "section_path": r.get("section_path", ""),
            } for i, r in enumerate(all_passages, 1)]
            runs = []
            for _ in range(2):
                try:
                    resp = judge_client.complete_json(
                        system_prompt=JUDGE_SYSTEM_PROMPT,
                        user_prompt=_judge_user_prompt(
                            q["question"],
                            (a.get("internal_answer") or a.get("answer") or "").strip(),
                            topics, passages,
                        ),
                    )
                    runs.append(resp.data)
                except Exception as exc:
                    runs.append({"judge_error": str(exc)})
            entry["judge_runs"] = runs
            entry["judge_agreement"] = runs[0] == runs[1] if len(runs) == 2 else None

        rows.append(entry)

    # ── Build coverage per document (from the DB, read-only) ──
    # Two distinct numbers:
    #   build_coverage:      accepted body chunks assigned to exactly one
    #                        section/window card (manifest integrity; the
    #                        acceptance gate is 100% by construction)
    #   provenance_coverage: fraction of accepted body chunks reachable as
    #                        evidence via stored card provenance (the LLM
    #                        cites a subset of each batch's chunks)
    build_coverage: dict[str, dict] = {}
    for doc in documents:
        doc_id = doc.get("document_id")
        chunk_rows = con.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        if not chunk_rows:
            continue
        rows_for_doc = [dict(r) for r in chunk_rows]
        manifest = build_manifest(rows_for_doc, doc.get("structure_status", ""))
        assigned: list[str] = []
        for g in manifest.groups:
            assigned.extend(g.chunk_ids)
        accepted = set(manifest.accepted_chunk_ids)
        exact_once = len(set(assigned)) == len(assigned) and set(assigned) == accepted
        section_rows = con.execute(
            "SELECT source_chunk_ids_json FROM document_summary_nodes "
            "WHERE notebook_id = ? AND document_id = ? AND level = 'section'",
            (args.notebook_id, doc_id),
        ).fetchall()
        represented: set[str] = set()
        for (ids_json,) in section_rows:
            represented.update(json.loads(ids_json or "[]"))
        build_coverage[doc_id] = {
            "filename": doc.get("filename", ""),
            "grouping_mode": manifest.grouping_mode,
            "accepted_body_chunks": len(accepted),
            "excluded_chunks": len(manifest.excluded_chunk_ids),
            "section_nodes": len(section_rows),
            "build_coverage": 1.0 if exact_once else None,
            "provenance_coverage": (len(accepted & represented) / len(accepted))
                                   if accepted else None,
        }

    report = {
        "api": args.api,
        "notebook_id": args.notebook_id,
        "judge_prompt_version": JUDGE_PROMPT_VERSION if args.judge else None,
        "rows": rows,
        "build_coverage": build_coverage,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")

    # ── Aggregate summary ──
    def mean(key):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    print(f"n_questions={len(rows)}")
    print(f"exact_recall_at_10: {mean('exact_recall_at_10')}")
    print(f"window_recall_at_10: {mean('window_recall_at_10')}")
    print(f"section_coverage: {mean('section_coverage')}")
    print(f"citation_validity: {mean('citation_validity')}")
    print(f"hierarchical route statuses: " + str(
        {r.get('hierarchical_status') for r in rows}
    ))
    for doc_id, bc in build_coverage.items():
        print(f"build {bc['filename'][:50]}: mode={bc['grouping_mode']} "
              f"sections={bc['section_nodes']} body={bc['accepted_body_chunks']} "
              f"assign={bc['build_coverage']} proven={bc['provenance_coverage']}")
    if args.judge:
        n = sum(1 for r in rows if r.get("judge_runs"))
        agree = sum(1 for r in rows if r.get("judge_agreement") is True)
        print(f"judge: {n} judged, agreement {agree}/{n}")
        covered = 0
        total = 0
        unsupported = []
        for r in rows:
            for run in r.get("judge_runs", []):
                for v in (run.get("topic_coverage") or {}).values():
                    total += 1
                    covered += int(bool(v))
                unsupported.extend(run.get("unsupported_claims") or [])
        print(f"expected-topic coverage (judge): {round(covered/total, 4) if total else None}")
        print(f"unsupported-major-claims (judge): {len(unsupported)}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
