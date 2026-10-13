#!/usr/bin/env python3
"""Gather blind human-review material for the unsupported-claim gate.

For each overview question: runs /api/ask (capturing the exact answer,
retrieved chunk texts in the order the generator saw them, and citation
labels), then runs the frozen judge twice on those exact passages. The
output feeds the blind review sheet — the backend's feature flag
determines which configuration is captured, so run this once with the
flag off and once with it on.

Usage:
    python backend/scripts/overview_review_gather.py \\
        --questions <gt.json> --notebook-id <id> --out <capture.json>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / "local-data" / "rag_companion.db"

if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services.llm import LLMConfig  # noqa: E402
from app.services.llm.client import create_llm_client  # noqa: E402
from scripts.overview_eval import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    _judge_user_prompt,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    with httpx.Client(timeout=60) as client:
        resp = client.get(f"{args.api}/api/documents",
                          params={"notebook_id": args.notebook_id})
        resp.raise_for_status()
        documents = resp.json()
    by_hash = {d.get("content_hash", ""): d for d in documents}

    cfg = LLMConfig()
    cfg.temperature = 0.0
    judge = create_llm_client(cfg)

    captured = []
    for q in questions:
        hashes = [q.get("document_hash", "")]
        hashes += q.get("additional_document_hashes", [])
        doc_ids = []
        for h in hashes:
            m = by_hash.get(h)
            if m:
                doc_ids.append(m["document_id"])
        a = _ask(args.api, q["question"], args.notebook_id, doc_ids)
        answer = (a.get("answer") or "").strip()
        ask_ids = [c.get("chunk_id", "") for c in a.get("retrieved_chunks", [])
                   if not str(c.get("chunk_id", "")).startswith("web:")]
        passages = []
        for i, cid in enumerate(ask_ids[:10], 1):
            row = con.execute("SELECT * FROM chunks WHERE id = ?", (cid,)).fetchone()
            if row:
                r = dict(row)
                passages.append({
                    "ref": f"S{i}",
                    "text": r["text"],
                    "section_path": r.get("section_path", ""),
                    "page_start": r.get("page_start", -1),
                    "page_end": r.get("page_end", -1),
                })
        topics = q.get("topic_groups", []) or []
        judge_runs = []
        for _ in range(2):
            try:
                jr = judge.complete_json(
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_prompt=_judge_user_prompt(
                        q["question"], answer, topics, passages,
                    ),
                )
                judge_runs.append(jr.data)
            except Exception as exc:
                judge_runs.append({"judge_error": str(exc)})
        captured.append({
            "id": q.get("id"),
            "question": q.get("question"),
            "answer": answer,
            "agent_status": a.get("agent_status"),
            "hierarchical": next(
                (t for t in a.get("tool_trace", [])
                 if t.get("name") == "hierarchical_overview"), None,
            ),
            "passages": passages,
            "judge_runs": judge_runs,
        })

    args.out.write_text(json.dumps(captured, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"captured {len(captured)} questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
