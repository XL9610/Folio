#!/usr/bin/env python3
"""Build (or refresh) hierarchical overview summaries for existing documents.

Roadmap Phase 3 backfill. Restartable: documents with a valid cached
hierarchy are skipped unless --force is supplied. Prints per-document
status, group count, LLM-call count, elapsed time, cache hit/miss, and
failure reason. Requires a configured LLM (never sends source content to a
remote model when the existing LLM setting is disabled).

Usage:
    python backend/scripts/build_hierarchical_summaries.py \\
        --notebook-id <id> [--document-id <id>] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.repositories.document_repository import get_document, list_documents  # noqa: E402
from app.services.generation.hierarchical_summary_builder import (  # noqa: E402
    HierarchicalSummaryBuilder,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook-id", required=True, help="Notebook to backfill")
    parser.add_argument("--document-id", default=None,
                        help="Restrict to one document (optional)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even when a valid cached hierarchy exists")
    args = parser.parse_args()

    documents = list_documents(args.notebook_id)
    if args.document_id:
        documents = [d for d in documents if d["id"] == args.document_id]
    if not documents:
        print(f"No documents found in notebook {args.notebook_id}")
        return 1

    builder = HierarchicalSummaryBuilder()
    failed = 0
    for doc in documents:
        if not doc.get("chunk_count"):
            print(f"- {doc['filename']}: skip (no chunks)")
            continue
        if doc.get("status") not in ("completed", "fts_only", "indexed_partial"):
            print(f"- {doc['filename']}: skip (status {doc.get('status')})")
            continue
        t0 = time.monotonic()
        result = builder.build_document(
            notebook_id=args.notebook_id,
            document_id=doc["id"],
            force=args.force,
        )
        elapsed = time.monotonic() - t0
        cache = "hit" if result.cache_hit else "miss"
        line = (
            f"- {doc['filename']}: {result.status} "
            f"(groups={result.group_count}, nodes={result.node_count}, "
            f"llm_calls={result.llm_calls}, cache={cache}, "
            f"{elapsed:.1f}s, mode={result.grouping_mode})"
        )
        if result.reason and result.status != "built":
            line += f" — {result.reason}"
        print(line)
        if result.status not in ("built", "up_to_date", "skipped_llm"):
            failed += 1

    if failed:
        print(f"{failed} document(s) failed to build")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
