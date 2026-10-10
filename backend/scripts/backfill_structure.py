"""Backfill structure_status and related fields for legacy documents.

Documents ingested before the StructureStatus enum was introduced have
structure_status = 'unknown_legacy'.  This script re-processes those
documents to populate the six structure fields from the original PDF.

Usage:
    python -m scripts.backfill_structure [--dry-run] [--doc-id ID] [--pdf-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import get_connection
from app.services.pdf_ingestion.chunker import PDFChunker
from app.services.pdf_ingestion.structure_metadata import (
    build_structure_metadata,
    STRUCTURE_FIELDS,
)
from app.services.pdf_ingestion.structure_status import StructureStatus

logger = logging.getLogger(__name__)


def _is_structure_row(row: dict, field: str) -> bool:
    """Check if a db column belongs to the six structure fields."""
    return field in STRUCTURE_FIELDS


def backfill_document(
    doc: dict,
    pdf_dir: str | None,
    dry_run: bool,
) -> bool:
    """Re-process a single document and update its structure fields.

    Returns True if the document was updated (or would be, in dry-run mode).
    """
    doc_id = doc["id"]
    filename = doc.get("filename", "")
    current_status = doc.get("structure_status", "")

    # Skip documents that already have a known (non-legacy) status
    if current_status and current_status not in (
        StructureStatus.UNKNOWN_LEGACY,
        "",
    ):
        logger.debug("Skipping %s: already has status=%s", doc_id, current_status)
        return False

    # Locate the PDF file
    pdf_path = None
    if pdf_dir:
        candidate = os.path.join(pdf_dir, filename)
        if os.path.isfile(candidate):
            pdf_path = candidate

    if not pdf_path:
        # Try the filename as an absolute path
        if os.path.isabs(filename) and os.path.isfile(filename):
            pdf_path = filename

    if not pdf_path:
        logger.warning(
            "Cannot locate PDF for %s (filename=%s, pdf_dir=%s) — skipping.",
            doc_id, filename, pdf_dir,
        )
        return False

    if dry_run:
        logger.info(
            "[DRY-RUN] Would re-process %s (%s) → status currently '%s'",
            doc_id, filename, current_status,
        )
        return True

    # Re-chunk and re-evaluate structure
    logger.info("Re-processing %s from %s", doc_id, pdf_path)
    try:
        chunker = PDFChunker()
        result = chunker.chunk(pdf_path, document_id=doc_id)
        metadata = build_structure_metadata(result)

        conn = get_connection()
        conn.execute(
            """UPDATE source_documents
               SET structure_status = ?,
                   embedded_toc_present = ?,
                   synthetic_outline_mode = ?,
                   structure_confidence = ?,
                   structure_user_note = ?,
                   structure_llm_instruction = ?
               WHERE id = ?""",
            (
                metadata.structure_status,
                None
                if metadata.embedded_toc_present is None
                else int(bool(metadata.embedded_toc_present)),
                metadata.synthetic_outline_mode,
                metadata.structure_confidence,
                metadata.structure_user_note,
                metadata.structure_llm_instruction,
                doc_id,
            ),
        )
        conn.commit()
        logger.info(
            "Updated %s: status=%s confidence=%s",
            doc_id,
            metadata.structure_status,
            metadata.structure_confidence,
        )
        return True
    except Exception as exc:
        logger.error("Failed to re-process %s: %s", doc_id, exc, exc_info=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill structure_status for legacy documents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing to the database.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Process only a single document by ID.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directory containing original PDF files (looked up by filename).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    conn = get_connection()

    if args.doc_id:
        row = conn.execute(
            "SELECT * FROM source_documents WHERE id = ?", (args.doc_id,)
        ).fetchone()
        if row is None:
            logger.error("Document not found: %s", args.doc_id)
            sys.exit(1)
        docs = [dict(row)]
    else:
        rows = conn.execute(
            "SELECT * FROM source_documents WHERE structure_status = ? OR structure_status = ''",
            (StructureStatus.UNKNOWN_LEGACY,),
        ).fetchall()
        docs = [dict(r) for r in rows]

    if not docs:
        logger.info("No documents with unknown_legacy status found.")
        return

    updated = 0
    for doc in docs:
        if backfill_document(doc, args.pdf_dir, args.dry_run):
            updated += 1

    if args.dry_run:
        logger.info(
            "[DRY-RUN] Would update %d of %d legacy documents.",
            updated, len(docs),
        )
    else:
        logger.info(
            "Backfill complete: updated %d of %d legacy documents.",
            updated, len(docs),
        )


if __name__ == "__main__":
    main()
