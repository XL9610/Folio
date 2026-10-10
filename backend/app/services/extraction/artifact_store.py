"""Extraction artifact store — persist ExtractedDocument to durable files.

Writes under local-data/extraction_artifacts/{document_id}/:
  extracted.json  — metadata, warnings, TOC, slides, segments, structure summary
  content.txt     — full_text
  content.md      — markdown (when present)
  pages.jsonl     — one JSON object per DocumentPage (when pages are present)

The companion extraction_repository writes the SQLite metadata row.
This store only deals with filesystem persistence.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.config import config
from ..pdf_ingestion.models import DocumentPage, TocEntry
from .models import AudioSegment, ExtractedDocument, SlideContent, extracted_structure_summary

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Path descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionArtifactPaths:
    """Absolute paths to all artifact files for one document extraction."""

    root_dir: Path
    extracted_json: Path
    content_txt: Path
    content_md: Path | None
    pages_jsonl: Path | None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _toc_entry_to_dict(entry: TocEntry) -> dict[str, Any]:
    result: dict[str, Any] = {
        "title": entry.title,
        "level": entry.level,
        "page_start": entry.page_start,
        "page_end": entry.page_end,
    }
    if entry.children:
        result["children"] = [_toc_entry_to_dict(c) for c in entry.children]
    return result


def _slide_to_dict(slide: SlideContent) -> dict[str, Any]:
    return {
        "slide_number": slide.slide_number,
        "title": slide.title,
        "body_text": slide.body_text,
        "notes": slide.notes,
        "metadata": slide.metadata,
    }


def _segment_to_dict(segment: AudioSegment) -> dict[str, Any]:
    return {
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "text": segment.text,
        "speaker": segment.speaker,
        "metadata": segment.metadata,
    }


def _document_page_to_json(page: DocumentPage) -> str:
    """Serialize one DocumentPage as a JSON line for pages.jsonl."""
    obj: dict[str, Any] = {
        "page_number": page.page_number,
        "text": page.text,
    }
    if page.metadata:
        obj["metadata"] = page.metadata
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ExtractionArtifactStore:
    """Writes and validates durable extraction checkpoints on disk.

    Usage::

        store = ExtractionArtifactStore()
        paths = store.save(document_id, extracted_doc)
        ok, errors = store.validate(document_id)
        store.delete(document_id)
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is not None:
            self._base = Path(base_dir)
        else:
            self._base = Path(config.local_data_dir) / "extraction_artifacts"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def artifact_dir(self, document_id: str) -> Path:
        """Return the directory that would hold artifacts for *document_id*."""
        return self._base / document_id

    def save(self, document_id: str, document: ExtractedDocument) -> ExtractionArtifactPaths:
        """Persist all extraction artifacts to disk.

        Creates the artifact directory and overwrites any existing artifacts.
        """
        root = self.artifact_dir(document_id)
        root.mkdir(parents=True, exist_ok=True)

        # content.txt
        content_txt = root / "content.txt"
        content_txt.write_text(document.full_text, encoding="utf-8")

        # content.md (optional)
        content_md: Path | None = None
        if document.markdown is not None:
            content_md = root / "content.md"
            content_md.write_text(document.markdown, encoding="utf-8")

        # pages.jsonl (optional)
        pages_jsonl: Path | None = None
        if document.pages:
            pages_jsonl = root / "pages.jsonl"
            with pages_jsonl.open("w", encoding="utf-8") as fh:
                for page in document.pages:
                    fh.write(_document_page_to_json(page) + "\n")

        # extracted.json
        extracted_json = root / "extracted.json"
        extracted_obj = _build_extracted_json(document)
        extracted_json.write_text(
            json.dumps(extracted_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return ExtractionArtifactPaths(
            root_dir=root,
            extracted_json=extracted_json,
            content_txt=content_txt,
            content_md=content_md,
            pages_jsonl=pages_jsonl,
        )

    def validate(self, document_id: str) -> tuple[bool, list[str]]:
        """Check that required artifact files exist and are readable.

        Returns (is_valid, list of error messages for missing/corrupt files).
        """
        errors: list[str] = []
        root = self.artifact_dir(document_id)
        if not root.is_dir():
            errors.append(f"Artifact directory missing: {root}")
            return False, errors

        # Required files
        for name in ("extracted.json", "content.txt"):
            fp = root / name
            if not fp.is_file():
                errors.append(f"Missing required artifact: {fp}")
            else:
                try:
                    fp.read_text(encoding="utf-8")
                except Exception as exc:
                    errors.append(f"Cannot read {fp}: {exc}")

        return len(errors) == 0, errors

    def delete(self, document_id: str) -> None:
        """Remove the artifact directory for *document_id* and all its contents."""
        root = self.artifact_dir(document_id)
        if root.exists():
            shutil.rmtree(root)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_extracted_json(document: ExtractedDocument) -> dict[str, Any]:
    """Build the extracted.json payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "title": document.title,
        "extraction_engine": document.extraction_engine,
        "metadata": document.metadata,
        "warnings": document.warnings,
        "raw_toc": [_toc_entry_to_dict(e) for e in document.raw_toc] if document.raw_toc else [],
        "slides": [_slide_to_dict(s) for s in document.slides] if document.slides else [],
        "segments": [_segment_to_dict(s) for s in document.segments] if document.segments else [],
        "structure_summary": extracted_structure_summary(document),
    }
