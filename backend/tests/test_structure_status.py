"""M2 Tests: StructureStatus enum consolidation.

Tests the new StructureStatus enum as the single source of truth for
document structure reliability classification, plus API field coverage.
"""

from __future__ import annotations

from unittest.mock import patch

from app.services.pdf_ingestion.structure_status import StructureStatus
from app.services.pdf_ingestion.structure_metadata import (
    UNKNOWN_LEGACY_STRUCTURE,
    build_structure_metadata,
    is_reliable_structure_status,
    is_unreliable_structure_status,
)
from app.services.pdf_ingestion.chunker import PDFChunker


# ---------------------------------------------------------------------------
# Unit tests — enum properties
# ---------------------------------------------------------------------------

class TestStructureStatusProperties:
    """Verify enum .is_reliable and .has_any_outline properties."""

    def test_is_reliable_embedded_toc(self):
        """embedded_toc -> is_reliable=True (only this status is reliable)."""
        assert StructureStatus.EMBEDDED_TOC.is_reliable is True

    def test_is_reliable_auto_detected(self):
        """auto_detected_outline -> is_reliable=False."""
        assert StructureStatus.AUTO_DETECTED_OUTLINE.is_reliable is False

    def test_is_unreliable_page_window(self):
        """page_window_outline -> is_reliable=False."""
        assert StructureStatus.PAGE_WINDOW_OUTLINE.is_reliable is False

    def test_is_unreliable_no_outline(self):
        """no_reliable_outline -> is_reliable=False."""
        assert StructureStatus.NO_RELIABLE_OUTLINE.is_reliable is False

    def test_is_unreliable_unknown_legacy(self):
        """unknown_legacy -> is_reliable=False."""
        assert StructureStatus.UNKNOWN_LEGACY.is_reliable is False

    def test_has_any_outline_embedded_toc(self):
        """embedded_toc has outline."""
        assert StructureStatus.EMBEDDED_TOC.has_any_outline is True

    def test_has_any_outline_auto_detected(self):
        """auto_detected_outline has outline."""
        assert StructureStatus.AUTO_DETECTED_OUTLINE.has_any_outline is True

    def test_has_any_outline_page_window(self):
        """page_window_outline has outline."""
        assert StructureStatus.PAGE_WINDOW_OUTLINE.has_any_outline is True

    def test_has_any_outline_no_reliable(self):
        """no_reliable_outline does NOT have outline."""
        assert StructureStatus.NO_RELIABLE_OUTLINE.has_any_outline is False

    def test_has_any_outline_unknown_legacy(self):
        """unknown_legacy does NOT have outline."""
        assert StructureStatus.UNKNOWN_LEGACY.has_any_outline is False

    def test_str_equality(self):
        """Enum values are strings (str, Enum)."""
        assert StructureStatus.EMBEDDED_TOC == "embedded_toc"
        assert StructureStatus.PAGE_WINDOW_OUTLINE == "page_window_outline"
        assert StructureStatus.UNKNOWN_LEGACY == "unknown_legacy"


# ---------------------------------------------------------------------------
# Unit tests — legacy status functions (use the enum via conversion)
# ---------------------------------------------------------------------------

class TestLegacyStatusFunctions:
    """is_unreliable_structure_status / is_reliable_structure_status use enum."""

    def test_unreliable_status_from_string(self):
        """String status converted via enum: page_window_outline is unreliable."""
        assert is_unreliable_structure_status("page_window_outline") is True
        assert is_unreliable_structure_status("no_reliable_outline") is True
        assert is_unreliable_structure_status("unknown_legacy") is True

    def test_unreliable_status_auto_detected(self):
        """auto_detected_outline is no longer in the reliable set."""
        assert is_unreliable_structure_status("auto_detected_outline") is True

    def test_reliable_status_from_string(self):
        """embedded_toc is the only reliable status."""
        assert is_reliable_structure_status("embedded_toc") is True

    def test_reliable_status_auto_detected(self):
        """auto_detected_outline is NOT reliable under the new enum."""
        assert is_reliable_structure_status("auto_detected_outline") is False

    def test_unreliable_none(self):
        """None status defaults to unknown_legacy -> unreliable."""
        assert is_unreliable_structure_status(None) is True

    def test_reliable_none(self):
        """None status -> not reliable."""
        assert is_reliable_structure_status(None) is False

    def test_unreliable_invalid_string(self):
        """Unknown status string -> falls through to unreliable."""
        assert is_unreliable_structure_status("bogus_status") is True

    def test_reliable_invalid_string(self):
        """Unknown status string -> falls through to not reliable."""
        assert is_reliable_structure_status("bogus_status") is False


# ---------------------------------------------------------------------------
# Unit test — UNKNOWN_LEGACY_STRUCTURE default
# ---------------------------------------------------------------------------

class TestUnknownLegacyDefault:
    """Verify the default structure metadata uses the enum."""

    def test_unknown_legacy_default(self):
        """UNKNOWN_LEGACY_STRUCTURE has structure_status == unknown_legacy."""
        assert UNKNOWN_LEGACY_STRUCTURE.structure_status == StructureStatus.UNKNOWN_LEGACY
        assert UNKNOWN_LEGACY_STRUCTURE.structure_status == "unknown_legacy"

    def test_unknown_legacy_has_instruction(self):
        """UNKNOWN_LEGACY_STRUCTURE carries an LLM instruction."""
        assert UNKNOWN_LEGACY_STRUCTURE.structure_llm_instruction
        assert "unknown" in UNKNOWN_LEGACY_STRUCTURE.structure_llm_instruction.lower()


# ---------------------------------------------------------------------------
# Integration tests — full chunking pipeline with real PDFs
# ---------------------------------------------------------------------------

class TestChunkingPipelineEnum:
    """End-to-end: chunk a PDF and verify structure_status uses the enum."""

    def test_embedded_toc_enum(self, pdf_with_toc):
        """PDF with bookmarks -> embedded_toc."""
        chunker = PDFChunker()
        result = chunker.chunk(pdf_with_toc)
        metadata = build_structure_metadata(result)
        assert metadata.structure_status == StructureStatus.EMBEDDED_TOC
        assert metadata.embedded_toc_present is True
        assert metadata.structure_confidence == "high"

    def test_page_window_enum(self, pdf_without_toc):
        """PDF without TOC -> synthetic_outline with page_windows -> page_window_outline."""
        chunker = PDFChunker()
        result = chunker.chunk(pdf_without_toc)
        metadata = build_structure_metadata(result)
        assert metadata.structure_status == StructureStatus.PAGE_WINDOW_OUTLINE
        assert metadata.structure_confidence == "low"

    def test_no_reliable_outline_for_empty_pages(self, pdf_with_broken_toc):
        """PDF with broken/empty pages -> token_fallback -> no_reliable_outline."""
        chunker = PDFChunker()
        result = chunker.chunk(pdf_with_broken_toc)
        metadata = build_structure_metadata(result)
        # token_fallback strategy yields no_reliable_outline status
        assert metadata.structure_status == StructureStatus.NO_RELIABLE_OUTLINE


# ---------------------------------------------------------------------------
# API field coverage test
# ---------------------------------------------------------------------------

class TestAPIStructureFields:
    """Confirm the GET /api/documents/{id} endpoint returns all 6 structure fields."""

    STRUCTURE_FIELDS = {
        "structure_status",
        "embedded_toc_present",
        "synthetic_outline_mode",
        "structure_confidence",
        "structure_user_note",
        "structure_llm_instruction",
    }

    @patch("app.routers.documents_router.get_document_sections")
    @patch("app.routers.documents_router.get_document")
    def test_api_returns_all_structure_fields(
        self, mock_get_document, mock_get_sections
    ):
        """GET /api/documents/{id} includes all 6 structure fields."""
        mock_get_document.return_value = {
            "id": "test-doc-1",
            "filename": "test.pdf",
            "display_title": "Test Document",
            "notebook_id": "nb-1",
            "page_count": 10,
            "chunk_count": 25,
            "status": "completed",
            "chunking_mode": "hierarchical_toc",
            "structure_status": "embedded_toc",
            "embedded_toc_present": True,
            "synthetic_outline_mode": None,
            "structure_confidence": "high",
            "structure_user_note": "Original table of contents",
            "structure_llm_instruction": "This document has a usable embedded PDF outline.",
            "created_at": "2025-01-01T00:00:00",
            "content_hash": "abc123",
        }
        mock_get_sections.return_value = []

        from app.routers.documents_router import get_document_detail
        import asyncio

        # Call the async endpoint function synchronously via asyncio
        result = asyncio.run(get_document_detail("test-doc-1"))

        for field in self.STRUCTURE_FIELDS:
            assert field in result, f"Missing field '{field}' in API response"

        assert result["structure_status"] == "embedded_toc"
        assert result["embedded_toc_present"] is True
        assert result["synthetic_outline_mode"] is None
        assert result["structure_confidence"] == "high"
        assert result["structure_user_note"] == "Original table of contents"
        assert result["structure_llm_instruction"] == (
            "This document has a usable embedded PDF outline."
        )

    @patch("app.routers.documents_router.get_document_sections")
    @patch("app.routers.documents_router.get_document")
    def test_list_endpoint_includes_structure_fields(
        self, mock_get_document, mock_get_sections
    ):
        """GET /api/documents list endpoint includes structure fields."""
        # Test document_structure_fields directly — the list/detail endpoints
        # both use this function to inject structure fields into responses.
        from app.services.pdf_ingestion.structure_metadata import (
            document_structure_fields,
        )
        from app.services.pdf_ingestion.structure_status import StructureStatus

        doc_row = {
            "id": "test-doc-2",
            "filename": "test2.pdf",
            "display_title": "",
            "page_count": 5,
            "chunk_count": 10,
            "status": "completed",
            "notebook_id": "nb-1",
            "content_hash": "def456",
            "structure_status": "page_window_outline",
            "embedded_toc_present": False,
            "synthetic_outline_mode": "page_windows",
            "structure_confidence": "low",
            "structure_user_note": "Page-range outline",
            "structure_llm_instruction": "No reliable chapter structure.",
            "created_at": "2025-01-01T00:00:00",
            "chunking_mode": "synthetic_outline",
        }

        fields = document_structure_fields(doc_row)
        for field in self.STRUCTURE_FIELDS:
            assert field in fields, f"Missing field '{field}' in document_structure_fields"
        assert fields["structure_status"] == StructureStatus.PAGE_WINDOW_OUTLINE
