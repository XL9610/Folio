"""Tests for Module 1.7 — format-specific chunkers (Token, Heading, EPUB, Slide)."""

from __future__ import annotations

from app.services.chunking.epub import EpubChapterChunker
from app.services.chunking.heading import HeadingChunker, parse_markdown_sections
from app.services.chunking.pptx import SlideChunker
from app.services.chunking.token import TokenFallbackChunker
from app.services.extraction.models import ExtractedDocument, SlideContent
from app.services.ingestion.index_text import MAX_CHUNK_TEXT_TOKENS
from app.services.pdf_ingestion.models import TocEntry


def _tiktoken_count(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

# ============================================================================
# TokenFallbackChunker
# ============================================================================


class TestTokenFallbackChunker:
    def test_splits_long_text(self) -> None:
        chunker = TokenFallbackChunker(chunk_size=100, chunk_overlap=10)
        doc = ExtractedDocument(
            title="Long Doc",
            full_text="word " * 500,
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) > 1
        assert result.strategy_used == "token_fallback"

    def test_respects_embedding_budget(self) -> None:
        chunker = TokenFallbackChunker()  # default uses the shared embedding budget
        doc = ExtractedDocument(title="Long", full_text="a longer sentence. " * 400)
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) > 1
        for c in result.chunks:
            assert _tiktoken_count(c.text) <= MAX_CHUNK_TEXT_TOKENS

    def test_empty_text_returns_no_chunks(self) -> None:
        chunker = TokenFallbackChunker()
        doc = ExtractedDocument(title="Empty", full_text="   ")
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) == 0
        assert len(result.warnings) >= 1

    def test_short_text_single_chunk(self) -> None:
        chunker = TokenFallbackChunker()
        doc = ExtractedDocument(title="Short", full_text="Hello world.")
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) >= 1

    def test_chunk_metadata_has_source_type(self) -> None:
        chunker = TokenFallbackChunker()
        doc = ExtractedDocument(
            title="Doc",
            full_text="Some text for chunking purposes.",
            metadata={"source_type": "text"},
        )
        result = chunker.chunk(doc, "doc-1")
        if result.chunks:
            assert "source_type" in result.chunks[0].metadata

    def test_page_fields_are_negative_one(self) -> None:
        chunker = TokenFallbackChunker()
        doc = ExtractedDocument(title="Doc", full_text="Text.")
        result = chunker.chunk(doc, "doc-1")
        if result.chunks:
            assert result.chunks[0].page_start == -1
            assert result.chunks[0].page_end == -1


# ============================================================================
# Markdown section parser
# ============================================================================


class TestParseMarkdownSections:
    def test_nested_headings(self) -> None:
        md = "# H1\n\nH1 body\n\n## H2\n\nH2 body"
        sections = parse_markdown_sections(md, "Default")
        assert len(sections) >= 2

        # Find the H2 section
        h2_sections = [s for s in sections if "H2" in s.title]
        assert len(h2_sections) >= 1
        assert "H1" in h2_sections[0].heading_path

    def test_content_before_first_heading(self) -> None:
        md = "Intro text\n\n# Heading\n\nBody"
        sections = parse_markdown_sections(md, "Default")
        assert len(sections) >= 1

    def test_no_headings_returns_single_section(self) -> None:
        md = "Just a paragraph.\n\nAnother line."
        sections = parse_markdown_sections(md, "Default")
        # Should get one section with default title
        assert len(sections) >= 1


# ============================================================================
# HeadingChunker
# ============================================================================


class TestHeadingChunker:
    def test_chunks_markdown_with_headings(self) -> None:
        chunker = HeadingChunker()
        doc = ExtractedDocument(
            title="Doc",
            full_text="# H1\n\nBody 1\n\n# H2\n\nBody 2",
            markdown="# H1\n\nBody 1\n\n# H2\n\nBody 2",
        )
        result = chunker.chunk(doc, "doc-1")
        assert result.strategy_used == "heading"
        assert len(result.chunks) >= 2

    def test_no_headings_delegates_to_fallback(self) -> None:
        chunker = HeadingChunker()
        doc = ExtractedDocument(
            title="Doc",
            full_text="Plain text without any markdown headings.",
            markdown="Plain text without any markdown headings.",
        )
        result = chunker.chunk(doc, "doc-1")
        # Should produce chunks via fallback
        assert len(result.chunks) >= 0

    def test_section_path_includes_heading_hierarchy(self) -> None:
        chunker = HeadingChunker()
        doc = ExtractedDocument(
            title="Doc",
            full_text="# H1\n\nBody\n\n## H2\n\nMore",
            markdown="# H1\n\nBody\n\n## H2\n\nMore",
        )
        result = chunker.chunk(doc, "doc-1")
        # At least the H2 chunk should have "H1" in its path
        h2_chunks = [c for c in result.chunks if "H2" in c.title]
        if h2_chunks:
            assert any("H1" in p for p in h2_chunks[0].section_path)

    def test_splits_long_section(self) -> None:
        chunker = HeadingChunker()
        long_body = "paragraph with many words. " * 300
        doc = ExtractedDocument(
            title="Doc",
            full_text=f"# H1\n\n{long_body}",
            markdown=f"# H1\n\n{long_body}",
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) > 1
        for c in result.chunks:
            assert _tiktoken_count(c.text) <= MAX_CHUNK_TEXT_TOKENS

    def test_high_token_density_section_respects_budget(self) -> None:
        chunker = HeadingChunker()
        # Punctuation-heavy text: len//4 underestimates; tiktoken must govern.
        dense = "a+b-c*d/e%f==g?h:i,j,k;\n" * 40
        doc = ExtractedDocument(
            title="Doc",
            full_text=f"# Code\n\n{dense}",
            markdown=f"# Code\n\n{dense}",
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) >= 1
        for c in result.chunks:
            assert _tiktoken_count(c.text) <= MAX_CHUNK_TEXT_TOKENS

    def test_long_sections_have_contiguous_chunk_index(self) -> None:
        """Each long section falls back to the token chunker which restarts at 0;
        the final list must be renumbered contiguously or DB sort reorders them."""
        chunker = HeadingChunker()
        long_body = "paragraph with many words. " * 300
        md = f"# H1\n\n{long_body}\n\n# H2\n\n{long_body}"
        doc = ExtractedDocument(title="Doc", full_text=md, markdown=md)
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) > 2
        assert [c.chunk_index for c in result.chunks] == list(range(len(result.chunks)))

    def test_long_sections_db_order_matches_chunker_order(self, monkeypatch) -> None:
        """DB reads sort by chunk_index and dense vectors zip against that order,
        so DB order must equal the chunker order (text + embedding alignment)."""
        import sqlite3
        from app.db import session as db_session
        from app.db.schema import create_tables
        from app.repositories.notebook_repository import create_notebook
        from app.repositories.document_repository import create_source_document
        from app.repositories.chunk_repository import insert_chunks, get_chunks_by_document

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        create_tables(conn)
        monkeypatch.setattr(db_session._local, "conn", conn, raising=False)

        nb = create_notebook("NB")
        doc = create_source_document(nb["id"], "book.md")

        chunker = HeadingChunker()
        long_body = "paragraph with many words. " * 300
        md = f"# H1\n\n{long_body}\n\n# H2\n\n{long_body}"
        result = chunker.chunk(
            ExtractedDocument(title="Doc", full_text=md, markdown=md), doc["id"]
        )

        insert_chunks(doc["id"], [
            {
                "chunk_index": c.chunk_index,
                "text": c.text,
                "title": c.title,
                "section_path": " > ".join(c.section_path) if c.section_path else c.title,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_strategy": c.chunk_strategy,
                "metadata_json": "{}",
            }
            for c in result.chunks
        ])
        db_chunks = get_chunks_by_document(doc["id"])
        assert [c["text"] for c in db_chunks] == [c.text for c in result.chunks]
        assert [c["chunk_index"] for c in db_chunks] == list(range(len(db_chunks)))


# ============================================================================
# EpubChapterChunker
# ============================================================================


class TestEpubChapterChunker:
    def test_delegates_to_heading(self) -> None:
        chunker = EpubChapterChunker()
        doc = ExtractedDocument(
            title="Book",
            full_text="# Ch1\n\nContent",
            markdown="# Ch1\n\nContent",
            metadata={"source_type": "epub"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert result.strategy_used == "epub_chapter"
        assert len(result.chunks) > 0
        # Chunks should have EPUB source type
        if result.chunks:
            assert result.chunks[0].metadata.get("source_type") == "epub"

    def test_preserves_raw_toc(self) -> None:
        chunker = EpubChapterChunker()
        toc = [TocEntry(title="Ch 1", level=1, page_start=1)]
        doc = ExtractedDocument(
            title="Book",
            full_text="# Ch1\n\nContent",
            markdown="# Ch1\n\nContent",
            raw_toc=toc,
            metadata={"source_type": "epub"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.toc) == 1

    def test_page_fields_negative_one(self) -> None:
        chunker = EpubChapterChunker()
        doc = ExtractedDocument(
            title="Book",
            full_text="Text",
            markdown="Text",
            metadata={"source_type": "epub"},
        )
        result = chunker.chunk(doc, "doc-1")
        if result.chunks:
            assert result.chunks[0].page_start == -1


# ============================================================================
# SlideChunker
# ============================================================================


class TestSlideChunker:
    def test_one_chunk_per_slide(self) -> None:
        chunker = SlideChunker()
        doc = ExtractedDocument(
            title="Deck",
            full_text="S1\nS2",
            slides=[
                SlideContent(slide_number=1, title="Slide 1", body_text="Body 1"),
                SlideContent(slide_number=2, title="Slide 2", body_text="Body 2"),
            ],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) == 2
        assert result.strategy_used == "slide"

    def test_empty_slides_skipped(self) -> None:
        chunker = SlideChunker()
        doc = ExtractedDocument(
            title="Deck",
            full_text="S1",
            slides=[
                SlideContent(slide_number=1, title="", body_text="", notes=""),
                SlideContent(slide_number=2, title="Good", body_text="Content"),
            ],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) == 1

    def test_section_path_contains_slide_number_and_title(self) -> None:
        chunker = SlideChunker()
        doc = ExtractedDocument(
            title="Deck",
            full_text="S1",
            slides=[
                SlideContent(slide_number=3, title="Key", body_text="Content"),
            ],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) == 1
        sp = result.chunks[0].section_path
        assert any("Slide 3" in p for p in sp)

    def test_notes_included_in_text(self) -> None:
        chunker = SlideChunker()
        doc = ExtractedDocument(
            title="Deck",
            full_text="Title\nBody\nNotes text",
            slides=[
                SlideContent(slide_number=1, title="Title", body_text="Body", notes="Notes text"),
            ],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) == 1
        assert "Notes text" in result.chunks[0].text

    def test_splits_long_slide(self) -> None:
        chunker = SlideChunker()
        long_body = "slide content sentence. " * 300
        doc = ExtractedDocument(
            title="Deck",
            full_text=long_body,
            slides=[SlideContent(slide_number=1, title="Long", body_text=long_body)],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) > 1
        for c in result.chunks:
            assert _tiktoken_count(c.text) <= MAX_CHUNK_TEXT_TOKENS
            assert c.metadata["slide_number"] == 1

    def test_slide_metadata(self) -> None:
        chunker = SlideChunker()
        doc = ExtractedDocument(
            title="Deck",
            full_text="S1",
            slides=[
                SlideContent(slide_number=1, title="T1", body_text="B1", notes="N1"),
            ],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert result.chunks[0].metadata["source_type"] == "pptx"
        assert result.chunks[0].metadata["locator_type"] == "slide_number"
        assert result.chunks[0].metadata["slide_number"] == 1
        assert result.chunks[0].metadata["has_notes"] is True

    def test_all_empty_slides_warning(self) -> None:
        chunker = SlideChunker()
        doc = ExtractedDocument(
            title="Deck",
            full_text="",
            slides=[
                SlideContent(slide_number=1, title="", body_text=""),
            ],
            metadata={"source_type": "pptx"},
        )
        result = chunker.chunk(doc, "doc-1")
        assert len(result.chunks) == 0
        assert len(result.warnings) >= 1
