"""Heading chunker — splits markdown by ATX headings, falling back to token split."""

from __future__ import annotations

from dataclasses import dataclass

from ..extraction.models import ExtractedDocument
from ..pdf_ingestion.models import Chunk, ChunkingResult
from ..ingestion.index_text import MAX_CHUNK_TEXT_TOKENS
from .base import DocumentChunker
from .token import TokenFallbackChunker, count_tokens


@dataclass(frozen=True)
class MarkdownSection:
    """A section of markdown bounded by a heading and its body text."""

    heading_path: list[str]
    title: str
    body: str
    start_line: int
    end_line: int


class HeadingChunker(DocumentChunker):
    """Chunk markdown by ATX heading hierarchy.

    - Delegates overlong or no-heading content to TokenFallbackChunker.
    - Each section becomes a Chunk unless it exceeds max_section_tokens.
    """

    strategy_name = "heading"

    def __init__(
        self,
        fallback_chunker: TokenFallbackChunker | None = None,
        max_section_tokens: int = MAX_CHUNK_TEXT_TOKENS,
    ) -> None:
        self._fallback = fallback_chunker or TokenFallbackChunker()
        self.max_section_tokens = max_section_tokens

    # ------------------------------------------------------------------
    # DocumentChunker interface
    # ------------------------------------------------------------------

    def can_chunk(self, document: ExtractedDocument) -> bool:
        return document.markdown is not None and bool(document.markdown.strip())

    def chunk(self, document: ExtractedDocument, document_id: str) -> ChunkingResult:
        markdown = document.markdown or ""
        if not markdown.strip():
            return self._fallback.chunk(document, document_id)

        sections = _parse_markdown_sections(markdown, document.title)
        if not sections:
            return self._fallback.chunk(document, document_id)

        chunks: list[Chunk] = []
        for idx, section in enumerate(sections):
            section_text = f"{section.title}\n\n{section.body}" if section.title else section.body
            if not section_text.strip():
                continue

            # If section is too long, sub-chunk with token fallback
            if _estimate_tokens(section_text) > self.max_section_tokens:
                sub_chunks = self._fallback.chunk(
                    ExtractedDocument(
                        title=section.title or document.title,
                        full_text=section_text,
                        metadata=document.metadata,
                    ),
                    document_id,
                )
                # Re-label sub-chunks with this section's path
                for sc in sub_chunks.chunks:
                    sc.section_path = section.heading_path
                    sc.chunk_strategy = self.strategy_name
                    chunks.append(sc)
            else:
                chunks.append(
                    Chunk(
                        text=section_text,
                        document_id=document_id,
                        chunk_index=idx,
                        title=section.title or document.title,
                        section_path=section.heading_path,
                        page_start=-1,
                        page_end=-1,
                        chunk_strategy=self.strategy_name,
                        metadata={
                            "source_type": document.metadata.get("source_type", "unknown"),
                        },
                    )
                )

        # Long sections fall back to TokenFallbackChunker, which restarts its own
        # chunk_index at 0 per section. DB reads sort by chunk_index and dense
        # vectors are zipped positionally against that order, so reset indices
        # would pair vectors to the wrong chunks. Renumber contiguously.
        for new_idx, chunk in enumerate(chunks):
            chunk.chunk_index = new_idx

        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            strategy_used=self.strategy_name,
            first_chunk_title=chunks[0].title if chunks else document.title,
        )


# ---------------------------------------------------------------------------
# Markdown section parser
# ---------------------------------------------------------------------------


def _parse_markdown_sections(markdown: str, default_title: str) -> list[MarkdownSection]:
    """Parse ATX headings and group body text into sections."""
    import re

    lines = markdown.split("\n")
    sections: list[MarkdownSection] = []
    heading_stack: list[str] = []
    current_title = ""
    current_body: list[str] = []
    start_line = 0

    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            # Save previous section
            body_text = "\n".join(current_body).strip()
            if body_text or current_title:
                sections.append(
                    MarkdownSection(
                        heading_path=list(heading_stack),
                        title=current_title or default_title,
                        body=body_text,
                        start_line=start_line,
                        end_line=i,
                    )
                )

            level = len(match.group(1))
            title = match.group(2).strip()

            # Update heading stack
            heading_stack = [h for h in heading_stack if not h.startswith("#" * (level + 1))]
            # Clean stack at this level
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)

            current_title = title
            current_body = []
            start_line = i
        else:
            current_body.append(line)

    # Final section
    body_text = "\n".join(current_body).strip()
    if body_text or current_title:
        sections.append(
            MarkdownSection(
                heading_path=list(heading_stack),
                title=current_title or default_title,
                body=body_text,
                start_line=start_line,
                end_line=len(lines),
            )
        )

    return sections


def _estimate_tokens(text: str) -> int:
    """Token count using tiktoken (accurate, not the ~4 chars/token heuristic)."""
    return count_tokens(text)


def parse_markdown_sections(markdown: str, default_title: str) -> list[MarkdownSection]:
    """Public helper — parse markdown text into MarkdownSection list."""
    return _parse_markdown_sections(markdown, default_title)
