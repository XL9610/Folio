"""Token fallback chunker — final fallback that splits full_text by token count."""

from __future__ import annotations

from ..extraction.models import ExtractedDocument
from ..pdf_ingestion.models import Chunk, ChunkingResult
from ..ingestion.index_text import MAX_CHUNK_TEXT_TOKENS
from .base import DocumentChunker


class TokenFallbackChunker(DocumentChunker):
    """Chunk full_text using tiktoken-based token counting.

    This is the final fallback for any text that has no better structure.
    page_start and page_end are set to -1 for non-page sources.
    """

    strategy_name = "token_fallback"

    def __init__(
        self,
        tokenizer_name: str = "cl100k_base",
        chunk_size: int = MAX_CHUNK_TEXT_TOKENS,
        chunk_overlap: int = 40,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # DocumentChunker interface
    # ------------------------------------------------------------------

    def can_chunk(self, document: ExtractedDocument) -> bool:
        return bool(document.full_text.strip())

    def chunk(self, document: ExtractedDocument, document_id: str) -> ChunkingResult:
        text = document.full_text
        if not text.strip():
            return ChunkingResult(
                document_id=document_id,
                strategy_used=self.strategy_name,
                warnings=["No text content to chunk"],
            )

        # Tokenize and split
        sections = _split_text_with_overlap(
            text,
            tokenizer_name=self.tokenizer_name,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        chunks: list[Chunk] = []
        for idx, section_text in enumerate(sections):
            chunks.append(
                Chunk(
                    text=section_text,
                    document_id=document_id,
                    chunk_index=idx,
                    title=document.title,
                    section_path=[document.title] if document.title else [],
                    page_start=-1,
                    page_end=-1,
                    chunk_strategy=self.strategy_name,
                    metadata={"source_type": document.metadata.get("source_type", "unknown")},
                )
            )

        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            strategy_used=self.strategy_name,
            first_chunk_title=document.title,
        )


# ---------------------------------------------------------------------------
# Token split helper
# ---------------------------------------------------------------------------


def _split_text_with_overlap(
    text: str,
    tokenizer_name: str = "cl100k_base",
    chunk_size: int = MAX_CHUNK_TEXT_TOKENS,
    overlap: int = 40,
) -> list[str]:
    """Split text into overlapping sections by token count."""
    import tiktoken

    try:
        enc = tiktoken.get_encoding(tokenizer_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    sections: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        section_tokens = tokens[start:end]
        sections.append(enc.decode(section_tokens))
        if end >= len(tokens):
            break
        start = end - overlap
        if start <= 0:
            start = end

    return sections


def count_tokens(text: str, tokenizer_name: str = "cl100k_base") -> int:
    """Count tokens in *text* using tiktoken (accurate for code/URLs/rare tokens)."""
    import tiktoken

    try:
        enc = tiktoken.get_encoding(tokenizer_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
