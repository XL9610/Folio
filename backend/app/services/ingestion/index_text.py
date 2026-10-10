"""Index text builder — deterministic context-aware indexing text.

Provides a pure helper that converts document title, section path, and
canonical chunk text into deterministic index-time text with a metadata
context prefix. No repository, embedder, tokenizer, dense index, or FastAPI
imports.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTEXT_MODE_METADATA = "metadata"

# Max chunk text tokens (cl100k) that still fit the bundled bge-small-en-v1.5
# embedder once build_index_text prepends the [title > section] context prefix.
# bge-small-en-v1.5 truncates at 512 WordPiece tokens (~380-420 cl100k); 320
# leaves headroom for the prefix so chunks embed without silent truncation.
MAX_CHUNK_TEXT_TOKENS = 320

# Cap the metadata context prefix so a pathological title/section_path cannot
# consume the whole embedding window. The ingestion service enforces the hard
# 512-token ceiling on the final indexed_text against the bundled BGE tokenizer.
MAX_CONTEXT_PREFIX_CHARS = 160


@dataclass(frozen=True)
class IndexText:
    """Result of building index-time text with deterministic metadata context."""

    context_prefix: str
    context_mode: str
    indexed_text: str


def build_index_text(
    document_title: str,
    section_path: str,
    chunk_text: str,
) -> IndexText:
    """Build deterministic index-time text with a metadata context prefix.

    When *section_path* is non-empty after trimming, the returned
    ``indexed_text`` is ``[<document_title> > <section_path>]\\n\\n<chunk_text>``.
    Otherwise the original *chunk_text* is returned unchanged and
    *context_prefix* is the empty string.

    Only *document_title* and *section_path* are trimmed.  *chunk_text* is
    preserved exactly.
    """
    trimmed_title = document_title.strip()
    trimmed_path = section_path.strip()

    if not trimmed_path:
        return IndexText(
            context_prefix="",
            context_mode=CONTEXT_MODE_METADATA,
            indexed_text=chunk_text,
        )

    context_prefix = f"[{trimmed_title} > {trimmed_path}]"
    if len(context_prefix) > MAX_CONTEXT_PREFIX_CHARS:
        context_prefix = context_prefix[: MAX_CONTEXT_PREFIX_CHARS - 1] + "…"
    indexed_text = f"{context_prefix}\n\n{chunk_text}"

    return IndexText(
        context_prefix=context_prefix,
        context_mode=CONTEXT_MODE_METADATA,
        indexed_text=indexed_text,
    )


def compose_indexed_text(context_prefix: str, chunk_text: str) -> str:
    """Reconstruct index-time text from a stored prefix and canonical chunk text.

    Lets keyword backends and rebuild paths derive indexed text from
    persisted fields without re-computing the prefix.
    """
    if not context_prefix.strip():
        return chunk_text
    return f"{context_prefix}\n\n{chunk_text}"
