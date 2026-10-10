"""Generalized summary generator — LLM synthesizes document summaries.

Supports three scope layers (page range, TOC match, semantic) and
three length/style dimensions. Chunk sampling keeps LLM context
manageable for large documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .scope_resolver import ScopeResolver
from ..retrieval.base import RetrievedChunk
from ..qa.types import make_book_citation
from ._title_utils import clean_filename
from .sampling import sample_body_chunks

logger = logging.getLogger(__name__)

_PAGE_RANGE_TITLE_RE = re.compile(
    r"(?:p\.?|pages?|pp\.?)\s*(\d+)\s*[-–to]+\s*(\d+)",
    re.IGNORECASE,
)


LENGTH_GUIDANCE = {
    "short": "Produce a 100-200 word summary covering the 2-3 most important points.",
    "medium": "Produce a 300-500 word summary with 4-6 main points and brief examples.",
    "detailed": "Produce a comprehensive 600-1000 word summary with subsections, examples, and key takeaways.",
}

STYLE_GUIDANCE = {
    "prose": "Write in flowing paragraphs. Avoid bullet lists.",
    "bullets": "Use bullet lists organized by topic. Keep prose to a minimum.",
    "structured": "Use markdown subheadings (### Topic) for each major area, with short bullet lists under each.",
}

SUMMARY_SYSTEM_PROMPT = """You are summarizing source material for a student. Use only the provided passages — do not invent content. Cite specific passages with [1], [2] format.

{length_guidance}
{style_guidance}

If the material is sparse or off-topic relative to the user's scope, say so honestly rather than padding.  When the material covers multiple distinct topics, organize the summary by topic rather than by passage order.

At the end, add a "---" separator line followed by "**Scope coverage:** <note about what was covered and what may be missing>".  Use double-asterisk **bold** for topic labels and key terms."""


def _unresolved_scope_message(meta: dict) -> str:
    if meta.get("scope_resolved", True) is not False:
        return ""
    return (
        meta.get("unresolved_reason")
        or "The requested chapter or section could not be matched. Use a page range or topic keyword instead."
    )


class SummaryGenerator:
    """Generate markdown summaries from document chunks using LLM synthesis."""

    def __init__(self, llm_client, scope_resolver: ScopeResolver):
        self._llm = llm_client
        self._resolver = scope_resolver

    def generate(
        self,
        notebook_id: str,
        document_id: str,
        scope: str,
        length: str = "medium",
        style: str = "structured",
        doc_filename: str = "",
    ) -> dict[str, Any]:
        """Generate a summary for the given scope.

        Returns:
            dict with keys: title, markdown, sources (chunk_ids),
            stats (chunk_count, scope_layer, scope).
        """
        # Resolve scope to chunks
        result = self._resolver.resolve(document_id, notebook_id, scope)
        chunks = result.chunks
        meta = result.metadata

        if not chunks:
            message = _unresolved_scope_message(meta)
            return {
                "title": self._make_title(scope, doc_filename, matched_sections=[]),
                "markdown": message or "No relevant content was found in the selected scope.",
                "sources": [],
                "stats": {
                    "chunk_count": 0,
                    "scope_layer": meta.get("layer", "unknown"),
                    "scope_resolved": meta.get("scope_resolved", True),
                    "unresolved_reason": message or meta.get("unresolved_reason", ""),
                },
            }

        # Sample chunks if too many (body-aware distributed sampling)
        source_chunks = sample_body_chunks(chunks, max_chunks=25)

        # Build prompts
        system_prompt = SUMMARY_SYSTEM_PROMPT.format(
            length_guidance=LENGTH_GUIDANCE.get(length, LENGTH_GUIDANCE["medium"]),
            style_guidance=STYLE_GUIDANCE.get(style, STYLE_GUIDANCE["structured"]),
        )
        user_prompt = self._build_user_prompt(source_chunks, scope, doc_filename)

        max_tokens = 2048 if length == "detailed" else 1024
        try:
            markdown = self._llm.complete_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Summary generation failed: %s", exc)
            raise

        matched_sections = meta.get("matched_sections", [])
        return {
            "title": self._make_title(scope, doc_filename, matched_sections),
            "markdown": markdown,
            "sources": [
                make_book_citation(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id or document_id,
                    section_path=c.section_path or c.title or "",
                    page_start=c.page_start,
                    page_end=c.page_end,
                    title=doc_filename,
                    ref_label=f"[{i}]",
                    text_snippet=(c.text or "")[:120],
                )
                for i, c in enumerate(source_chunks, 1)
            ],
            "stats": {
                "chunk_count": len(chunks),
                "sampled_count": len(source_chunks),
                "scope_layer": meta.get("layer", "unknown"),
                "scope": scope,
                "page_start": meta.get("page_start", -1),
                "page_end": meta.get("page_end", -1),
            },
        }

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        chunks: list[RetrievedChunk],
        scope: str,
        doc_filename: str,
    ) -> str:
        lines = []
        scope_line = f"Scope: {scope}" if scope else "Scope: whole document"
        lines.append(f"Document: {doc_filename or '(untitled)'}")
        lines.append(scope_line)
        lines.append("")
        lines.append("Source passages:")
        lines.append("")

        for i, c in enumerate(chunks, 1):
            loc = f"p.{c.page_start}" if c.page_start > 0 else ""
            sec = c.section_path or ""
            header = f"[{i}] {sec}"
            if loc:
                header += f" ({loc})"
            lines.append(header)
            lines.append(c.text[:2000])
            lines.append("---")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Title helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_title(
        scope: str,
        doc_filename: str,
        matched_sections: list[str] | None = None,
    ) -> str:
        """Produce a human-readable title."""
        base = clean_filename(doc_filename)

        if not scope:
            return f"{base} — Overview"

        # Page range
        m = _PAGE_RANGE_TITLE_RE.search(scope)
        if m:
            return f"{base}, pp. {m.group(1)}-{m.group(2)}"

        # TOC match
        if matched_sections and len(matched_sections) == 1:
            return matched_sections[0]

        if matched_sections and len(matched_sections) > 1:
            return matched_sections[0]

        # Semantic / generic
        return f"{base} — {scope}"
