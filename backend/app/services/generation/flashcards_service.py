"""Flashcards generator — LLM extracts term/definition pairs from chunks."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from .scope_resolver import ScopeResolver
from ..retrieval.base import RetrievedChunk
from ..qa.types import collect_book_citations
from ._title_utils import clean_filename
from .sampling import (
    sample_body_chunks,
    is_body_chunk,
    group_chunks_by_major_section,
    sample_section_chunks,
    allocate_flashcard_budget,
    major_section_key,
)

logger = logging.getLogger(__name__)


FLASHCARDS_SYSTEM_PROMPT = """You are creating study flashcards from textbook material. Extract key terms, concepts, names, and events that a student should memorize. Each flashcard has a clear "term" (front) and a concise "definition" (back).

Rules:
1. Use only the provided passages — do not invent terms or definitions.
2. Each term should be atomic: one concept, person, event, or definition target.
3. Each definition should be 1-3 sentences, self-contained, grounded in the passage text.
4. Reference the passage number that supports each card via "source": <passage_number>.
5. Skip trivial terms (e.g., "the", "music"). Skip terms that the passages mention but do not define or describe.
6. Aim for the requested count. Generate fewer if the material does not support that many distinct, study-worthy terms — do not pad with weak cards.

Output ONLY valid JSON in this exact shape:
{
  "cards": [
    {"term": "...", "definition": "...", "source": <integer>},
    ...
  ]
}

No extra text, no markdown, no code fences, just the JSON object."""


def _unresolved_scope_message(meta: dict) -> str:
    if meta.get("scope_resolved", True) is not False:
        return ""
    return (
        meta.get("unresolved_reason")
        or "The requested chapter or section could not be matched. Use a page range or topic keyword instead."
    )


class FlashcardsGenerator:
    """Generate flashcards from document chunks using LLM extraction."""

    def __init__(self, llm_client, scope_resolver: ScopeResolver):
        self._llm = llm_client
        self._resolver = scope_resolver

    def generate(
        self,
        notebook_id: str,
        document_id: str,
        scope: str,
        count: int = 20,
        doc_filename: str = "",
    ) -> dict[str, Any]:
        result = self._resolver.resolve(document_id, notebook_id, scope)
        chunks = result.chunks
        meta = result.metadata

        if not chunks:
            message = _unresolved_scope_message(meta)
            return {
                "title": self._make_title(scope or "", doc_filename),
                "message": message or "No relevant content was found in the selected scope.",
                "cards": [],
                "sources": [],
                "stats": {
                    "card_count": 0,
                    "requested_count": count,
                    "scope_layer": meta.get("layer", "unknown"),
                    "scope_resolved": meta.get("scope_resolved", True),
                    "unresolved_reason": message or meta.get("unresolved_reason", ""),
                },
            }

        # Filter to body chunks
        body_chunks = [c for c in chunks if is_body_chunk(c)]
        if not body_chunks:
            body_chunks = chunks

        # Determine if this is a whole-document scope
        is_whole_doc = self._is_whole_document_scope(scope, meta)

        if is_whole_doc and len(body_chunks) > 30:
            # Section-balanced generation for whole book
            result = self._generate_section_balanced(
                body_chunks, scope, doc_filename, count, meta,
            )
        else:
            # Original single-call generation (scoped or small documents)
            source_chunks = sample_body_chunks(body_chunks, max_chunks=25)
            result = self._generate_single_call(
                source_chunks, scope, doc_filename, count, meta,
            )
        result["sources"] = collect_book_citations(result.get("cards", []), document_id, doc_filename)
        return result

    @staticmethod
    def _is_whole_document_scope(scope: str, meta: dict) -> bool:
        """Check if scope represents whole-document generation."""
        if not scope or not scope.strip():
            return True
        if meta.get("layer") == "all":
            return True
        return False

    def _generate_single_call(
        self,
        source_chunks: list[RetrievedChunk],
        scope: str,
        doc_filename: str,
        count: int,
        meta: dict,
    ) -> dict[str, Any]:
        """Single LLM call for scoped or small-document flashcards."""
        user_prompt = self._build_user_prompt(source_chunks, scope, doc_filename, count)

        try:
            raw = self._llm.complete_chat(
                messages=[
                    {"role": "system", "content": FLASHCARDS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(8192, 200 + count * 180),
            )
        except Exception as exc:
            logger.error("Flashcards LLM call failed: %s", exc)
            raise

        cards_raw = self._parse_cards(raw)
        cards = self._enrich_cards(cards_raw, source_chunks)

        if not cards:
            raise RuntimeError(
                "LLM produced no valid flashcards. The output may have been "
                "truncated or malformed. Try again, or reduce the requested count."
            )

        return {
            "title": self._make_title(scope or "", doc_filename),
            "cards": cards,
            "stats": {
                "card_count": len(cards),
                "requested_count": count,
                "scope_layer": meta.get("layer", "unknown"),
                "scope": scope or "",
            },
        }

    def _generate_section_balanced(
        self,
        body_chunks: list[RetrievedChunk],
        scope: str,
        doc_filename: str,
        count: int,
        meta: dict,
    ) -> dict[str, Any]:
        """Generate flashcards section-by-section for even whole-book coverage.

        1. Group chunks by major section
        2. Allocate card budget per section
        3. For each section: sample representative chunks, call LLM
        4. Merge and dedupe results
        """
        sections = group_chunks_by_major_section(body_chunks)
        if not sections:
            return self._generate_single_call(body_chunks, scope, doc_filename, count, meta)

        # If too few sections, fall back to single-call
        if len(sections) <= 2:
            return self._generate_single_call(
                sample_body_chunks(body_chunks, max_chunks=25),
                scope, doc_filename, count, meta,
            )

        # Allocate budget
        budget = allocate_flashcard_budget(sections, count)
        if not budget:
            return self._generate_single_call(body_chunks, scope, doc_filename, count, meta)

        # Generate per section
        all_cards: list[dict] = []
        seen_terms: set[str] = set()
        llm_calls = 0
        sections_generated = 0
        consumed_chunk_ids: set[str] = set()  # track chunks already used

        for section in sections:
            alloc = budget.get(section.key, 0)
            if alloc <= 0:
                continue

            # Sample chunks from this section
            section_chunks = sample_section_chunks(section, max_chunks=4)
            consumed_chunk_ids.update(c.chunk_id for c in section_chunks)

            # Build section-specific prompt
            section_prompt = self._build_section_prompt(
                section_chunks, section.key, doc_filename, alloc,
            )

            try:
                raw = self._llm.complete_chat(
                    messages=[
                        {"role": "system", "content": FLASHCARDS_SYSTEM_PROMPT},
                        {"role": "user", "content": section_prompt},
                    ],
                    max_tokens=min(4096, 200 + alloc * 200),
                )
                llm_calls += 1
                sections_generated += 1
            except Exception as exc:
                logger.error(
                    "Flashcards LLM call failed for section %s: %s", section.key, exc,
                )
                continue

            cards_raw = self._parse_cards(raw)
            cards = self._enrich_cards(cards_raw, section_chunks)

            # Dedupe
            for c in cards:
                term_lower = str(c.get("term", "")).strip().lower()
                if term_lower and term_lower not in seen_terms:
                    seen_terms.add(term_lower)
                    all_cards.append(c)

        # Top-up: if section-balanced produced fewer than 80% of requested,
        # do a single-call top-up with UNUSED body chunks.
        top_up_used = False
        if 0 < len(all_cards) < count * 0.8 and len(body_chunks) > 0:
            shortage = count - len(all_cards)
            # Gather chunks NOT already consumed during section generation
            remaining_chunks = [c for c in body_chunks if c.chunk_id not in consumed_chunk_ids]
            if not remaining_chunks:
                remaining_chunks = body_chunks  # fallback
            top_up_chunks = sample_body_chunks(remaining_chunks, max_chunks=25)
            if top_up_chunks:
                logger.info(
                    "Top-up: section-balanced produced %d cards (requested %d), "
                    "running top-up for %d more.",
                    len(all_cards), count, shortage,
                )
                try:
                    top_up_result = self._generate_single_call(
                        top_up_chunks, scope, doc_filename, shortage, meta,
                    )
                    top_up_cards = top_up_result.get("cards", [])
                    for c in top_up_cards:
                        term_lower = str(c.get("term", "")).strip().lower()
                        if term_lower and term_lower not in seen_terms:
                            seen_terms.add(term_lower)
                            all_cards.append(c)
                    llm_calls += 1
                    top_up_used = True
                except Exception as exc:
                    logger.error("Top-up LLM call failed: %s", exc)

        # Empty result: fall back to single-call
        if not all_cards:
            logger.warning(
                "Section-balanced generation produced 0 flashcards — "
                "falling back to single-call generation."
            )
            return self._generate_single_call(
                sample_body_chunks(body_chunks, max_chunks=25),
                scope, doc_filename, count, meta,
            )

        # Trim to requested count
        if len(all_cards) > count:
            all_cards = all_cards[:count]

        # Compute coverage stats
        section_counts: dict[str, int] = defaultdict(int)
        for c in all_cards:
            sp = major_section_key(c.get("section_path") or "")
            if sp:
                section_counts[sp] += 1

        return {
            "title": self._make_title(scope or "", doc_filename),
            "cards": all_cards,
            "stats": {
                "card_count": len(all_cards),
                "requested_count": count,
                "scope_layer": meta.get("layer", "unknown"),
                "scope": scope or "",
                "generation_mode": "section_balanced",
                "sections_covered": len(section_counts),
                "sections_total": len(sections),
                "sections_generated": sections_generated,
                "llm_calls": llm_calls,
                "top_up_used": top_up_used,
                "coverage_by_section": dict(
                    sorted(section_counts.items(), key=lambda x: -x[1])[:10]
                ),
            },
        }

    def _build_section_prompt(
        self,
        chunks: list[RetrievedChunk],
        section_name: str,
        doc_filename: str,
        count: int,
    ) -> str:
        """Build a prompt for a single section's flashcard generation."""
        lines = [
            f"Document: {doc_filename or '(untitled)'}",
            f"Section: {section_name}",
            f"Generate ONLY flashcards from this section. Do not include terms from other sections.",
            f"Requested count for this section: {count}",
            "",
            "Source passages:",
            "",
        ]
        for i, c in enumerate(chunks, 1):
            sec = c.section_path or c.title or ""
            loc = f"p.{c.page_start}" if c.page_start > 0 else ""
            header = f"[{i}] {sec}" + (f" ({loc})" if loc else "")
            lines.append(header)
            lines.append(c.text[:1500])
            lines.append("---")
        lines.append("")
        lines.append(f"Generate up to {count} flashcards from THIS SECTION in the JSON format specified.")
        return "\n".join(lines)

    def _build_user_prompt(self, chunks, scope, doc_filename, count):
        lines = [
            f"Document: {doc_filename or '(untitled)'}",
            f"Scope: {scope or 'whole document'}",
            f"Requested count: {count}",
            "",
            "Source passages:",
            "",
        ]
        for i, c in enumerate(chunks, 1):
            sec = c.section_path or c.title or ""
            loc = f"p.{c.page_start}" if c.page_start > 0 else ""
            header = f"[{i}] {sec}" + (f" ({loc})" if loc else "")
            lines.append(header)
            lines.append(c.text[:1500])
            lines.append("---")
        lines.append("")
        lines.append(f"Generate up to {count} flashcards in the JSON format specified.")
        return "\n".join(lines)

    def _parse_cards(self, raw: str) -> list[dict]:
        """Extract cards array from LLM output.

        Robust to:
          - Markdown code fences (```json ... ```)
          - LLM truncation mid-object (drops the last incomplete card and recovers)
          - Trailing prose after JSON object
        """
        text = raw.strip()

        # Strip code fences if present
        if text.startswith("```"):
            parts = text.split("```", 2)
            text = parts[1] if len(parts) >= 2 else text
            text = text.lstrip("json").strip()

        # Locate the JSON object
        start = text.find("{")
        if start == -1:
            logger.warning("Flashcards: no JSON object in response")
            return []

        # Try direct parse first (happy path)
        end = text.rfind("}")
        if end > start:
            try:
                obj = json.loads(text[start : end + 1])
                cards = obj.get("cards", [])
                return [c for c in cards if isinstance(c, dict) and c.get("term") and c.get("definition")]
            except json.JSONDecodeError as e:
                logger.warning("Flashcards JSON parse failed: %s. Attempting truncation recovery.", e)

        # Truncation recovery: walk the cards array extracting complete objects
        recovered = self._recover_truncated_cards(text[start:])
        if recovered:
            logger.info("Flashcards: recovered %d cards via truncation repair", len(recovered))
        return recovered

    def _recover_truncated_cards(self, text: str) -> list[dict]:
        """Salvage as many complete card objects as possible from a truncated JSON string.

        Strategy: locate the "cards" array, then scan character-by-character tracking
        brace depth and string state to extract each balanced {...} object. Stop at the
        first object that doesn't close cleanly.
        """
        # Find array start
        arr_idx = text.find('"cards"')
        if arr_idx == -1:
            return []
        bracket = text.find("[", arr_idx)
        if bracket == -1:
            return []

        cards: list[dict] = []
        i = bracket + 1
        n = len(text)
        while i < n:
            # Skip whitespace and commas
            while i < n and text[i] in " \t\n\r,":
                i += 1
            if i >= n or text[i] == "]":
                break
            if text[i] != "{":
                # Unexpected token; bail
                break

            # Track a balanced object
            depth = 0
            in_string = False
            escape = False
            obj_start = i
            while i < n:
                ch = text[i]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif in_string:
                    if ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                i += 1

            if depth != 0:
                # Truncated object — stop here, don't include partial
                break

            chunk = text[obj_start:i]
            try:
                card = json.loads(chunk)
                if isinstance(card, dict) and card.get("term") and card.get("definition"):
                    cards.append(card)
            except json.JSONDecodeError:
                # Skip malformed individual card, continue with next
                pass

        return cards

    def _enrich_cards(self, cards_raw: list[dict], chunks: list[RetrievedChunk]) -> list[dict]:
        """Map LLM's 'source' (passage number) back to chunk metadata."""
        enriched = []
        for c in cards_raw:
            src_idx = c.get("source")
            chunk = None
            if isinstance(src_idx, int) and 1 <= src_idx <= len(chunks):
                chunk = chunks[src_idx - 1]
            enriched.append({
                "term": str(c.get("term", "")).strip(),
                "definition": str(c.get("definition", "")).strip(),
                "section_path": chunk.section_path if chunk else "",
                "page_start": chunk.page_start if chunk else -1,
                "page_end": chunk.page_end if chunk else -1,
                "source_chunk_id": chunk.chunk_id if chunk else "",
            })
        return enriched

    @staticmethod
    def _make_title(scope: str, doc_filename: str) -> str:
        base = clean_filename(doc_filename)
        if not scope:
            return f"Flashcards: {base}"
        return f"Flashcards: {base} — {scope}"
