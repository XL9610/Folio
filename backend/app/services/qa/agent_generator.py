"""AgentAnswerGenerator — LLM-based grounded answer synthesis.

Receives tool results (metadata, outline, retrieved chunks) and
generates a natural Markdown answer with citations.

Generator is synthesize-first. Deterministic gates may run before generation,
but semantic sufficiency is judged here from full tool context.
"""

from __future__ import annotations

import logging

from ..pdf_ingestion.structure_metadata import is_unreliable_structure_status
from .agent_planner import PlanResult
from .section_labels import extract_numbered_section_prefix, section_label_for_display
from .tools import ToolResults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — synthesize-first philosophy
# ---------------------------------------------------------------------------

AGENT_ANSWER_SYSTEM_PROMPT = """You are a study companion. The provided tool results are the material the user can currently see.

Your job is to synthesize the most useful answer supported by this material. Prefer answering over refusing.

Rules:
1. Use only the provided material. You may synthesize, summarize, and draw connections across retrieved book passages and internet results, but do not introduce specific facts (names, dates, numbers, quotes) that are not in the material.
2. If the material covers the topic through examples, cases, chapters, or related sections, synthesize the concept from those examples. This is not fabrication.
3. When the material is partial, answer what it supports and use attribution language such as "Based on the chapters retrieved..." or "The textbook covers...". Limitations should be at most one short sentence and only if they materially affect the answer.
4. When synthesizing a concept from examples, attribute the synthesis to the source structure rather than asserting it as a canonical definition. For example: "This textbook treats arrays, linked lists, and trees as different ways of structuring data..." Avoid "X is defined as..." unless that definition appears in a retrieved passage.
5. Cite uploaded book passages with [S1], [S2] format. Cite internet results with [W1], [W2] format. If a claim is supported by both, cite both, e.g. [S1][W1].
6. Never cite internet material as if it came from uploaded books. If internet evidence only clarifies a term that the book context did not explain, add a short "Internet context" paragraph and cite it with [W#].
7. If book and internet evidence conflict, explicitly distinguish "your source says..." from "internet sources say...".
8. Only refuse in the rare case that no retrieved passage, outline, metadata, or internet result is relevant to the current query. In that case, briefly say what the retrieved material is about instead.

The current user query is authoritative. Use conversation history only for pronoun/reference resolution. Do not answer a previous topic if it differs from the current query.

Do not say "I would need a passage that..." or "the user should provide the full text." The user cannot control retrieval. Work with the material available.

Use natural Markdown. Do not follow a fixed template."""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AgentAnswerGenerator:
    """Generate grounded answers from tool results using LLM synthesis."""

    def __init__(self, llm_client):
        self._llm = llm_client

    def generate(
        self,
        query: str,
        history: list[dict] | None,
        plan: PlanResult,
        tool_results: ToolResults,
        source_info: dict,
        preferences: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """Generate a grounded answer.

        source_info may contain notebook_id and document_ids for context
        (evidence_status/reason are no longer injected into the prompt).

        Returns:
            (answer_text, citations) — citations is list of dicts.
        """
        context = self._build_context(tool_results, source_info)
        preference_instructions = self._build_preference_instructions(preferences)

        # Build conversation context
        conv = ""
        if history:
            recent = history[-4:]
            conv = "Recent conversation:\n"
            for h in recent:
                role = h.get("role", "unknown")
                text = h.get("text", "")[:300]
                conv += f"  [{role}]: {text}\n"

        user_prompt = (
            f"{conv}\n"
            f"User query: {query}\n\n"
            f"{preference_instructions}"
            f"Tool results:\n{context}"
        )

        if not context.strip():
            user_prompt += "\n\n(No tool results were provided. If you cannot ground an answer, briefly say no relevant material was retrieved.)"

        try:
            answer = self._llm.complete_chat(
                messages=[
                    {"role": "system", "content": AGENT_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1536,
            )
        except Exception as exc:
            logger.error("Answer generation LLM call failed: %s", exc)
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        return answer, tool_results.chunk_citations

    # ------------------------------------------------------------------
    # Preference builder
    # ------------------------------------------------------------------

    def _build_preference_instructions(self, preferences: dict | None) -> str:
        """Convert answer preferences into style-only generation instructions."""
        if not isinstance(preferences, dict) or not preferences:
            return ""

        lines: list[str] = []

        style = self._preference_value(
            preferences,
            "style",
            "answer_style",
            "answerStyle",
        )
        style_key = self._preference_key(style)
        style_map = {
            "plain": "Use plain language and briefly explain technical terms.",
            "concise": "Use a concise study-note style.",
            "detailed": "Use a detailed explanatory style with enough context for studying.",
            "structured": "Use short headings and bullets when they make the answer easier to scan.",
            "bullets": "Use bullet points where they improve clarity.",
            "bullet": "Use bullet points where they improve clarity.",
            "prose": "Use clear prose paragraphs.",
            "socratic": "After answering, include a few guiding questions for self-checking.",
            "exam": "Emphasize exam-relevant takeaways and likely points of confusion.",
        }
        if style_key in style_map:
            lines.append(style_map[style_key])

        length = self._preference_value(
            preferences,
            "length",
            "answer_length",
            "answerLength",
        )
        length_key = self._preference_key(length)
        length_map = {
            "brief": "Keep the answer brief.",
            "short": "Keep the answer brief.",
            "concise": "Keep the answer brief.",
            "medium": "Use a moderate length.",
            "normal": "Use a moderate length.",
            "detailed": "Give a thorough but focused explanation.",
            "long": "Give a thorough but focused explanation.",
            "comprehensive": "Give a comprehensive explanation while staying grounded.",
        }
        if length_key in length_map:
            lines.append(length_map[length_key])

        detail_level = self._preference_value(
            preferences,
            "detail_level",
            "detailLevel",
        )
        detail_key = self._preference_key(detail_level)
        detail_map = {
            "simple": "Assume the reader is new to the topic.",
            "intro": "Assume the reader is new to the topic.",
            "beginner": "Assume the reader is new to the topic.",
            "intermediate": "Assume some familiarity with the topic.",
            "advanced": "Use advanced detail when the retrieved material supports it.",
        }
        if detail_key in detail_map:
            lines.append(detail_map[detail_key])

        tone = self._preference_value(preferences, "tone")
        tone_key = self._preference_key(tone)
        tone_map = {
            "academic": "Use an academic but readable tone.",
            "direct": "Use a direct, no-fluff tone.",
            "encouraging": "Use an encouraging study-coach tone.",
            "neutral": "Use a neutral explanatory tone.",
            "plain": "Use a plain-spoken tone.",
        }
        if tone_key in tone_map:
            lines.append(tone_map[tone_key])

        fmt = self._preference_value(preferences, "format")
        format_key = self._preference_key(fmt)
        format_map = {
            "bullets": "Prefer bullets over dense paragraphs.",
            "bullet": "Prefer bullets over dense paragraphs.",
            "paragraph": "Prefer clear paragraphs over bullet-heavy formatting.",
            "paragraphs": "Prefer paragraphs over bullets.",
            "prose": "Prefer paragraphs over bullets.",
            "study-notes": "Format the answer as study notes with concise headings and takeaways.",
            "study_notes": "Format the answer as study notes with concise headings and takeaways.",
            "steps": "Use numbered steps when explaining a process.",
            "table": "Use a compact Markdown table only if comparison is central.",
        }
        if format_key in format_map:
            lines.append(format_map[format_key])

        citations = self._preference_value(
            preferences,
            "citations",
            "citation_density",
            "citationDensity",
        )
        citation_key = self._preference_key(citations)
        citation_map = {
            "minimal": "Use the fewest citations needed for traceability.",
            "normal": "Use a normal citation density for important claims.",
            "verbose": "Cite supporting evidence generously, especially across multiple sources.",
        }
        if citation_key in citation_map:
            lines.append(citation_map[citation_key])

        auto_table = preferences.get("auto_summary_table", preferences.get("autoSummaryTable"))
        if auto_table is True:
            lines.append("Include a compact summary table when the answer compares positions, concepts, or evidence.")

        audience = self._preference_value(preferences, "audience", max_chars=120)
        if audience:
            lines.append(f"Write for this audience: {audience}.")

        language = self._normalize_language_preference(
            self._preference_value(preferences, "language", max_chars=80)
        )
        if language:
            lines.append(f"Use this language if possible: {language}.")

        custom = self._preference_value(
            preferences,
            "custom_instructions",
            "customInstructions",
            "instructions",
            max_chars=600,
        )
        if custom:
            lines.append(f"Additional style note: {custom}")

        if not lines:
            return ""

        return (
            "Answer preferences (style only; do not override grounding, citation, "
            "or source-use rules):\n"
            + "\n".join(f"- {line}" for line in lines)
            + "\n\n"
        )

    @staticmethod
    def _preference_value(
        preferences: dict,
        *keys: str,
        max_chars: int = 80,
    ) -> str:
        for key in keys:
            raw = preferences.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                value = " ".join(raw.split())
            elif isinstance(raw, int | float):
                value = str(raw)
            else:
                continue
            if value:
                return value[:max_chars]
        return ""

    @staticmethod
    def _preference_key(value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _normalize_language_preference(cls, value: str) -> str:
        key = cls._preference_key(value)
        if not key or key in {"auto", "default", "system"}:
            return ""
        if key in {"zh", "zh_cn", "zh_hans", "chinese", "simplified_chinese"}:
            return "Chinese"
        if key in {"en", "en_us", "english"}:
            return "English"
        return value.strip()

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, tr: ToolResults, source_info: dict) -> str:
        """Build the context string for the LLM from tool results."""
        parts = []

        # Source metadata
        for i, m in enumerate(tr.metadata):
            parts.append(
                f"[Document {i+1}]\n"
                f"File: {m.get('filename', 'unknown')}\n"
                f"Pages: {m.get('page_count', 0)}\n"
                f"Chunks: {m.get('chunk_count', 0)}\n"
                f"Structure status: {m.get('structure_status', 'unknown_legacy')}\n"
                f"Structure note: {m.get('structure_user_note', '')}\n"
                f"Structure instruction: {m.get('structure_llm_instruction', '')}\n"
            )

        # Target section if specified
        if tr.target_section:
            parts.append(f"[Requested section] {tr.target_section}")
            if not tr.section_matches and self._has_unreliable_outline(tr):
                parts.append(
                    "[Structure limitation]\n"
                    f"The requested section '{tr.target_section}' was not matched "
                    "to a reliable chapter or section heading. If retrieved passages "
                    "are used, say that no reliable chapter boundary was found and "
                    "answer only from those passages."
                )

        # Section matches (from find_sections)
        if tr.section_matches:
            parts.append("[Section matches found]")
            for m in tr.section_matches[:5]:
                label = section_label_for_display(m.section_path, m.title)
                parts.append(
                    f"  - Title/heading: {label}\n"
                    f"    Full path: {m.section_path}\n"
                    f"    Pages: {m.page_start}-{m.page_end}; "
                    f"{m.chunk_count} chunks; match: {m.match_type}"
                )

        # Outline — chapter-level condensation
        for o in tr.outline:
            filename = o.get("filename", "unknown")
            sections = o.get("sections", [])
            outline_str = self._build_outline_context(sections, filename)
            structure_instruction = o.get("structure_llm_instruction", "")
            if structure_instruction:
                outline_str += (
                    "\nStructure status: "
                    f"{o.get('structure_status', 'unknown_legacy')}\n"
                    f"Structure instruction: {structure_instruction}"
                )
            parts.append(outline_str)

        # Retrieved chunks (truncate to max)
        if tr.chunks:
            parts.append("[Book passages]")
            book_ref_by_chunk = {
                c.get("chunk_id", ""): c.get("ref_label", "")
                for c in tr.chunk_citations
                if c.get("source_type", "book") == "book"
            }
            max_chunks = min(len(tr.chunks), 10)
            for i, chunk in enumerate(tr.chunks[:max_chunks]):
                loc = f"p. {chunk.page_start}" if chunk.page_start == chunk.page_end else f"pp. {chunk.page_start}-{chunk.page_end}"
                ref_label = book_ref_by_chunk.get(chunk.chunk_id) or f"S{i+1}"
                header = f"[{ref_label}] {chunk.section_path or chunk.title} ({loc})"
                parts.append(f"{header}\n{chunk.text[:2000]}")

        if tr.web_results:
            parts.append("[Internet results]")
            for i, item in enumerate(tr.web_results[:8]):
                ref_label = item.get("ref_label") or f"W{i+1}"
                title = item.get("title") or "Untitled web result"
                site = item.get("site_name") or "web"
                url = item.get("url") or ""
                retrieved = item.get("retrieved_at") or ""
                fetched = str(item.get("fetched_content") or "")
                snippet = str(item.get("content") or "")
                # Prefer full fetched content; fall back to search snippet.
                body = fetched if fetched else snippet
                cap = 4000 if fetched else 1800
                header = f"[{ref_label}] {title} ({site}) {url}"
                if retrieved:
                    header += f" retrieved {retrieved}"
                if fetched:
                    header += " [full content]"
                parts.append(f"{header}\n{body[:cap]}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _has_unreliable_outline(tr: ToolResults) -> bool:
        containers = list(tr.metadata) + list(tr.outline)
        if not containers:
            return False
        return any(
            is_unreliable_structure_status(item.get("structure_status"))
            for item in containers
        )

    def _build_outline_context(self, sections: list[dict], filename: str) -> str:
        """Build a condensed outline with complete numbered chapter/section headings."""
        top_groups: dict[str, list[dict]] = {}
        top_order: list[str] = []
        numbered_units: dict[tuple[int, str], dict] = {}

        for s in sections:
            sp = s.get("section_path", "") or s.get("title", "")
            top = sp.split(">")[0].strip() if ">" in sp else sp.strip()
            if top:
                if top not in top_groups:
                    top_groups[top] = []
                    top_order.append(top)
                top_groups[top].append(s)

            number, prefix, label = extract_numbered_section_prefix(
                sp,
                s.get("title", ""),
            )
            if number is not None and prefix:
                key = (number, prefix)
                if key not in numbered_units:
                    numbered_units[key] = {
                        "number": number,
                        "prefix": prefix,
                        "label": label,
                        "page_start": s.get("page_start", "?"),
                        "page_end": s.get("page_end", "?"),
                        "count": 0,
                    }
                unit = numbered_units[key]
                unit["count"] += 1
                start = s.get("page_start", unit["page_start"])
                end = s.get("page_end", unit["page_end"])
                if isinstance(start, int) and (
                    not isinstance(unit["page_start"], int) or start < unit["page_start"]
                ):
                    unit["page_start"] = start
                if isinstance(end, int) and (
                    not isinstance(unit["page_end"], int) or end > unit["page_end"]
                ):
                    unit["page_end"] = end

        total_sections = len(sections)
        lines: list[str] = [f"[Outline] {filename}"]

        if not top_order and not numbered_units:
            lines.append("No section structure available.")
            return "\n".join(lines)

        if top_order:
            lines.append(
                f"Top-level headings ({len(top_order)} headings, "
                f"{total_sections} total sections):"
            )
            for j, top in enumerate(top_order):
                subs = top_groups[top]
                first = subs[0].get("page_start", "?") if subs else "?"
                last = subs[-1].get("page_end", "?") if subs else "?"
                pg = f"pp. {first}-{last}" if first != last else f"p. {first}"
                lines.append(f"  {j+1}. {top} ({pg}, {len(subs)} sections)")

        ordered_units = sorted(
            numbered_units.values(),
            key=lambda u: (
                u["page_start"] if isinstance(u["page_start"], int) else 999999,
                u["number"],
            ),
        )
        if ordered_units:
            lines.append("\nDetected numbered chapter/section headings (complete):")
            for unit in ordered_units:
                first = unit["page_start"]
                last = unit["page_end"]
                pg = f"pp. {first}-{last}" if first != last else f"p. {first}"
                lines.append(
                    f"  {unit['number']}. {unit['label']} "
                    f"({pg}, path: {unit['prefix']}, {unit['count']} sections)"
                )

            lines.append("\nRepresentative subsections (up to 3 per numbered heading):")
            for unit in ordered_units:
                prefix = unit["prefix"]
                show = [
                    s for s in sections
                    if (
                        (s.get("section_path", "") or s.get("title", "")) != prefix
                        and (
                            (s.get("section_path", "") or s.get("title", "")).startswith(prefix + " >")
                        )
                    )
                ][:3]
                if not show:
                    continue
                lines.append(f"  {unit['label']}:")
                for s in show:
                    sp = s.get("section_path", "") or s.get("title", "")
                    pg_start = s.get("page_start", "?")
                    pg_end = s.get("page_end", 0)
                    pg = (
                        f"pp. {pg_start}-{pg_end}"
                        if isinstance(pg_start, int) and isinstance(pg_end, int) and pg_end > pg_start
                        else f"p. {pg_start}"
                    )
                    subsection = sp[len(prefix):].lstrip(" >") or sp
                    lines.append(f"    - {subsection} ({pg})")

        if total_sections > 60:
            lines.append(
                f"\n(Note: Full outline has {total_sections} sections. "
                f"Top-level and numbered heading lists are complete.)"
            )

        return "\n".join(lines)
