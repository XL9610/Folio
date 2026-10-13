"""Hierarchical overview query adapter — roadmap Phase 3 query path.

Deterministic route for ``retrieval_scope == "whole_source"`` when the
feature flag is enabled and every requested document has a current,
complete summary hierarchy. Loads the cards into a ``summary_context``
collection (never ``RetrievedChunk`` instances, never citation labels),
selects representative raw evidence round-robin across sections so early
chapters cannot consume the citation budget, and hydrates only canonical
raw chunks as citable evidence.

The caller (``QAToolExecutor``) keeps the existing 6/4 coverage+relevance
route as the only fallback and never broadens notebook/document
authorization: every read here is scoped to ``notebook_id`` and the
requested ``document_ids``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ...repositories.chunk_repository import (
    get_chunk_by_id,
    get_chunk_identity_by_document,
)
from ...repositories.document_repository import get_document
from ...repositories.summary_node_repository import get_document_nodes
from ..generation.hierarchical_summary_builder import compute_build_signature
from ..generation.sampling import _allocate_document_budget
from ..qa.citations import MAX_BOOK_CITATIONS

logger = logging.getLogger(__name__)

# Bounded summary-context budget (frozen for evaluation runs). Document cards
# always fit first; section cards fill the remainder round-robin across
# documents. Document cards are the compression layer — whole documents are
# never silently dropped on similarity rank.
SUMMARY_CONTEXT_BUDGET_CHARS = 12000


@dataclass
class HierarchicalContext:
    status: str = "unavailable"  # ready | unavailable
    reason: str = ""
    summary_context: list[dict] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    trace_args: dict = field(default_factory=dict)


def _render_card_chars(card: dict) -> int:
    return len(json.dumps(card, ensure_ascii=False))


def _fit_to_budget(item: dict, headroom: int) -> dict | None:
    """Fit ``item`` into ``headroom`` chars by shrinking its summary text.

    Returns the (possibly truncated) item, or None when even a minimal
    card cannot fit. Deterministic: the summary is halved until it fits.
    """
    if _render_card_chars(item) <= headroom:
        return item
    full_summary = item["summary"]
    low, high = 0, len(full_summary)
    while low < high:
        mid = (low + high + 1) // 2
        item["summary"] = full_summary[:mid]
        if _render_card_chars(item) <= headroom:
            low = mid
        else:
            high = mid - 1
    item["summary"] = full_summary[:low]
    if _render_card_chars(item) > headroom:
        return None
    return item


class HierarchicalContextAdapter:
    """Loads and validates one document set's hierarchy for a whole-source ask."""

    def __init__(self, notebook_id: str, document_ids: list[str]):
        self._notebook_id = notebook_id
        self._document_ids = list(document_ids)[:10]

    def prepare(self) -> HierarchicalContext:
        """Validate every requested document and assemble the context.

        Returns status "ready" with cards + evidence ids, or "unavailable"
        with a reason (missing/stale/incomplete/failed validation). The
        caller falls back to the existing 6/4 route when unavailable.
        """
        per_doc: list[dict] = []
        try:
            for doc_id in self._document_ids:
                entry = self._load_document(doc_id)
                if entry is None:
                    return HierarchicalContext(
                        status="unavailable", reason="missing_or_stale_index",
                    )
                per_doc.append(entry)
        except Exception as exc:  # noqa: BLE001 — route failure must fall back
            logger.warning("Hierarchical overview unavailable: %s", exc)
            return HierarchicalContext(
                status="unavailable",
                reason=f"hierarchy_validation_failed: {exc}",
            )

        if not per_doc:
            return HierarchicalContext(
                status="unavailable", reason="missing_or_stale_index",
            )

        summary, total_chars = self._build_summary_context(per_doc)
        evidence = self._select_evidence(per_doc)
        if not evidence:
            return HierarchicalContext(
                status="unavailable", reason="hierarchy_validation_failed",
            )

        return HierarchicalContext(
            status="ready",
            summary_context=summary,
            evidence_chunk_ids=evidence,
            trace_args={
                "documents": len(per_doc),
                "section_nodes": sum(e["section_count"] for e in per_doc),
                "document_nodes": len(per_doc),
                "grouping_modes": sorted({
                    e["grouping_mode"] for e in per_doc
                }),
                "cache": "hit",
                "raw_evidence_count": len(evidence),
                "summary_context_chars": total_chars,
                # Section keys represented in the summary context — titles
                # only, for eval/debug (no source text).
                "section_keys": sorted({
                    card["section_key"]
                    for card in summary
                    if card.get("section_key")
                })[:64],
            },
        )

    # ------------------------------------------------------------------
    # Loading + validation
    # ------------------------------------------------------------------

    def _load_document(self, doc_id: str) -> dict | None:
        doc = get_document(doc_id)
        if not doc or doc.get("notebook_id") != self._notebook_id:
            return None  # never broaden authorization boundaries
        # Identity-only read: the signature needs ordered ids + content
        # hashes, and provenance validation needs the id set. Full chunk
        # text is hydrated later, only for the evidence actually selected.
        identity_rows = get_chunk_identity_by_document(doc_id)
        if not identity_rows:
            return None
        signature = compute_build_signature(
            doc.get("content_hash", ""), identity_rows,
        )
        nodes = get_document_nodes(self._notebook_id, doc_id)
        doc_nodes = [n for n in nodes if n["level"] == "document"]
        section_nodes = [n for n in nodes if n["level"] == "section"]
        if len(doc_nodes) != 1 or not section_nodes:
            return None
        if any(n["build_signature"] != signature for n in nodes):
            return None
        cards: list[dict] = []
        for n in section_nodes:
            card = json.loads(n["summary_json"])
            if not _valid_section_card_shape(card):
                return None
            cards.append({"node_id": n["id"], "section_key": n["section_key"],
                          "source_chunk_ids": n["source_chunk_ids"], **card})
        doc_node = doc_nodes[0]
        doc_card = json.loads(doc_node["summary_json"])
        if not _valid_document_card_shape(doc_card):
            return None
        # Stored provenance must resolve to real chunks in this document.
        valid_ids = {r["id"] for r in identity_rows}
        if not set(doc_node["source_chunk_ids"]) <= valid_ids:
            return None
        for n in section_nodes:
            if not set(n["source_chunk_ids"]) <= valid_ids:
                return None
        return {
            "document_id": doc_id,
            "filename": doc.get("filename", ""),
            "grouping_mode": doc_card.get("grouping_mode", ""),
            "section_count": len(section_nodes),
            "doc_card": doc_card,
            "section_cards": cards,
            "doc_provenance": doc_node["source_chunk_ids"],
        }

    # ------------------------------------------------------------------
    # Summary context assembly
    # ------------------------------------------------------------------

    def _build_summary_context(self, per_doc: list[dict]) -> tuple[list[dict], int]:
        """Document cards first (compression layer), then section cards
        round-robin across documents while the budget allows."""
        budget = SUMMARY_CONTEXT_BUDGET_CHARS
        summary: list[dict] = []
        total = 0

        for e in per_doc:
            item = {
                "document_id": e["document_id"],
                "filename": e["filename"],
                "title": e["doc_card"].get("title", "Document overview"),
                "summary": e["doc_card"].get("summary", ""),
                "major_topics": [
                    {"text": t.get("text", "")}
                    for t in e["doc_card"].get("major_topics", [])
                ],
                "grouping_mode": e["grouping_mode"],
            }
            fitted = _fit_to_budget(item, budget - total)
            if fitted is None:
                # Never silently drop a whole document: keep a minimal card.
                fitted = {
                    "document_id": item["document_id"],
                    "filename": item["filename"],
                    "title": item["title"],
                    "grouping_mode": item["grouping_mode"],
                }
                if _render_card_chars(fitted) > budget - total:
                    fitted["title"] = fitted["title"][:24]
            summary.append(fitted)
            total += _render_card_chars(fitted)

        # Section cards fill the remainder, fairly across documents. A card
        # that cannot fit is requeued so other documents' smaller cards can
        # still be added; the loop ends only when a full pass adds nothing.
        queues = [list(e["section_cards"]) for e in per_doc]
        while total < budget:
            progressed = False
            for i, queue in enumerate(queues):
                if not queue:
                    continue
                card = queue.pop(0)
                item = {
                    "document_id": per_doc[i]["document_id"],
                    "filename": per_doc[i]["filename"],
                    "title": card.get("title", ""),
                    "summary": card.get("summary", ""),
                    "points": [{"text": p.get("text", "")}
                               for p in card.get("points", [])],
                    "section_key": card.get("section_key", ""),
                }
                fitted = _fit_to_budget(item, budget - total)
                if fitted is None:
                    queues[i] = [card, *queue]
                    continue
                summary.append(fitted)
                total += _render_card_chars(fitted)
                progressed = True
            if not progressed:
                break

        return summary, total

    # ------------------------------------------------------------------
    # Evidence selection + hydration ids
    # ------------------------------------------------------------------

    def _select_evidence(self, per_doc: list[dict]) -> list[str]:
        """Round-robin representative raw chunk ids across sections.

        Single document: pass 1 takes each section's first provenance chunk,
        pass 2 the next, and so on, so early chapters cannot consume the
        citation budget. Multiple documents: the budget is allocated fairly
        across documents first (existing sampling helper), then round-robin
        within each document. Document-card provenance fills any remainder.
        """
        total_budget = MAX_BOOK_CITATIONS
        doc_budgets = _allocate_document_budget(
            [sum(len(c["source_chunk_ids"]) for c in e["section_cards"])
             or 1 for e in per_doc],
            total_budget,
        )
        selected: list[str] = []
        for e, budget in zip(per_doc, doc_budgets):
            selected.extend(self._round_robin_sections(e["section_cards"], budget))
        # Fill remaining slots from document-card provenance, in order.
        if len(selected) < total_budget:
            seen = set(selected)
            for e in per_doc:
                for cid in e["doc_provenance"]:
                    if len(selected) >= total_budget:
                        break
                    if cid not in seen:
                        seen.add(cid)
                        selected.append(cid)
        return selected[:total_budget]

    @staticmethod
    def _round_robin_sections(section_cards: list[dict], budget: int) -> list[str]:
        pools = [list(dict.fromkeys(c["source_chunk_ids"]))
                 for c in section_cards]
        out: list[str] = []
        seen: set[str] = set()
        idx = 0
        while len(out) < budget:
            progressed = False
            for pool in pools:
                if len(out) >= budget:
                    break
                if idx < len(pool) and pool[idx] not in seen:
                    seen.add(pool[idx])
                    out.append(pool[idx])
                    progressed = True
            if not progressed:
                break
            idx += 1
        return out


# ---------------------------------------------------------------------------
# Card shape validation (query-time re-check)
# ---------------------------------------------------------------------------


def _valid_section_card_shape(card: dict) -> bool:
    return (
        isinstance(card, dict)
        and str(card.get("title") or "").strip()
        and str(card.get("summary") or "").strip()
        and isinstance(card.get("points"), list)
        and bool(card["points"])
    )


def _valid_document_card_shape(card: dict) -> bool:
    return (
        isinstance(card, dict)
        and str(card.get("title") or "").strip()
        and str(card.get("summary") or "").strip()
        and isinstance(card.get("major_topics"), list)
        and bool(card["major_topics"])
    )


def hydrate_chunks(document_ids: list[str], chunk_ids: list[str]) -> list[dict]:
    """Resolve evidence ids to canonical chunk rows, enforcing scope.

    Raises ValueError when any id cannot be resolved inside the authorized
    document set — the caller falls back to the 6/4 route rather than
    answer from partial evidence.
    """
    allowed = set(document_ids)
    rows: list[dict] = []
    for cid in chunk_ids:
        rec = get_chunk_by_id(cid)
        if not rec or rec["document_id"] not in allowed:
            raise ValueError(f"chunk {cid} not in authorized document set")
        rows.append(rec)
    return rows
