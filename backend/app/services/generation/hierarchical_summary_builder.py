"""Hierarchical summary builder — traceable two-level overview index.

Roadmap Phase 3. Builds the derived hierarchy above the canonical raw
chunks:

    raw chunks -> section/window cards -> document overview card

Design rules enforced here:
  - Raw chunks remain canonical evidence; cards carry raw-chunk provenance
    on every generated point, never the other way around.
  - No sampling while building leaf summaries: every accepted body chunk is
    assigned to exactly one section or fallback window and processed once.
  - Structure before clustering: the source's chapter/section hierarchy is
    used when reliable, deterministic contiguous windows otherwise.
  - Build once, read many: LLM work happens at construction time only.
  - Failure is non-destructive: a failed build leaves the previous complete
    build (and raw retrieval) untouched.
  - Cache identity is explicit: source content hash + ordered chunk IDs and
    hashes + prompt version + builder version. Changing models alone does
    not invalidate a cache, but --force allows a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ...repositories.chunk_repository import get_chunks_by_document
from ...repositories.document_repository import get_document
from ...repositories.summary_node_repository import (
    DOCUMENT_SECTION_KEY,
    get_build_signature,
    get_document_nodes,
    replace_document_nodes,
)
from ...services.llm import LLMConfig
from ...services.llm.client import create_llm_client
from ...services.pdf_ingestion.structure_metadata import (
    is_unreliable_structure_status,
)
from ...services.retrieval.base import RetrievedChunk
from .sampling import group_chunks_by_major_section, is_body_chunk

logger = logging.getLogger(__name__)

# Frozen build identities. Increment PROMPT_VERSION whenever a prompt below
# changes meaningfully (it invalidates caches via the build signature).
BUILDER_VERSION = "2"
PROMPT_VERSION = "3"

# Bounded, deterministic budgets (frozen for evaluation runs).
# Concurrent section-summary calls per document. Lower it if the provider
# starts returning rate-limit errors.
SECTION_BUILD_WORKERS = 8
SECTION_BATCH_CHUNKS = 12   # chunks per section-summarization prompt batch
DOCUMENT_BATCH_CARDS = 6    # section cards per document-card prompt batch
WINDOW_SIZE = 10            # contiguous fallback window (allowed range 8-12)
MAX_CARD_RETRIES = 1        # one bounded retry for malformed structured output
# Output budgets for structured cards: the app-wide LLM_MAX_TOKENS is tuned
# for compact planner JSON and truncates full card output (finish=length).
SECTION_CARD_MAX_TOKENS = 4096
DOCUMENT_CARD_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Grouping manifest
# ---------------------------------------------------------------------------


@dataclass
class SectionGroupManifest:
    key: str
    chunk_ids: list[str]  # ordered by chunk_index
    page_start: int = 0
    page_end: int = 0


@dataclass
class BuildManifest:
    grouping_mode: str  # "section" | "window"
    groups: list[SectionGroupManifest]
    accepted_chunk_ids: list[str]  # every accepted body chunk, in order
    excluded_chunk_ids: list[str]  # front/back matter, counted not processed
    total_chunks: int = 0


def build_manifest(
    chunk_rows: list[dict],
    structure_status: str,
) -> BuildManifest:
    """Assign every chunk to exactly one group (or to the excluded set).

    ``chunk_rows`` must be the document's rows from
    ``get_chunks_by_document`` (already ordered by chunk_index). Ordering of
    groups follows the first chunk's chunk_index; page numbers are display
    metadata only. Falls back to contiguous windows when the structure is
    unreliable or yields fewer than two meaningful groups.
    """
    chunks = [RetrievedChunk(
        chunk_id=r["id"],
        document_id=r["document_id"],
        text=r["text"],
        title=r.get("title", ""),
        section_path=r.get("section_path", ""),
        page_start=r.get("page_start", -1),
        page_end=r.get("page_end", -1),
    ) for r in chunk_rows]

    body = [c for c in chunks if is_body_chunk(c)]
    excluded = [c for c in chunks if not is_body_chunk(c)]

    index_by_id = {r["id"]: r.get("chunk_index", i)
                   for i, r in enumerate(chunk_rows)}
    page_by_id = {c.chunk_id: (c.page_start, c.page_end) for c in chunks}

    # Reuses the existing major-section grouping; groups are re-ordered by
    # their first chunk's chunk_index (page numbers stay display metadata).
    section_groups = group_chunks_by_major_section(body)
    use_sections = (
        not is_unreliable_structure_status(structure_status)
        and len(section_groups) >= 2
    )

    groups: list[SectionGroupManifest] = []
    if use_sections:
        items = sorted(
            section_groups,
            key=lambda g: min(index_by_id[c.chunk_id] for c in g.chunks),
        )
        for sg in items:
            ids = [c.chunk_id for c in sg.chunks]
            pages = [page_by_id[c.chunk_id] for c in sg.chunks]
            starts = [p[0] for p in pages if p[0] > 0]
            ends = [p[1] for p in pages if p[1] > 0]
            groups.append(SectionGroupManifest(
                key=sg.key,
                chunk_ids=ids,
                page_start=starts[0] if starts else 0,
                page_end=ends[-1] if ends else 0,
            ))
        mode = "section"
    else:
        # Deterministic contiguous windows over the ordered body chunks.
        for i in range(0, len(body), WINDOW_SIZE):
            win = body[i:i + WINDOW_SIZE]
            ids = [c.chunk_id for c in win]
            groups.append(SectionGroupManifest(
                key=f"window-{i + 1}-{i + len(win)}",
                chunk_ids=ids,
            ))
        mode = "window"

    accepted = [c.chunk_id for c in body]
    assert len(accepted) == sum(len(g.chunk_ids) for g in groups), (
        "manifest must assign every accepted body chunk exactly once"
    )
    assert len(set(accepted)) == len(accepted), "duplicate chunk assignment"
    assert all(
        g.chunk_ids == sorted(g.chunk_ids, key=lambda cid: index_by_id[cid])
        for g in groups
    ), "group chunk ordering must follow chunk_index"

    return BuildManifest(
        grouping_mode=mode,
        groups=groups,
        accepted_chunk_ids=accepted,
        excluded_chunk_ids=[c.chunk_id for c in excluded],
        total_chunks=len(chunk_rows),
    )


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------


def compute_build_signature(doc_content_hash: str, chunk_rows: list[dict]) -> str:
    """Deterministic cache identity for one document's hierarchy.

    Includes the source document content hash, the ordered source chunk IDs
    and content hashes, the prompt version, and the builder version. Changing
    the model alone does not invalidate a cache.
    """
    payload = {
        "doc_content_hash": doc_content_hash,
        "chunks": [
            [r["id"], r.get("content_hash", "")]
            for r in chunk_rows  # ordered by chunk_index from the repository
        ],
        "prompt_version": PROMPT_VERSION,
        "builder_version": BUILDER_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Structured card validation
# ---------------------------------------------------------------------------


def _coerce_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if isinstance(x, str) and x.strip()]


def _validate_section_card(
    raw: object,
    allowed_chunk_ids: set[str],
) -> dict | None:
    """Validate one LLM section card; discard unknown/cross-document IDs.

    Every retained point must carry at least one valid raw chunk ID from the
    batch it was generated from. Returns the cleaned card dict, or None when
    the card cannot be validated.
    """
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or "").strip()
    summary = str(raw.get("summary") or "").strip()
    if not title or not summary:
        return None
    raw_points = raw.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        return None
    points: list[dict] = []
    for p in raw_points:
        if not isinstance(p, dict):
            continue
        text = str(p.get("text") or "").strip()
        if not text:
            continue
        ids = [str(i) for i in (p.get("source_chunk_ids") or [])
               if str(i) in allowed_chunk_ids]
        if not ids:
            continue  # invented or cross-document provenance — discarded
        points.append({"text": text, "source_chunk_ids": ids})
    if not points:
        return None
    card: dict = {"title": title, "summary": summary, "points": points}
    themes = _coerce_str_list(raw.get("themes"))
    entities = _coerce_str_list(raw.get("entities"))
    if themes:
        card["themes"] = themes
    if entities:
        card["entities"] = entities
    return card


def _validate_document_card(
    raw: object,
    section_node_ids: set[str],
    allowed_chunk_ids: set[str],
) -> dict | None:
    """Validate one LLM document card; every topic needs valid provenance."""
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or "Document overview").strip()
    summary = str(raw.get("summary") or "").strip()
    if not title or not summary:
        return None
    raw_topics = raw.get("major_topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        return None
    topics: list[dict] = []
    for t in raw_topics:
        if not isinstance(t, dict):
            continue
        text = str(t.get("text") or "").strip()
        if not text:
            continue
        section_ids = [str(i) for i in (t.get("section_node_ids") or [])
                       if str(i) in section_node_ids]
        chunk_ids = [str(i) for i in (t.get("source_chunk_ids") or [])
                     if str(i) in allowed_chunk_ids]
        if not section_ids and not chunk_ids:
            continue  # no valid provenance — discarded
        topics.append({
            "text": text,
            "section_node_ids": section_ids,
            "source_chunk_ids": chunk_ids,
        })
    if not topics:
        return None
    return {"title": title, "summary": summary, "major_topics": topics}


def _merge_partial_cards(partials: list[dict]) -> dict:
    """Reduce ordered partial cards into one card (union of provenance)."""
    merged_points: list[dict] = []
    seen_points: set[str] = set()
    themes: list[str] = []
    entities: list[str] = []
    summaries: list[str] = []
    for card in partials:
        summaries.append(card["summary"])
        for p in card.get("points", []):
            if p["text"] in seen_points:
                continue
            seen_points.add(p["text"])
            merged_points.append(p)
        for t in card.get("themes", []):
            if t not in themes:
                themes.append(t)
        for e in card.get("entities", []):
            if e not in entities:
                entities.append(e)
    merged: dict = {
        "title": partials[0]["title"],
        "summary": "\n".join(summaries),
        "points": merged_points,
    }
    if themes:
        merged["themes"] = themes
    if entities:
        merged["entities"] = entities
    return merged


def _merge_partial_documents(partials: list[dict]) -> dict:
    merged_topics: list[dict] = []
    seen: set[str] = set()
    summaries: list[str] = []
    for card in partials:
        summaries.append(card["summary"])
        for t in card.get("major_topics", []):
            if t["text"] in seen:
                continue
            seen.add(t["text"])
            merged_topics.append(t)
    return {
        "title": "Document overview",
        "summary": "\n".join(summaries),
        "major_topics": merged_topics,
    }


def _enrich_document_card_provenance(
    doc_card: dict,
    section_cards: list[tuple[SectionGroupManifest, dict]],
    section_node_ids: list[str],
) -> None:
    """Ensure every document-card topic carries raw-chunk provenance.

    Topics that cite only section nodes get the union of those sections'
    validated raw provenance (bounded). Deterministic — no extra LLM work.
    """
    chunk_ids_by_node: dict[str, list[str]] = {}
    for (_group, card), node_id in zip(section_cards, section_node_ids, strict=False):
        ids: list[str] = []
        for p in card.get("points", []):
            ids.extend(p.get("source_chunk_ids", []))
        chunk_ids_by_node[node_id] = list(dict.fromkeys(ids))
    for t in doc_card.get("major_topics", []):
        if t.get("source_chunk_ids"):
            continue
        derived: list[str] = []
        for node_id in t.get("section_node_ids", []):
            for cid in chunk_ids_by_node.get(node_id, []):
                if cid not in derived:
                    derived.append(cid)
        t["source_chunk_ids"] = derived[:4]


# ---------------------------------------------------------------------------
# Prompts (PROMPT_VERSION above covers changes to these)
# ---------------------------------------------------------------------------

_SECTION_CARD_SYSTEM_PROMPT = """You are building a traceable section index for a study companion. You receive passages from ONE section of a source document. Produce a concise structured card.

Return ONLY valid JSON:
{
  "title": "The section's heading",
  "summary": "A concise account of this section in 2-4 sentences.",
  "points": [
    {"text": "One important event, argument, or idea.", "source_chunk_ids": ["<passage-id>"]}
  ],
  "themes": ["short theme label"],
  "entities": ["notable person, place, or work"]
}

Rules:
1. Use ONLY the passages provided. Do not invent content.
2. Every point MUST cite at least one passage id from the list below. Copy the ids verbatim. A point may cite multiple ids.
3. Do not invent passage ids.
4. Keep the summary and points faithful to the passages, in reading order where possible.
5. Keep the output compact: at most 4-8 points, each 1-2 sentences."""


_DOCUMENT_CARD_SYSTEM_PROMPT = """You are building a traceable document overview for a study companion. You receive section cards from ONE source document, each with its node id and the raw passage ids it covers.

Return ONLY valid JSON:
{
  "title": "Document overview",
  "summary": "A concise whole-document synthesis.",
  "major_topics": [
    {"text": "A cross-section theme or argument.", "section_node_ids": ["<node-id>"], "source_chunk_ids": ["<passage-id>"]}
  ]
}

Rules:
1. Use ONLY the section cards provided. Do not invent content.
2. Every major topic MUST cite at least one section node id or raw passage id from the cards below. Copy ids verbatim.
3. Do not invent node ids or passage ids.
4. Prefer topics that connect multiple sections.
5. Keep the output compact: at most 6 topics, each 1-2 sentences."""


def _section_batch_user_prompt(
    section_key: str,
    passages: list[dict],
) -> str:
    lines = [f"Section: {section_key}", "", "Passages:"]
    for p in passages:
        loc = (
            f"p. {p['page_start']}" if p["page_start"] == p["page_end"]
            else f"pp. {p['page_start']}-{p['page_end']}"
        ) if p.get("page_start", -1) > 0 else ""
        header = f"[{p['id']}] {p.get('section_path') or p.get('title') or ''}"
        if loc:
            header += f" ({loc})"
        lines.append(header)
        lines.append(p["text"][:2000])
    return "\n".join(lines)


def _document_user_prompt(section_cards: list[dict]) -> str:
    lines = ["Section cards (node ids and passage ids are authoritative):"]
    for card in section_cards:
        lines.append(json.dumps(card, ensure_ascii=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    document_id: str
    status: str = "failed"  # built | up_to_date | failed | skipped_llm
    grouping_mode: str = ""
    group_count: int = 0
    node_count: int = 0
    llm_calls: int = 0
    cache_hit: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class HierarchicalSummaryBuilder:
    """Build the two-level summary hierarchy for one document."""

    def __init__(self, llm_client=None):
        # Injectable for fake-LLM tests; defaults to the configured provider.
        self._llm = llm_client
        self._config = LLMConfig()
        self._model_id = self._config.model

    # -- public -------------------------------------------------------------

    def build_document(
        self,
        notebook_id: str,
        document_id: str,
        force: bool = False,
    ) -> BuildResult:
        """Build (or reuse) the hierarchy for one document.

        Never raises: every failure returns a BuildResult with status
        "failed" and a reason, leaving the previous complete build active.
        """
        try:
            return self._build(notebook_id, document_id, force)
        except Exception as exc:  # noqa: BLE001 — non-destructive by design
            logger.error("Hierarchy build failed for %s: %s", document_id, exc)
            return BuildResult(
                document_id=document_id, status="failed", reason=str(exc),
            )

    # -- internals ----------------------------------------------------------

    def _build(
        self,
        notebook_id: str,
        document_id: str,
        force: bool,
    ) -> BuildResult:
        doc = get_document(document_id)
        if not doc or doc.get("notebook_id") != notebook_id:
            return BuildResult(
                document_id=document_id,
                status="failed",
                reason="document not found in notebook",
            )
        chunk_rows = get_chunks_by_document(document_id)
        if not chunk_rows:
            return BuildResult(
                document_id=document_id,
                status="failed",
                reason="document has no chunks",
            )

        signature = compute_build_signature(
            doc.get("content_hash", ""), chunk_rows,
        )

        if not force and self._is_complete_current(notebook_id, document_id, signature):
            counts = self._stored_counts(notebook_id, document_id)
            return BuildResult(
                document_id=document_id,
                status="up_to_date",
                grouping_mode=self._stored_grouping_mode(notebook_id, document_id),
                group_count=counts.get("section", 0),
                node_count=sum(counts.values()),
                cache_hit=True,
            )

        if not self._config.is_configured:
            return BuildResult(
                document_id=document_id,
                status="skipped_llm",
                reason="LLM is not configured; hierarchical summaries disabled",
            )

        structure_status = doc.get("structure_status", "")
        manifest = build_manifest(chunk_rows, structure_status)
        if not manifest.groups:
            return BuildResult(
                document_id=document_id,
                status="failed",
                reason="no accepted body chunks to summarize",
            )

        chunk_ids = {r["id"] for r in chunk_rows}
        by_id = {r["id"]: r for r in chunk_rows}

        # ponytail: per-section LLM cost is linear in section count (a
        # 246-chunk slide deck costs ~sections + sections/8 calls). Accepted
        # at single-user desktop scale with per-document cost reporting; add
        # a build budget or batched construction if it ever becomes one.
        llm_calls = 0
        section_cards: list[tuple[SectionGroupManifest, dict]] = []
        # Sections are independent, so summarize them concurrently: wall time
        # drops ~SECTION_BUILD_WORKERS-fold at identical token cost. map()
        # keeps manifest order. The client is created first so worker threads
        # never race on the lazy init.
        if self._llm is None:
            self._llm = create_llm_client(self._config)
        failed = threading.Event()  # fail fast: stop queued sections once one fails

        def summarize(group: SectionGroupManifest) -> tuple[dict | None, int]:
            if failed.is_set():
                return None, 0
            card, calls = self._summarize_group(group, by_id)
            if card is None:
                failed.set()
            return card, calls

        with ThreadPoolExecutor(max_workers=SECTION_BUILD_WORKERS) as pool:
            outcomes = list(pool.map(summarize, manifest.groups))
        for group, (card, calls) in zip(manifest.groups, outcomes, strict=True):
            llm_calls += calls
            if card is None:
                return BuildResult(
                    document_id=document_id,
                    status="failed",
                    grouping_mode=manifest.grouping_mode,
                    llm_calls=llm_calls,
                    reason=f"section card validation failed: {group.key}",
                )
            section_cards.append((group, card))

        # Section node ids are assigned before the document card is built so
        # the LLM can cite them; the document card's provenance is validated
        # against exactly these ids before commit.
        section_node_ids = [str(uuid.uuid4()) for _ in section_cards]
        doc_card, calls = self._summarize_document(
            section_cards, section_node_ids, chunk_ids,
        )
        llm_calls += calls
        if doc_card is None:
            return BuildResult(
                document_id=document_id,
                status="failed",
                grouping_mode=manifest.grouping_mode,
                llm_calls=llm_calls,
                reason="document card validation failed",
            )
        # The document card must carry raw-chunk provenance in addition to
        # section-node provenance. When a topic cites only section nodes,
        # derive the raw ids deterministically from those sections' own
        # validated provenance.
        _enrich_document_card_provenance(doc_card, section_cards, section_node_ids)

        nodes: list[dict] = []
        for (group, card), node_id in zip(section_cards, section_node_ids, strict=False):
            provenance: list[str] = []
            for p in card["points"]:
                provenance.extend(p["source_chunk_ids"])
            nodes.append({
                "id": node_id,  # stored as-is so document-card provenance
                                # references real stored section node ids
                "level": "section",
                "section_key": group.key,
                "ordinal": len(nodes) + 1,
                "summary_json": json.dumps(card, ensure_ascii=False),
                "source_chunk_ids": list(dict.fromkeys(provenance)),
                "child_node_ids": [],
                "build_signature": signature,
                "prompt_version": PROMPT_VERSION,
                "model_id": self._model_id,
            })
        nodes.append({
            "id": str(uuid.uuid4()),
            "level": "document",
            "section_key": DOCUMENT_SECTION_KEY,
            "ordinal": 0,
            "summary_json": json.dumps(
                {**doc_card, "grouping_mode": manifest.grouping_mode},
                ensure_ascii=False,
            ),
            "source_chunk_ids": list(dict.fromkeys(
                cid for t in doc_card["major_topics"]
                for cid in t["source_chunk_ids"]
            )),
            "child_node_ids": section_node_ids,
            "build_signature": signature,
            "prompt_version": PROMPT_VERSION,
            "model_id": self._model_id,
        })

        replace_document_nodes(notebook_id, document_id, nodes)
        return BuildResult(
            document_id=document_id,
            status="built",
            grouping_mode=manifest.grouping_mode,
            group_count=len(section_cards),
            node_count=len(nodes),
            llm_calls=llm_calls,
            reason=(
                f"excluded {len(manifest.excluded_chunk_ids)}/"
                f"{manifest.total_chunks} front/back-matter chunks"
            ),
        )

    def _summarize_group(
        self,
        group: SectionGroupManifest,
        by_id: dict,
    ) -> tuple[dict | None, int]:
        """Summarize one group in bounded batches; reduce partials into one card."""
        calls = 0
        partials: list[dict] = []
        passages = [by_id[cid] for cid in group.chunk_ids]
        for i in range(0, len(passages), SECTION_BATCH_CHUNKS):
            batch = passages[i:i + SECTION_BATCH_CHUNKS]
            # Provenance is validated against the prompt batch, so a point
            # can only cite a passage the model actually saw ("supporting
            # chunk IDs from that batch"); unknown and cross-document ids
            # are discarded by the same check.
            batch_ids = {p["id"] for p in batch}
            card, n = self._llm_structured_card(
                system=_SECTION_CARD_SYSTEM_PROMPT,
                user=_section_batch_user_prompt(group.key, batch),
                validate=lambda raw, batch_ids=batch_ids: _validate_section_card(raw, batch_ids),
                max_tokens=SECTION_CARD_MAX_TOKENS,
            )
            calls += n
            if card is None:
                return None, calls
            partials.append(card)
        if len(partials) == 1:
            return partials[0], calls
        return _merge_partial_cards(partials), calls

    def _summarize_document(
        self,
        section_cards: list[tuple[SectionGroupManifest, dict]],
        section_node_ids: list[str],
        chunk_ids: set[str],
    ) -> tuple[dict | None, int]:
        """Build the document card from all section cards in ordered batches.

        Every section card is included: batches cover the ordered list in
        full, and middle sections are never omitted.
        """
        calls = 0
        partials: list[dict] = []
        view = [
            {"node_id": node_id, "section_key": group.key, **card}
            for (group, card), node_id in zip(section_cards, section_node_ids, strict=False)
        ]
        node_id_set = set(section_node_ids)
        for i in range(0, len(view), DOCUMENT_BATCH_CARDS):
            batch = view[i:i + DOCUMENT_BATCH_CARDS]
            card, n = self._llm_structured_card(
                system=_DOCUMENT_CARD_SYSTEM_PROMPT,
                user=_document_user_prompt(batch),
                validate=lambda raw: _validate_document_card(
                    raw, node_id_set, chunk_ids,
                ),
                max_tokens=DOCUMENT_CARD_MAX_TOKENS,
            )
            calls += n
            if card is None:
                return None, calls
            partials.append(card)
        if len(partials) == 1:
            return partials[0], calls
        return _merge_partial_documents(partials), calls

    def _llm_structured_card(
        self,
        system: str,
        user: str,
        validate,
        max_tokens: int | None = None,
    ) -> tuple[dict | None, int]:
        """One LLM structured call with exactly one bounded retry."""
        if self._llm is None:
            self._llm = create_llm_client(self._config)
        last_error: Exception | None = None
        for attempt in range(MAX_CARD_RETRIES + 1):
            try:
                result = self._llm.complete_json(
                    system_prompt=system,
                    user_prompt=user,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            card = validate(result.data if result else {})
            if card is not None:
                return card, attempt + 1
            last_error = ValueError("structured output failed validation")
        logger.warning("Structured card failed after %d attempt(s): %s",
                       MAX_CARD_RETRIES + 1, last_error)
        return None, MAX_CARD_RETRIES + 1

    # -- cache inspection ---------------------------------------------------

    @staticmethod
    def _is_complete_current(
        notebook_id: str,
        document_id: str,
        signature: str,
    ) -> bool:
        if get_build_signature(notebook_id, document_id) != signature:
            return False
        nodes = get_document_nodes(notebook_id, document_id)
        levels = [n["level"] for n in nodes]
        return "document" in levels and levels.count("section") >= 1

    @staticmethod
    def _stored_counts(notebook_id: str, document_id: str) -> dict[str, int]:
        from ...repositories.summary_node_repository import count_document_nodes
        return count_document_nodes(notebook_id, document_id)

    @staticmethod
    def _stored_grouping_mode(notebook_id: str, document_id: str) -> str:
        nodes = get_document_nodes(notebook_id, document_id)
        for n in nodes:
            if n["level"] == "document":
                try:
                    return json.loads(n["summary_json"]).get("grouping_mode", "")
                except (json.JSONDecodeError, TypeError):
                    return ""
        return ""
