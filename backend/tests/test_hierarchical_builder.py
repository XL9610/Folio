"""Tests for the hierarchical summary builder (roadmap Phase 3, Step 2).

Fake-LLM driven: no network, no real model. Covers the deterministic
grouping manifest, cache identity, structured-output validation (one-retry
limit, invented/cross-document ID rejection), end-to-end section + document
card builds, atomic replacement, and the backfill/cache path.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest

from app.db import session as db_session
from app.db.schema import create_tables
from app.repositories.chunk_repository import get_chunks_by_document, insert_chunks
from app.repositories.document_repository import create_source_document
from app.repositories.notebook_repository import create_notebook
from app.repositories.summary_node_repository import get_document_nodes
from app.services.generation.hierarchical_summary_builder import (
    PROMPT_VERSION,
    HierarchicalSummaryBuilder,
    build_manifest,
    compute_build_signature,
)

_ID_RE = re.compile(r"\[([0-9a-f]{8}-[0-9a-f-]{27,})\]")


class FakeLLM:
    """Structured JSON fake that echoes real ids out of each prompt."""

    def __init__(self, fail_attempts: int = 0):
        self.calls = 0
        self.fail_attempts = fail_attempts  # number of leading attempts to fail

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls <= self.fail_attempts:
            raise ValueError("fake provider failure")
        ids = _ID_RE.findall(user_prompt)
        if "section cards" in system_prompt:
            # Document-card prompt: node ids appear as "node_id" fields.
            node_ids = re.findall(r'"node_id": "([0-9a-f-]+)"', user_prompt)
            return _resp({
                "title": "Document overview",
                "summary": "whole document",
                "major_topics": [
                    {"text": "topic one", "section_node_ids": node_ids[:1],
                     "source_chunk_ids": []},
                ],
            })
        section = (user_prompt.splitlines()[0] or "").replace("Section: ", "")
        return _resp({
            "title": section,
            "summary": f"summary of {section}",
            "points": [
                {"text": f"point a of {section}", "source_chunk_ids": ids[:1]},
                {"text": f"point b of {section}", "source_chunk_ids": ids[1:2]},
            ],
            "themes": ["t1"],
            "entities": ["e1"],
        })


class _FakeResp:
    def __init__(self, data):
        self.data = data


def _resp(data) -> _FakeResp:
    return _FakeResp(data)


class InventingLLM(FakeLLM):
    """Always returns points whose ids do not exist in the document."""

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        return _resp({
            "title": "X",
            "summary": "s",
            "points": [{"text": "p", "source_chunk_ids": ["invented-1", "bogus"]}],
        })


@pytest.fixture
def test_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    original_conn = getattr(db_session._local, 'conn', None)
    db_session._local.conn = conn
    create_tables(conn)
    yield conn
    db_session._local.conn = original_conn
    conn.close()


def _seed_doc(test_db, nb_id, filename="book.pdf", structure_status="embedded_toc",
              content_hash="doc-hash"):
    doc = create_source_document(
        nb_id, filename, page_count=10, content_hash=content_hash,
        structure_status=structure_status,
    )
    chunks = []
    for i in range(14):
        chapter = f"Chapter {i // 5 + 1}"
        chunks.append({
            "chunk_index": i,
            "text": f"Body text {i} about {chapter.lower()} material. " * 8,
            "title": chapter,
            "section_path": f"{chapter} > Section {i}",
            "page_start": i + 1,
            "page_end": i + 1,
            "chunk_strategy": "test",
        })
    # Front/back matter: two chunks that must be excluded from the manifest.
    for i, title in enumerate(("Contents", "Index")):
        chunks.append({
            "chunk_index": 100 + i,
            "text": f"{title} filler text. " * 8,
            "title": title,
            "section_path": title,
            "page_start": 0,
            "page_end": 0,
            "chunk_strategy": "test",
        })
    insert_chunks(doc["id"], chunks)
    return doc


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_assigns_every_body_chunk_exactly_once(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    rows = get_chunks_by_document(doc["id"])
    m = build_manifest(rows, "embedded_toc")
    assert m.grouping_mode == "section"
    assert len(m.groups) == 3  # 14 body chunks / 5 per chapter -> 3 chapters
    assert len(m.accepted_chunk_ids) == 14
    assert len(m.excluded_chunk_ids) == 2  # Contents + Index counted
    assert m.total_chunks == 16
    assigned = [cid for g in m.groups for cid in g.chunk_ids]
    assert sorted(assigned) == sorted(m.accepted_chunk_ids)
    assert len(set(assigned)) == len(assigned)


def test_manifest_orders_groups_by_chunk_index_not_page(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    rows = get_chunks_by_document(doc["id"])
    m = build_manifest(rows, "embedded_toc")
    # Group order must follow the first chunk's chunk_index, which here is
    # the same order as chapter numbering.
    assert [g.key for g in m.groups] == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_manifest_windows_when_structure_unreliable(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"], structure_status="unknown_legacy")
    m = build_manifest(get_chunks_by_document(doc["id"]), "unknown_legacy")
    assert m.grouping_mode == "window"
    # 14 body chunks, window size 10 -> windows of 10 and 4.
    assert [len(g.chunk_ids) for g in m.groups] == [10, 4]
    assert m.groups[0].key == "window-1-10"


def test_manifest_windows_when_fewer_than_two_sections(test_db):
    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "flat.pdf", content_hash="h")
    insert_chunks(doc["id"], [{
        "chunk_index": i, "text": "flat body text " * 10,
        "title": "Flat", "section_path": "Flat",
        "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "test",
    } for i in range(12)])
    m = build_manifest(get_chunks_by_document(doc["id"]), "embedded_toc")
    assert m.grouping_mode == "window"


def test_manifest_empty_body(test_db):
    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "toc-only.pdf", content_hash="h")
    insert_chunks(doc["id"], [{
        "chunk_index": 0, "text": "contents listing " * 5,
        "title": "Contents", "section_path": "Contents",
        "page_start": 1, "page_end": 1, "chunk_strategy": "test",
    }])
    m = build_manifest(get_chunks_by_document(doc["id"]), "embedded_toc")
    assert m.groups == []
    assert m.accepted_chunk_ids == []
    assert len(m.excluded_chunk_ids) == 1


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------


def test_signature_stable_and_sensitive_to_chunks(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    rows = get_chunks_by_document(doc["id"])
    sig1 = compute_build_signature(doc["content_hash"], rows)
    sig2 = compute_build_signature(doc["content_hash"], rows)
    assert sig1 == sig2
    # Change one chunk's content hash -> signature changes.
    rows[0]["content_hash"] = "changed"
    assert compute_build_signature(doc["content_hash"], rows) != sig1
    # Different document hash -> signature changes.
    assert compute_build_signature("other-hash", rows) != sig1


# ---------------------------------------------------------------------------
# Builder end to end (fake LLM)
# ---------------------------------------------------------------------------


def test_fake_build_end_to_end(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    fake = FakeLLM()
    result = HierarchicalSummaryBuilder(llm_client=fake).build_document(
        nb["id"], doc["id"],
    )
    assert result.status == "built", result.reason
    assert result.grouping_mode == "section"
    assert result.group_count == 3
    assert result.node_count == 4
    assert fake.calls == 3 + 1  # one call per section + one document call

    nodes = get_document_nodes(nb["id"], doc["id"])
    sections = [n for n in nodes if n["level"] == "section"]
    docs = [n for n in nodes if n["level"] == "document"]
    assert len(sections) == 3 and len(docs) == 1

    # Every stored provenance id resolves to a raw chunk in this document.
    chunk_ids = {r["id"] for r in get_chunks_by_document(doc["id"])}
    for n in nodes:
        assert set(n["source_chunk_ids"]) <= chunk_ids
    # Document card must carry both section-node and raw-chunk provenance.
    doc_card = json.loads(docs[0]["summary_json"])
    assert doc_card["grouping_mode"] == "section"
    assert docs[0]["child_node_ids"] == [n["id"] for n in sections]
    topic = doc_card["major_topics"][0]
    assert set(topic["section_node_ids"]) <= {n["id"] for n in sections}

    # Second build is a cache hit (no new LLM calls) unless forced.
    again = HierarchicalSummaryBuilder(llm_client=fake).build_document(
        nb["id"], doc["id"],
    )
    assert again.status == "up_to_date"
    assert again.cache_hit is True
    assert fake.calls == 4
    forced = HierarchicalSummaryBuilder(llm_client=fake).build_document(
        nb["id"], doc["id"], force=True,
    )
    assert forced.status == "built"
    assert fake.calls == 8


def test_invented_ids_reject_build_and_preserve_previous(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    good = FakeLLM()
    builder = HierarchicalSummaryBuilder(llm_client=good)
    assert builder.build_document(nb["id"], doc["id"]).status == "built"

    bad = InventingLLM()
    result = HierarchicalSummaryBuilder(llm_client=bad).build_document(
        nb["id"], doc["id"], force=True,
    )
    assert result.status == "failed"
    assert "validation" in result.reason
    # Previous complete build is still active and readable.
    nodes = get_document_nodes(nb["id"], doc["id"])
    assert len(nodes) == 4


def test_one_retry_then_fail(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    # First attempt per card fails, second succeeds -> exactly 2 calls/card.
    fake = FakeLLM(fail_attempts=1)
    result = HierarchicalSummaryBuilder(llm_client=fake).build_document(
        nb["id"], doc["id"],
    )
    assert result.status == "built"
    assert fake.calls == 5  # 4 cards + 1 retried first attempt

    # Always-failing LLM: bounded at one retry per card, build fails fast.
    always = FakeLLM(fail_attempts=100)
    result2 = HierarchicalSummaryBuilder(llm_client=always).build_document(
        nb["id"], doc["id"], force=True,
    )
    assert result2.status == "failed"
    assert always.calls == 2  # one card, attempt + single retry, then abort


def test_window_grouping_build(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"], structure_status="unknown_legacy")
    fake = FakeLLM()
    result = HierarchicalSummaryBuilder(llm_client=fake).build_document(
        nb["id"], doc["id"],
    )
    assert result.status == "built"
    assert result.grouping_mode == "window"
    assert result.group_count == 2
    nodes = get_document_nodes(nb["id"], doc["id"])
    doc_card = json.loads(
        [n for n in nodes if n["level"] == "document"][0]["summary_json"]
    )
    assert doc_card["grouping_mode"] == "window"


def test_document_card_reduction_uses_all_sections_in_order(test_db):
    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "many.pdf", content_hash="h",
                       structure_status="embedded_toc")
    # 30 single-chunk sections -> section mode, 30 cards -> 4 document
    # reduction batches (DOCUMENT_BATCH_CARDS=8), none omitted.
    insert_chunks(doc["id"], [{
        "chunk_index": i, "text": f"chapter {i + 1} content. " * 10,
        "title": f"Chapter {i + 1}", "section_path": f"Chapter {i + 1}",
        "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "test",
    } for i in range(30)])

    class RecordingLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.doc_batches = []

        def complete_json(self, system_prompt, user_prompt):
            if "section cards" in system_prompt:
                self.doc_batches.append(len(re.findall(r'"node_id": "', user_prompt)))
            return super().complete_json(system_prompt, user_prompt)

    fake = RecordingLLM()
    result = HierarchicalSummaryBuilder(llm_client=fake).build_document(
        nb["id"], doc["id"],
    )
    assert result.status == "built"
    assert result.group_count == 30
    # 30 section calls + 4 document reduction calls, covering 8+8+8+6 cards.
    assert fake.calls == 34
    assert fake.doc_batches == [8, 8, 8, 6]
    nodes = get_document_nodes(nb["id"], doc["id"])
    doc_node = [n for n in nodes if n["level"] == "document"][0]
    assert len(doc_node["child_node_ids"]) == 30


def test_llm_disabled_skips_build(test_db):
    nb = create_notebook("NB")
    doc = _seed_doc(test_db, nb["id"])
    builder = HierarchicalSummaryBuilder(llm_client=FakeLLM())
    builder._config = _UnconfiguredConfig()
    result = builder.build_document(nb["id"], doc["id"])
    assert result.status == "skipped_llm"
    assert "not configured" in result.reason


class _UnconfiguredConfig:
    model = "unset"
    is_configured = False


def test_wrong_notebook_fails_scope(test_db):
    nb = create_notebook("NB A")
    nb2 = create_notebook("NB B")
    doc = _seed_doc(test_db, nb["id"])
    result = HierarchicalSummaryBuilder(llm_client=FakeLLM()).build_document(
        nb2["id"], doc["id"],
    )
    assert result.status == "failed"
    assert "notebook" in result.reason
