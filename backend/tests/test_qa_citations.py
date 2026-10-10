"""Ask-path citation finalization — ref-label renumbering + dangling-drop + title.

Guards the fix that made returned S#/W# labels match the passages actually shown
to the LLM (≤10 book chunks / ≤8 web results), so labels never collide across
rounds or dangle past what was shown, and book `title` stays a distinct field.
"""

from __future__ import annotations

from app.services.qa.tools import (
    MAX_BOOK_CITATIONS,
    MAX_WEB_CITATIONS,
    ToolResults,
    finalize_merged_citations,
)
from app.services.retrieval.base import RetrievedChunk


def make_chunk(cid: str, text: str, title: str = "", section_path: str = "", page: int = 1):
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", text=text, title=title,
        section_path=section_path, page_start=page, page_end=page,
    )


def test_finalize_renumbers_to_first_ten_and_drops_dangling():
    # 12 chunks -> only the first 10 should be cited, renumbered S1..S10.
    chunks = [make_chunk(f"c{i}", f"body {i}") for i in range(12)]
    tr = ToolResults(chunks=chunks)
    # Simulate colliding per-round labels (two rounds each restarting at S1).
    tr.chunk_citations = [
        {"source_type": "book", "ref_label": "S1", "chunk_id": "c0"},
        {"source_type": "book", "ref_label": "S2", "chunk_id": "c11"},  # collides
    ]

    finalize_merged_citations(tr)

    assert [c["ref_label"] for c in tr.chunk_citations] == [
        f"S{i}" for i in range(1, MAX_BOOK_CITATIONS + 1)
    ]
    cited_ids = {c["chunk_id"] for c in tr.chunk_citations}
    assert cited_ids == {f"c{i}" for i in range(MAX_BOOK_CITATIONS)}
    # The dangling chunks past the LLM window are not cited.
    assert "c10" not in cited_ids and "c11" not in cited_ids


def test_finalize_preserves_title_distinct_from_section_path():
    chunk = make_chunk(
        "c1", "body", title="Introduction", section_path="Ch 1 > Introduction",
    )
    tr = ToolResults(chunks=[chunk])
    finalize_merged_citations(tr)
    c = tr.chunk_citations[0]
    assert c["title"] == "Introduction"
    assert c["section_path"] == "Ch 1 > Introduction"


def test_finalize_renumbers_web_results():
    tr = ToolResults()
    tr.web_results = [
        {"title": "A", "url": "https://a", "site_name": "a.com",
         "content": "some content", "text_snippet": "a snippet"},
        {"title": "B", "url": "https://b", "site_name": "b.com",
         "content": "other content", "text_snippet": "b snippet"},
    ]
    finalize_merged_citations(tr)
    web = [c for c in tr.chunk_citations if c["source_type"] == "web"]
    assert [c["ref_label"] for c in web] == ["W1", "W2"]
    assert [c["title"] for c in web] == ["A", "B"]
    assert [c["url"] for c in web] == ["https://a", "https://b"]
    assert len(web) <= MAX_WEB_CITATIONS
