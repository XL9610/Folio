"""Summary generator `sources` — chunk-level citations + [N] label consistency.

Guards the fix that made `sources` correspond to the same sampled chunks the
prompt enumerated as [1]..[N] (previously it listed unsampled chunk_id strings).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.generation import summary_service
from app.services.generation.scope_resolver import ScopeResult
from app.services.generation.summary_service import SummaryGenerator
from app.services.qa.types import CitationItem


def make_chunk(cid: str, text: str, sec: str = "Sec", page: int = 1):
    from app.services.retrieval.base import RetrievedChunk

    return RetrievedChunk(
        chunk_id=cid, document_id="d1", text=text, title="",
        section_path=sec, page_start=page, page_end=page,
    )


def test_summary_sources_are_chunk_citations_matching_prompt_order(monkeypatch):
    chunks = [make_chunk("c1", "alpha", "Sec 1", 5), make_chunk("c2", "beta", "Sec 2", 9)]

    # Identity sampling so the prompt [N] order == source_chunks order.
    monkeypatch.setattr(
        summary_service, "sample_body_chunks", lambda cs, max_chunks=25: cs,
    )

    llm = MagicMock()
    llm.complete_chat.return_value = "## Summary\n\n[1] alpha, [2] beta."
    resolver = MagicMock()
    resolver.resolve.return_value = ScopeResult(chunks=chunks, metadata={"layer": "toc"})

    gen = SummaryGenerator(llm, resolver)
    result = gen.generate("nb1", "d1", "", doc_filename="book.pdf")

    sources = result["sources"]
    assert all(isinstance(c, CitationItem) for c in sources)
    assert [c.chunk_id for c in sources] == ["c1", "c2"]
    assert [c.ref_label for c in sources] == ["[1]", "[2]"]
    assert all(c.document_id == "d1" for c in sources)
    assert sources[0].section_path == "Sec 1"
    assert sources[0].page_start == 5
    assert sources[0].title == "book.pdf"
