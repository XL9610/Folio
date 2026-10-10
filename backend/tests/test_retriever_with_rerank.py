"""Verify HybridRetriever wires reranker correctly when provided."""

from unittest.mock import MagicMock
from app.services.retrieval.retriever import HybridRetriever
from app.services.retrieval.search_options import SearchOptions


def test_retriever_calls_reranker_when_provided():
    """When reranker is provided and options.rerank_enabled=True, HybridRetriever.search() should call reranker.rerank()."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384
    dense = MagicMock()
    dense.search.return_value = []

    reranker = MagicMock()
    reranker.rerank.return_value = []

    retriever = HybridRetriever(
        embedder=embedder,
        dense_store=dense,
        reranker=reranker,
        rerank_top_n=20,
    )

    # Patch FTS to return some chunks so reranker has something to work on
    from app.services.retrieval.base import RetrievedChunk
    c1 = RetrievedChunk(
        chunk_id="a", document_id="d1", text="hello", title="T", section_path="S",
        page_start=1, page_end=1, rrf_score=0.9,
    )
    c2 = RetrievedChunk(
        chunk_id="b", document_id="d1", text="world", title="T2", section_path="S2",
        page_start=2, page_end=2, rrf_score=0.8,
    )
    retriever._fts_search = lambda *a, **kw: ([
        {"chunk_id": "a", "document_id": "d1", "text": "hello", "title": "T",
         "section_path": "S", "page_start": 1, "page_end": 1, "score": -1.0},
        {"chunk_id": "b", "document_id": "d1", "text": "world", "title": "T2",
         "section_path": "S2", "page_start": 2, "page_end": 2, "score": -2.0},
    ], None)
    retriever._hydrate_from_db = lambda results: None

    try:
        results = retriever.search(
            notebook_id="nb1", query="q", top_k=5,
            options=SearchOptions(rerank_enabled=True, rerank_top_n=20),
        )
        # Reranker should have been called
        reranker.rerank.assert_called_once()
        assert len(results) == 0  # mock returns empty list
    finally:
        retriever._fts_search = retriever._fts_search


def test_retriever_no_reranker_does_not_call():
    """When no reranker is provided, search() should work normally."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384
    dense = MagicMock()
    dense.search.return_value = []

    retriever = HybridRetriever(embedder=embedder, dense_store=dense, reranker=None)
    # Just verify no crash when reranker is None
    result = retriever.search(notebook_id="nb1", query="q", top_k=5, document_ids=["d1"])
    assert isinstance(result, list)
