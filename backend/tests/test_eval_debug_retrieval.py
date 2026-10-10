from app.routers.ask_router import _debug_retrieved_chunks, _debug_source_metadata
from app.services.qa import ask_debug
from app.services.qa.tools import ToolResults
from app.services.retrieval.base import RetrievedChunk


def test_debug_retrieved_chunks_include_structure_and_chunk_metadata(monkeypatch):
    monkeypatch.setattr(
        ask_debug,
        "get_chunk_by_id",
        lambda chunk_id: {
            "id": chunk_id,
            "text": "Full database text",
            "metadata_json": (
                '{"outline_source": "page_window", '
                '"chunking_strategy": "toc_based", '
                '"section_label": "Pages 20-27"}'
            ),
        },
    )

    tr = ToolResults()
    tr.metadata = [{
        "document_id": "d1",
        "filename": "book.pdf",
        "page_count": 100,
        "chunk_count": 12,
        "structure_status": "page_window_outline",
        "structure_confidence": "low",
    }]
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="Retrieved text",
            title="Pages 20-27",
            section_path="Pages 20-27",
            page_start=20,
            page_end=27,
            rrf_score=0.42,
        )
    ]
    tr.chunk_citations = [{"chunk_id": "c1", "ref_label": "S1"}]

    chunks = _debug_retrieved_chunks(tr)

    assert chunks[0]["passage_id"] == "S1"
    assert chunks[0]["section_label"] == "Pages 20-27"
    assert chunks[0]["outline_source"] == "page_window"
    assert chunks[0]["chunking_strategy"] == "toc_based"
    assert chunks[0]["structure_status"] == "page_window_outline"
    assert chunks[0]["score"] == 0.42


def test_debug_source_metadata_hydrates_from_chunk_document(monkeypatch):
    monkeypatch.setattr(
        ask_debug,
        "get_document",
        lambda document_id: {
            "id": document_id,
            "filename": "legacy.pdf",
            "page_count": 10,
            "chunk_count": 2,
        },
    )

    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="legacy-doc",
            text="Text",
        )
    ]

    metadata = _debug_source_metadata(tr)

    assert metadata[0]["document_id"] == "legacy-doc"
    assert metadata[0]["structure_status"] == "unknown_legacy"
