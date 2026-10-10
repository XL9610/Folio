"""Unit tests for reranker — Module C front matter boost."""

from __future__ import annotations

from app.services.retrieval.base import RetrievedChunk
from app.services.retrieval.reranker import (
    get_downrank_penalty,
    get_front_matter_boost,
    rerank_results,
)
from app.services.retrieval.boost_config import BoostConfig


def _make_chunk(chunk_id: str, section_path: str, rrf_score: float, page_start: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        text="test text",
        title="",
        section_path=section_path,
        page_start=page_start,
        page_end=page_start + 1,
        rrf_score=rrf_score,
    )


class TestFrontMatterPenalty:
    def test_front_matter_content_no_penalty(self):
        assert get_downrank_penalty("Preface") == 1.0

    def test_front_matter_content_foreword_no_penalty(self):
        assert get_downrank_penalty("Foreword") == 1.0

    def test_front_matter_meta_still_penalized(self):
        assert get_downrank_penalty("Acknowledgments") == 0.3

    def test_copyright_meta_still_penalized(self):
        assert get_downrank_penalty("Copyright") == 0.3

    def test_exercises_always_penalized(self):
        assert get_downrank_penalty("Exercises 3.1") == 0.1

    def test_empty_section_path_neutral(self):
        assert get_downrank_penalty("") == 1.0


class TestFrontMatterBoost:
    def test_content_boost_on(self):
        bc = BoostConfig(front_matter_boost=True)
        assert get_front_matter_boost("Preface", bc) == 1.5

    def test_content_boost_off(self):
        bc = BoostConfig(front_matter_boost=False)
        assert get_front_matter_boost("Preface", bc) == 1.0

    def test_meta_boost_on_no_effect(self):
        bc = BoostConfig(front_matter_boost=True)
        assert get_front_matter_boost("Acknowledgments", bc) == 1.0

    def test_meta_boost_off_no_effect(self):
        bc = BoostConfig(front_matter_boost=False)
        assert get_front_matter_boost("Acknowledgments", bc) == 1.0

    def test_intro_content_boost_on(self):
        bc = BoostConfig(front_matter_boost=True)
        assert get_front_matter_boost("Introduction to Economics", bc) == 1.5

    def test_empty_section_path_boost(self):
        bc = BoostConfig(front_matter_boost=True)
        assert get_front_matter_boost("", bc) == 1.0


class TestCompositeScoreOrdering:
    def test_front_matter_boosted_ranks_higher(self):
        bc = BoostConfig(front_matter_boost=True)
        chunks = [
            _make_chunk("c1", "Preface", 1.0),
            _make_chunk("c2", "Chapter 1: Basics", 1.0),
            _make_chunk("c3", "Acknowledgments", 1.0),
        ]
        result = rerank_results(chunks, downrank_roles=[], boost_terms=[], boost_config=bc)
        ids_in_order = [c.chunk_id for c in result]
        assert ids_in_order[0] == "c1"

    def test_boost_off_no_front_matter_boost(self):
        bc = BoostConfig(front_matter_boost=False)
        chunks = [
            _make_chunk("c1", "Preface", 1.0),
            _make_chunk("c2", "Chapter 1: Basics", 1.0),
        ]
        result = rerank_results(chunks, downrank_roles=[], boost_terms=[], boost_config=bc)
        ids_in_order = [c.chunk_id for c in result]
        assert ids_in_order[0] == "c1" or ids_in_order[0] == "c2"

    def test_meta_front_matter_not_boosted(self):
        bc = BoostConfig(front_matter_boost=True)
        chunks = [
            _make_chunk("c1", "Acknowledgments", 1.0),
            _make_chunk("c2", "Chapter 1: Basics", 1.0),
        ]
        result = rerank_results(chunks, boost_terms=[], boost_config=bc)
        ids_in_order = [c.chunk_id for c in result]
        assert ids_in_order[0] == "c2"

    def test_exercises_stay_at_bottom(self):
        bc = BoostConfig()
        chunks = [
            _make_chunk("c1", "Exercises 5.1", 1.0),
            _make_chunk("c2", "Chapter 5: Content", 1.0),
        ]
        result = rerank_results(chunks, boost_config=bc)
        ids_in_order = [c.chunk_id for c in result]
        assert ids_in_order[0] == "c2"

    def test_rerank_defaults_work_with_boost_config(self):
        bc = BoostConfig(front_matter_boost=True)
        chunks = [
            _make_chunk("c1", "Preface", 0.8),
            _make_chunk("c2", "Chapter 1", 0.9),
        ]
        result = rerank_results(chunks, boost_config=bc)
        assert len(result) == 2
        assert result[0].chunk_id == "c1"
        assert result[0].rrf_score > 1.0


class TestLeadParagraphBoost:
    def test_lead_boost_within_threshold(self):
        from app.services.retrieval.reranker import get_lead_paragraph_boost
        bc = BoostConfig(lead_paragraph_boost=True, lead_page_threshold=10)
        assert get_lead_paragraph_boost(5, bc) == 1.2

    def test_lead_boost_at_threshold(self):
        from app.services.retrieval.reranker import get_lead_paragraph_boost
        bc = BoostConfig(lead_paragraph_boost=True, lead_page_threshold=10)
        assert get_lead_paragraph_boost(10, bc) == 1.2

    def test_lead_boost_beyond_threshold(self):
        from app.services.retrieval.reranker import get_lead_paragraph_boost
        bc = BoostConfig(lead_paragraph_boost=True, lead_page_threshold=10)
        assert get_lead_paragraph_boost(11, bc) == 1.0

    def test_lead_boost_flag_off(self):
        from app.services.retrieval.reranker import get_lead_paragraph_boost
        bc = BoostConfig(lead_paragraph_boost=False, lead_page_threshold=10)
        assert get_lead_paragraph_boost(5, bc) == 1.0

    def test_lead_boost_page_negative_one_guard(self):
        from app.services.retrieval.reranker import get_lead_paragraph_boost
        bc = BoostConfig(lead_paragraph_boost=True, lead_page_threshold=10)
        assert get_lead_paragraph_boost(-1, bc) == 1.0

    def test_lead_boost_threshold_zero(self):
        from app.services.retrieval.reranker import get_lead_paragraph_boost
        bc = BoostConfig(lead_paragraph_boost=True, lead_page_threshold=0)
        assert get_lead_paragraph_boost(5, bc) == 1.0

    def test_composite_lead_and_front_matter_boost(self):
        bc = BoostConfig(front_matter_boost=True, lead_paragraph_boost=True, lead_page_threshold=10)
        chunks = [
            _make_chunk("c1", "Preface", 1.0, page_start=3),
            _make_chunk("c2", "Chapter 1", 1.2, page_start=50),
        ]
        result = rerank_results(chunks, downrank_roles=[], boost_terms=[], boost_config=bc)
        assert result[0].chunk_id == "c1"
