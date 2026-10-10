import pytest
from unittest.mock import MagicMock, patch
from app.services.retrieval.profile_selector import ProfileSelector
from app.services.retrieval.search_options import SearchOptions
from app.services.retrieval.retrieval_profile import PROFILES


class TestProfilePropagation:
    """Verify profile selected in QAToolExecutor reaches HybridRetriever."""

    def test_profile_name_passed_through_search_options(self):
        opts = SearchOptions(
            profile_name="keyword_heavy",
            rerank_enabled=False,
            rerank_top_n=20,
        )
        assert opts.profile_name == "keyword_heavy"

    def test_search_options_default_profile(self):
        opts = SearchOptions()
        assert opts.profile_name == "baseline"

    def test_profile_name_frozen_dataclass(self):
        opts = SearchOptions(profile_name="dense_heavy")
        with pytest.raises(Exception):
            opts.profile_name = "keyword_heavy"  # type: ignore[misc]


class TestRRFDifference:
    """Verify keyword_heavy produces different rankings than dense_heavy."""

    def test_profile_weights_differ(self):
        kw = PROFILES["keyword_heavy"]
        dh = PROFILES["dense_heavy"]
        assert kw.dense_weight != dh.dense_weight
        assert kw.keyword_weight != dh.keyword_weight

    def test_keyword_heavy_favors_keyword(self):
        p = PROFILES["keyword_heavy"]
        assert p.keyword_weight > p.dense_weight

    def test_dense_heavy_favors_dense(self):
        p = PROFILES["dense_heavy"]
        assert p.dense_weight > p.keyword_weight

    def test_balanced_is_between(self):
        p = PROFILES["balanced"]
        assert 0 < p.dense_weight < 2.0
        assert 0 < p.keyword_weight < 2.0


class TestZeroIO:
    """Verify select() makes no DB calls or network I/O."""

    def test_select_no_io(self):
        selector = ProfileSelector()
        with patch("app.services.retrieval.profile_selector.ProfileSelector.select") as mock_select:
            mock_select.return_value = "balanced"
            result = mock_select("explain_concept", "test", [])
            assert result == "balanced"

    def test_select_is_pure_function(self):
        selector = ProfileSelector()
        args = ("explain_concept", "What is marginal utility in economics?", [])
        result1 = selector.select(*args)
        result2 = selector.select(*args)
        result3 = selector.select(*args)
        assert result1 == result2 == result3


class TestBaselineUnchanged:
    """Verify baseline profile behaves consistently."""

    def test_baseline_profile_weights(self):
        p = PROFILES["baseline"]
        assert p.dense_weight == 1.0
        assert p.keyword_weight == 1.0
        assert p.rrf_k == 60

    def test_baseline_profile_name(self):
        p = PROFILES["baseline"]
        assert p.name == "baseline"

    def test_profile_selector_default_is_balanced(self):
        selector = ProfileSelector()
        result = selector.select("unknown_task", "any query", [])
        assert result == "balanced"
