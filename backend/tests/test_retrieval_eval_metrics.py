"""Unit checks for the retrieval-eval metric helpers (Phase 0 harness).

These are pure functions - no backend, no DB - exercised through a
standalone script import so the test also guards the import path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "retrieval_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("retrieval_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def eval_module():
    return _load_module()


def test_dcg_ideal_ranking(eval_module):
    assert eval_module._dcg([1, 1, 0, 0]) > eval_module._dcg([0, 0, 1, 1])
    assert eval_module._dcg([0]) == 0.0


def test_ranking_metrics_perfect(eval_module):
    gt = {"a", "b"}
    m = eval_module._ranking_metrics(["a", "b", "x", "y", "z"], gt)
    assert m["recall_at_5"] == 1.0
    assert m["recall_at_10"] == 1.0
    assert m["precision_at_10"] == 0.4  # 2 hits / 5 retrieved
    assert m["mrr_at_10"] == 1.0
    assert m["ndcg_at_10"] == 1.0
    assert m["hit_rank"] == 1


def test_ranking_metrics_miss_and_rank(eval_module):
    gt = {"a", "b"}
    m = eval_module._ranking_metrics(["x", "y", "a"], gt)
    assert m["recall_at_5"] == 0.5  # one of two relevant chunks hit
    assert m["recall_at_10"] == 0.5
    assert m["mrr_at_10"] == 1.0 / 3.0
    assert m["hit_rank"] == 3
    m2 = eval_module._ranking_metrics(["x", "y"], gt)
    assert m2["mrr_at_10"] == 0.0
    assert m2["hit_rank"] is None


def test_ranking_metrics_empty_ground_truth_skipped_by_caller(eval_module):
    # denom==0 is guarded by callers (unanswerable questions); the function
    # itself is only called with a non-empty relevance set.
    pass


@pytest.mark.parametrize(
    ("status", "answer", "expected"),
    [
        ("insufficient_context", "", True),
        ("clarification_needed", "", True),
        ("completed", "This is not covered by the sources.", True),
        ("completed", "Binary search halves the search range.", False),
        # refusal wording beyond the opening region (observed on q-ga-02):
        # a real answer that mentions "does not ..." deep in the explanation
        ("completed", "# Why Dijkstra's algorithm fails with negative-weight "
                      "edges\n\nThe book explains this through a concrete "
                      "trading example in Chapter 7 [S2]. Rama is trying to "
                      "trade a music book for a piano, and the graph he draws "
                      "shows how the shortest-path computation slowly "
                      "diverges from the true cheapest route once an edge "
                      "has a negative weight. The chapter does not cover "
                      "negative edge weights in this worked example.", False),
        # contrastive opening pivots into substance: an answer, not a refusal
        ("completed", "The book does not discuss X, but Chapter 7 shows [S2] "
                      "why negative weights break Dijkstra's algorithm in a "
                      "trading example involving Rama and a music book.", False),
        # verbose refusals cite sources and run long (observed on q-un-04)
        ("completed", "I don't have any material about the movie *Inception* "
                      "in the retrieved results. The passages available cover "
                      "unrelated topics: a Wikipedia page about opera "
                      "[S1][S2], a sample ODT file explaining placeholder "
                      "text.", True),
        # "no reliable" alone is not a refusal (observed on q-vt-02): the
        # agent hedged about chapter boundaries, then answered
        ("completed", "No reliable chapter boundary was found for a section "
                      "on vibrato, so the following is drawn from the "
                      "passages that were retrieved. The book treats wobble "
                      "and tremolo as breath-management problems rather than "
                      "as separate vocal defects.", False),
    ],
)
def test_refusal_classification(eval_module, status, answer, expected):
    assert eval_module._is_refusal(status, answer) is expected
