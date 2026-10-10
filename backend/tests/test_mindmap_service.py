"""Tests for the mind map generator helpers."""

from __future__ import annotations

from app.services.generation.mindmap_service import (
    MindmapGenerator,
    _build_tree_nodes,
    _shorten_label,
    _shorten_summary,
    _slugify,
)


def test_slugify_normalizes():
    assert _slugify("Binary Search Trees") == "binary-search-trees"


def test_shorten_label_strips_citations():
    assert _shorten_label("Method [1]") == "Method"


def test_shorten_summary_truncates():
    assert len(_shorten_summary("x" * 200)) <= 120


def test_build_tree_nodes_dedupes_duplicate_labels():
    tree = _build_tree_nodes(
        {"label": "Method", "children": [{"label": "Method"}, {"label": "Method"}]},
        "doc1",
    )
    ids = [tree["id"]] + [c["id"] for c in tree["children"]]
    assert len(ids) == len(set(ids))


def test_build_tree_nodes_sets_source_document_ids():
    tree = _build_tree_nodes(
        {"label": "Root", "children": [{"label": "Child"}]}, "doc1"
    )
    assert tree["source_document_ids"] == ["doc1"]
    assert tree["children"][0]["source_document_ids"] == ["doc1"]


def test_build_tree_nodes_ignores_non_dict_children():
    """Abnormal children types from a weak model must not crash node building."""
    tree = _build_tree_nodes(
        {"label": "Root", "children": ["not a dict", {"label": "Real Child"}, 42]},
        "doc1",
    )
    assert tree is not None
    assert [c["label"] for c in tree["children"]] == ["Real Child"]


def test_tree_to_markdown_headings():
    tree = _build_tree_nodes(
        {
            "label": "Root",
            "children": [{"label": "Child A", "children": [{"label": "Grandchild"}]}],
        },
        "doc1",
    )
    md = MindmapGenerator._tree_to_markdown(tree)
    assert "# Root" in md
    assert "## Child A" in md
    assert "### Grandchild" in md
