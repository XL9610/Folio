"""Mind Map generator — LLM produces JSON concept tree with markdown fallback."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Any

from .scope_resolver import ScopeResolver
from ..retrieval.base import RetrievedChunk
from ..qa.types import make_book_citation
from ._title_utils import clean_filename
from .sampling import sample_body_chunks, is_body_chunk

logger = logging.getLogger(__name__)


def _unresolved_scope_message(meta: dict) -> str:
    if meta.get("scope_resolved", True) is not False:
        return ""
    return (
        meta.get("unresolved_reason")
        or "The requested chapter or section could not be matched. Use a page range or topic keyword instead."
    )

MINDMAP_MARKDOWN_PROMPT = """You are creating a mind map from textbook material. Output markdown with a hierarchical heading/bullet structure that a mind map renderer can display.

Rules:
1. The top-level # heading is the central topic (e.g. the chapter title).
2. ## subheadings become first-level branches (key concepts / themes).
3. ### sub-subheadings become second-level branches (details / examples).
4. Use - bullet lists under each heading for leaf-level facts.
5. Use ONLY the provided passages — do not invent.
6. Aim for 4-8 first-level branches, each with 2-5 sub-items.
7. Cite passages with [1][2] format at the end of lines when relevant.

Output ONLY the markdown — no preamble, no code fences."""

MINDMAP_JSON_PROMPT = """You are creating an interactive concept map from a textbook source.

Return strict JSON only — no markdown, no code fences, no preamble.

Each node must be short, clickable, and understandable by itself.
Labels must be standalone noun phrases, 2-8 words, max 64 characters.
Do not use full sentences, clauses, dangling fragments, or pronouns as node labels.
Bad labels: "It can improve outcomes", "This chapter shows how", "The method is a form of".
Good labels: "Outcome Improvements", "Core Mechanism", "Method Family".
Use only provided passages.
Every child node must include source_document_ids (array with the doc reference number like ["[1]"]).
Every node should include summary: one concise sentence, max 120 characters, explaining what the card means.

Prefer conceptual labels over table-of-contents labels.
Do not use labels such as "Part One", "Part Two", "Chapter 1", "Unit II", "Introduction", or page ranges unless the passage itself gives no better concept.
If a passage uses a structural label, convert it into the concept, period, argument, method, or theme that the section is actually about.
Each parent label should preview what its children will contain.

Create a rich exploratory concept map, not a table-of-contents outline or a list of search hits.
Max 6 children per node. Max depth 3 levels below root.

Density requirements:
- Root should usually have 4-7 first-level conceptual branches.
- Important first-level branches should have 2-4 children.
- Do not return only 1-2 branches unless the provided passages are extremely sparse.
- Merge scattered passages into useful categories such as people, works, periods, methods, contexts, institutions, conflicts, consequences, or examples when supported by evidence.

Example output:
{
  "label": "The Intelligent Investor",
  "children": [
    {
      "label": "Definition of Investment",
      "children": [
        {"label": "Thorough Analysis", "source_document_ids": ["[1]"]},
        {"label": "Safety of Principal", "summary": "Investment requires protecting principal, not just chasing return.", "source_document_ids": ["[1]"]}
      ]
    }
  ]
}

Output ONLY the JSON object."""

_MAX_DEPTH = 3
_MAX_CHILDREN = 6
_MAX_LABEL_CHARS = 64


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:48]


def _shorten_label(label: str) -> str:
    label = re.sub(r"\[[^\]]+\]", "", str(label)).strip()
    if len(label) <= _MAX_LABEL_CHARS:
        shortened = label
    else:
        truncated = label[:_MAX_LABEL_CHARS].rstrip()
        last_space = truncated.rfind(" ")
        if last_space > _MAX_LABEL_CHARS // 2:
            shortened = truncated[:last_space]
        else:
            shortened = truncated
    dangling = re.compile(
        r"(?:\b(?:a|an|the|and|or|of|to|for|from|with|without|by|in|on|into|that|which|as|is|are|can|could|would|should)\s*)$",
        re.IGNORECASE,
    )
    while dangling.search(shortened) and len(shortened.split()) > 2:
        shortened = dangling.sub("", shortened).strip()
    return shortened or "Untitled"


def _shorten_summary(summary: str) -> str:
    s = re.sub(r"\s+", " ", str(summary or "")).strip()
    if not s:
        return ""
    if len(s) <= 120:
        return s
    cut = s[:120].rstrip()
    last = cut.rfind(" ")
    return (cut[:last] if last > 72 else cut).rstrip(" ,;:")


def _build_tree_nodes(
    raw_node: dict,
    source_doc_id: str,
    parent_path: list[str] | None = None,
    depth: int = 0,
    seen: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(raw_node, dict):
        return None
    if depth > _MAX_DEPTH:
        return None

    label = _shorten_label(str(raw_node.get("label", "?")))
    if not label or label == "?":
        return None

    parent_path = parent_path or []
    seen = seen if seen is not None else {}

    # Ensure unique ids even when two sibling nodes share a short label.
    base_id = _slugify(label) or "node"
    if base_id in seen:
        seen[base_id] += 1
        node_id = f"{base_id}-{seen[base_id]}"
    else:
        seen[base_id] = 1
        node_id = base_id

    node: dict[str, Any] = {
        "id": node_id,
        "label": label,
        "summary": _shorten_summary(
            raw_node.get("summary", "") or raw_node.get("description", "")
        ),
        "prompt_term": _shorten_label(str(raw_node.get("prompt_term", label))),
        "context_path": list(parent_path),
        "source_document_ids": [source_doc_id],
        "children": [],
    }

    raw_children = raw_node.get("children", [])
    if isinstance(raw_children, list) and depth < _MAX_DEPTH:
        child_path = parent_path + [label]
        for child in raw_children[:_MAX_CHILDREN]:
            child_node = _build_tree_nodes(
                child, source_doc_id, child_path, depth + 1, seen
            )
            if child_node:
                node["children"].append(child_node)

    return node


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_m:
        try:
            return json.loads(fence_m.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_m = re.search(r"\{[\s\S]*\}", text)
    if brace_m:
        try:
            return json.loads(brace_m.group(0))
        except json.JSONDecodeError:
            pass

    return None


class MindmapGenerator:
    """Generate an interactive concept map from a single document."""

    def __init__(self, llm_client, scope_resolver: ScopeResolver):
        self._llm = llm_client
        self._resolver = scope_resolver

    def generate(
        self,
        notebook_id: str,
        document_id: str,
        scope: str,
        doc_filename: str = "",
    ) -> dict[str, Any]:
        scope = (scope or "").strip()

        result = self._resolver.resolve(document_id, notebook_id, scope)
        chunks = result.chunks
        meta = result.metadata

        if not chunks:
            message = _unresolved_scope_message(meta)
            return {
                "title": "Mind Map",
                "markdown": message or "No relevant content was found in the selected scope.",
                "tree": None,
                "mode": "single_source_map",
                "sources": [],
                "stats": {
                    "chunk_count": 0,
                    "scope_layer": meta.get("layer", "unknown"),
                    "scope": scope or "",
                    "scope_resolved": meta.get("scope_resolved", True),
                    "unresolved_reason": message or meta.get("unresolved_reason", ""),
                },
            }

        body_chunks = [c for c in chunks if is_body_chunk(c)]
        if not body_chunks:
            body_chunks = chunks

        source_chunks = sample_body_chunks(body_chunks, max_chunks=28)

        ref_label = "[1]"
        chunk_refs = [(c, ref_label) for c in source_chunks]

        user_prompt = self._build_user_prompt(chunk_refs, scope, doc_filename)

        try:
            response = self._llm.complete_chat(
                messages=[
                    {"role": "system", "content": MINDMAP_JSON_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=3200,
            )
        except Exception as exc:
            logger.error("Mind map generation failed: %s", exc)
            raise

        parsed = _try_parse_json(response)
        if parsed and isinstance(parsed, dict):
            tree = _build_tree_nodes(parsed, document_id)
            if tree and self._is_sparse_tree(tree) and len(source_chunks) >= 8:
                retry_prompt = user_prompt + (
                    "\n\nQuality retry: the previous map was too sparse. "
                    "Return a richer concept map with 4-7 first-level branches and "
                    "2-4 children for important branches. Keep labels concise noun phrases."
                )
                try:
                    retry_response = self._llm.complete_chat(
                        messages=[
                            {"role": "system", "content": MINDMAP_JSON_PROMPT},
                            {"role": "user", "content": retry_prompt},
                        ],
                        max_tokens=3600,
                    )
                    retry_parsed = _try_parse_json(retry_response)
                    if retry_parsed and isinstance(retry_parsed, dict):
                        retry_tree = _build_tree_nodes(retry_parsed, document_id)
                        if retry_tree and not self._is_sparser_than(retry_tree, tree):
                            tree = retry_tree
                except Exception as exc:
                    logger.warning("Sparse retry failed: %s", exc)

            if tree:
                tree["source_document_ids"] = [document_id]
                tree["id"] = "root"
            markdown = self._tree_to_markdown(tree) if tree else response
            title = tree.get("label", parsed.get("label", "Mind Map")) if tree else parsed.get("label", "Mind Map")
        else:
            logger.warning("Mindmap JSON parse failed, using raw markdown fallback.")
            try:
                markdown = self._llm.complete_chat(
                    messages=[
                        {"role": "system", "content": MINDMAP_MARKDOWN_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2048,
                )
            except Exception:
                markdown = response
            tree = None
            first_line = markdown.strip().split("\n")[0] if markdown else ""
            title = first_line.lstrip("#").strip() or "Mind Map"

        return {
            "title": title,
            "markdown": markdown,
            "tree": tree,
            "mode": "single_source_map",
            "sources": [
                asdict(make_book_citation(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id or document_id,
                    section_path=c.section_path or c.title or "",
                    page_start=c.page_start,
                    page_end=c.page_end,
                    title=doc_filename,
                    ref_label=f"[{i}]",
                    text_snippet=(c.text or "")[:120],
                ))
                for i, c in enumerate(source_chunks, 1)
            ],
            "stats": {
                "chunk_count": len(chunks),
                "sampled_count": len(source_chunks),
                "scope_layer": meta.get("layer", "unknown"),
                "scope": scope or "",
            },
        }

    def _is_sparse_tree(self, tree: dict[str, Any]) -> bool:
        children = tree.get("children") or []
        if len(children) < 4:
            return True
        rich_children = [
            child for child in children
            if len(child.get("children") or []) >= 2
        ]
        return len(rich_children) < 2

    def _is_sparser_than(self, candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        def score(node: dict[str, Any]) -> tuple[int, int]:
            children = node.get("children") or []
            grand = sum(len(child.get("children") or []) for child in children)
            return (len(children), grand)
        return score(candidate) < score(current)

    def _build_user_prompt(
        self,
        chunk_refs: list[tuple[RetrievedChunk, str]],
        scope: str,
        doc_filename: str,
    ) -> str:
        lines: list[str] = []
        lines.append("Mode: single_source_map")
        scope_line = f"Scope: {scope}" if scope else "Scope: whole document"
        lines.append(scope_line)
        lines.append(
            "Instruction: create a rich exploratory concept map, not a table-of-contents "
            "outline or a list of search hits. Use 4-7 first-level branches when the evidence allows it."
        )
        lines.append(f"Document: {clean_filename(doc_filename) if doc_filename else '(untitled)'}")
        lines.append("")
        lines.append("Source passages:")
        lines.append("")

        for c, ref in chunk_refs:
            loc = f"p.{c.page_start}" if c.page_start > 0 else ""
            sec = c.section_path or ""
            header = f"{ref} {sec}"
            if loc:
                header += f" ({loc})"
            lines.append(header)
            lines.append(c.text[:2000])
            lines.append("---")

        return "\n".join(lines)

    @staticmethod
    def _tree_to_markdown(tree: dict) -> str:
        lines: list[str] = []

        def walk(node: dict, level: int):
            prefix = "#" * min(level + 1, 6)
            lines.append(f"{prefix} {node.get('label', '?')}")
            for child in node.get("children", []):
                walk(child, level + 1)

        if tree:
            walk(tree, 0)
        return "\n".join(lines)
