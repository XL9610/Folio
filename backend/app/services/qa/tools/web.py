"""web_search tool."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...web_search import fetch_page_content
from ..agent_planner import ToolCall
from .types import ToolResults

logger = logging.getLogger(__name__)


class WebTool:
    """web_search tool."""

    def _tool_web_search(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        query = (tc.args.get("query", "") or tc.args.get("search_query", "")).strip()
        max_results = max(1, min(int(tc.args.get("max_results", 5) or 5), 8))

        if not query:
            results.trace.append({
                "name": "web_search",
                "status": "error",
                "args": tc.args,
                "error": "No search query provided",
            })
            return

        if not getattr(self._web_search, "enabled", False):
            results.trace.append({
                "name": "web_search",
                "status": "disabled",
                "args": {
                    "query": query,
                    "max_results": max_results,
                    "provider": getattr(self._web_search, "name", "disabled"),
                },
                "error": "Internet research is enabled for this ask, but no web search provider is configured.",
            })
            return

        web_items = self._web_search.search(query=query, max_results=max_results)

        # Fetch full page content for top 2 results (in parallel).
        fetch_urls = [item.url for item in web_items[:2] if item.url]
        fetched: dict[str, str] = {}
        if fetch_urls:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(fetch_page_content, url, 10.0): url
                    for url in fetch_urls
                }
                for future in as_completed(futures, timeout=22.0):
                    url = futures[future]
                    try:
                        text = future.result()
                        if text:
                            fetched[url] = text
                    except Exception:
                        pass

        for item in web_items:
            ref_label = self._next_ref_label(results, "web")
            content = item.content[:1800].strip()
            snippet = content[:160].replace("\n", " ").strip()
            if len(content) > 160:
                snippet += "\u2026"
            result = {
                "source_type": "web",
                "ref_label": ref_label,
                "title": item.title,
                "url": item.url,
                "site_name": item.site_name,
                "published_date": item.published_date,
                "retrieved_at": item.retrieved_at,
                "score": item.score,
                "content": content,
                "fetched_content": fetched.get(item.url, ""),
                "text_snippet": snippet,
            }
            results.web_results.append(result)
            results.chunk_citations.append({
                "source_type": "web",
                "ref_label": ref_label,
                "chunk_id": f"web:{ref_label}",
                "document_id": "",
                "text_snippet": snippet,
                "section_path": item.site_name,
                "page_start": -1,
                "page_end": -1,
                "url": item.url,
                "title": item.title,
                "site_name": item.site_name,
                "published_date": item.published_date,
                "retrieved_at": item.retrieved_at,
            })

        results.trace.append({
            "name": "web_search",
            "status": "ok" if web_items else "no_match",
            "args": {
                "query": query,
                "max_results": max_results,
                "results": len(web_items),
                "provider": getattr(self._web_search, "name", "unknown"),
            },
        })
