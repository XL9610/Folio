"""Opt-in web search provider abstraction for agentic QA.

Web results are transient evidence for a single answer. They are not written
to the notebook index or mixed into source-document retrieval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str
    site_name: str = ""
    published_date: str = ""
    retrieved_at: str = ""
    score: float = 0.0


class WebSearchProvider:
    name = "base"
    enabled = False

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        raise NotImplementedError


class DisabledWebSearchProvider(WebSearchProvider):
    name = "disabled"
    enabled = False

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        return []


class TavilyWebSearchProvider(WebSearchProvider):
    name = "tavily"

    def __init__(self, api_key: str, timeout_seconds: float = 12.0):
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self.enabled = bool(api_key)

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        if not self.enabled:
            return []

        max_results = max(1, min(int(max_results or 5), 8))
        retrieved_at = datetime.now(timezone.utc).isoformat()

        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": False,
                    "include_raw_content": True,
                    "max_results": max_results,
                },
            )
            response.raise_for_status()
            data = response.json()

        results: list[WebSearchResult] = []
        for item in (data.get("results") or [])[:max_results]:
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url or "Untitled web result").strip()
            content = str(item.get("content") or item.get("snippet") or "").strip()
            if not url or not content:
                continue
            results.append(WebSearchResult(
                title=title,
                url=url,
                content=content,
                site_name=_site_name(url),
                published_date=str(item.get("published_date") or ""),
                retrieved_at=retrieved_at,
                score=float(item.get("score") or 0.0),
            ))
        return results


class CustomWebSearchProvider(WebSearchProvider):
    """Generic JSON search endpoint (Tavily-compatible response shape).

    GETs ``{base_url}?q={query}&count={max_results}`` with an optional
    ``Authorization: Bearer {api_key}`` header, and expects a JSON body with a
    ``results`` array of ``{title, url, content}`` items. Lets users plug in a
    self-hosted or third-party search API without a Tavily dependency.
    """

    name = "custom"

    def __init__(self, base_url: str, api_key: str = "", timeout_seconds: float = 12.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self.enabled = bool(base_url)

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        if not self.enabled:
            return []

        max_results = max(1, min(int(max_results or 5), 8))
        retrieved_at = datetime.now(timezone.utc).isoformat()
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.get(
                self._base_url,
                params={"q": query, "count": max_results},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        results: list[WebSearchResult] = []
        for item in (data.get("results") or [])[:max_results]:
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url or "Untitled web result").strip()
            content = str(item.get("content") or item.get("snippet") or "").strip()
            if not url or not content:
                continue
            results.append(WebSearchResult(
                title=title,
                url=url,
                content=content,
                site_name=_site_name(url),
                published_date=str(item.get("published_date") or ""),
                retrieved_at=retrieved_at,
                score=float(item.get("score") or 0.0),
            ))
        return results


def get_web_search_provider() -> WebSearchProvider:
    provider = os.getenv("WEB_SEARCH_PROVIDER", "disabled").strip().lower()
    api_key = os.getenv("WEB_SEARCH_API_KEY", "") or os.getenv("TAVILY_API_KEY", "")
    timeout_seconds = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "12"))
    if provider == "tavily":
        return TavilyWebSearchProvider(api_key=api_key, timeout_seconds=timeout_seconds)
    if provider == "custom":
        return CustomWebSearchProvider(
            base_url=os.getenv("WEB_SEARCH_BASE_URL", ""),
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    return DisabledWebSearchProvider()


def fetch_page_content(url: str, timeout_seconds: float = 10.0) -> str:
    """Fetch a URL and extract its main content as clean text.

    Uses trafilatura for boilerplate removal. Returns empty string on any error.
    """
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(
            url,
            decode=True,
            no_ssl_verify=False,
            timeout=timeout_seconds,
        )
        if not downloaded:
            return ""

        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
            output_format="txt",
        )
        return (content or "").strip()
    except Exception:
        return ""


def _site_name(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


