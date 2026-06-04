"""Keyless web-search tool — the 'external verification' half of resource checking.

The planner reasons about WHAT a learner should study but is told never to invent
URLs (see PLANNER_SYSTEM). This module supplies the missing tool: it fetches REAL
candidate links from DuckDuckGo's HTML endpoint so a verifier agent can confirm a
suggested resource actually exists instead of trusting the model's memory.

No API key, no extra dependency — a single HTML GET parsed with a small regex.
Cleanly separated so it could later be swapped for an MCP search server.
"""
from __future__ import annotations

import asyncio
import html
import re
import urllib.parse

import httpx

from praxis.learning.models import SearchHit


_DDG_URL = "https://html.duckduckgo.com/html/"
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snip>.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    """Strip HTML tags and unescape entities from a fragment."""
    return html.unescape(_TAG_RE.sub("", s)).strip()


def _unwrap(href: str) -> str:
    """DuckDuckGo wraps results as /l/?uddg=<url-encoded-target>. Unwrap to the real URL."""
    if "uddg=" in href:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if params.get("uddg"):
            return params["uddg"][0]
    if href.startswith("//"):
        return "https:" + href
    return href


async def web_search(query: str, max_results: int = 5) -> list[SearchHit]:
    """Return up to `max_results` real search hits for `query`.

    Raises RuntimeError on a transport/HTTP failure so callers can surface it; an
    empty list (no parseable results) is returned normally, not as an error.
    """
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            r = await c.post(
                _DDG_URL,
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Praxis-learning-agent/1.0)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            r.raise_for_status()
            body = r.text
    except Exception as e:
        raise RuntimeError(f"web search failed for {query!r}: {e}") from e

    titles = list(_RESULT_RE.finditer(body))
    snippets = list(_SNIPPET_RE.finditer(body))
    hits: list[SearchHit] = []
    for i, m in enumerate(titles[:max_results]):
        url = _unwrap(m.group("href"))
        title = _clean(m.group("title"))
        snippet = _clean(snippets[i].group("snip")) if i < len(snippets) else ""
        if url and title:
            hits.append(SearchHit(title=title, url=url, snippet=snippet))
    return hits


async def search_many(queries: list[str], per_query: int = 4) -> dict[str, list[SearchHit]]:
    """Run several searches concurrently. Returns {query: hits}; failed queries map to []."""
    async def one(q: str) -> tuple[str, list[SearchHit]]:
        try:
            return q, await web_search(q, max_results=per_query)
        except Exception:
            return q, []

    results = await asyncio.gather(*(one(q) for q in queries))
    return dict(results)
