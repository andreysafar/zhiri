"""News fetcher via RSS feeds from Russian news sources."""
import asyncio
import feedparser
import httpx
from datetime import datetime


RSS_FEEDS = {
    "РИА Новости": "https://ria.ru/export/rss2/archive/index.xml",
    "ТАСС": "https://tass.ru/rss/v2.xml",
    "Lenta.ru": "https://lenta.ru/rss/news",
    "RT": "https://russian.rt.com/rss",
}

# Simple in-memory cache
_cache: dict[str, tuple[datetime, list[dict]]] = {}
CACHE_TTL_MINUTES = 15


def _is_cache_valid(source: str) -> bool:
    if source not in _cache:
        return False
    cached_at, _ = _cache[source]
    age = (datetime.now() - cached_at).total_seconds() / 60
    return age < CACHE_TTL_MINUTES


async def fetch_feed(source: str, url: str, client: httpx.AsyncClient) -> list[dict]:
    if _is_cache_valid(source):
        return _cache[source][1]

    try:
        response = await client.get(url, timeout=10.0, follow_redirects=True)
        feed = feedparser.parse(response.text)
        items = []
        for entry in feed.entries[:10]:
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", ""))[:300],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        _cache[source] = (datetime.now(), items)
        return items
    except Exception:
        return _cache.get(source, (None, []))[1]


async def get_top_news(limit: int = 10) -> str:
    """Fetch top news from all sources and return formatted string."""
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        tasks = [fetch_feed(src, url, client) for src, url in RSS_FEEDS.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items = []
    for source, result in zip(RSS_FEEDS.keys(), results):
        if isinstance(result, list):
            for item in result[:3]:
                all_items.append(f"[{source}] {item['title']}")

    if not all_items:
        return "Не удалось загрузить новости."

    news_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(all_items[:limit]))
    return news_text


async def get_news_for_context(limit: int = 5) -> str:
    """Return short news digest to inject into LLM context."""
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        tasks = [fetch_feed(src, url, client) for src, url in RSS_FEEDS.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    headlines = []
    for source, result in zip(RSS_FEEDS.keys(), results):
        if isinstance(result, list) and result:
            item = result[0]
            headlines.append(f"- {item['title']}")

    if not headlines:
        return ""

    digest = "СВЕЖИЕ НОВОСТИ НА СЕГОДНЯ:\n" + "\n".join(headlines[:limit])
    return digest
