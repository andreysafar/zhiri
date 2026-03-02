"""
Scraper for Zhirinovsky's speeches from the State Duma transcripts.
Usage: python scrape_duma.py [--limit N]

Downloads speeches to speeches/ directory as JSON files.
Sources:
  - api.duma.gov.ru — official Duma API (transcripts)
  - ldpr.ru — LDPR party archive
"""
import asyncio
import argparse
import json
import re
import sys
from pathlib import Path

import httpx

SPEECHES_DIR = Path(__file__).parent / "speeches"

# Duma Open Data API
DUMA_TRANSCRIPT_SEARCH = (
    "http://api.duma.gov.ru/api/{token}/search.json"
    "?q=%D0%96%D0%B8%D1%80%D0%B8%D0%BD%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9"
    "&app_token={token}&limit={limit}&offset={offset}&sort=date"
)

# Fallback: LDPR news/speeches archive RSS
LDPR_RSS = "https://ldpr.ru/rss.xml"


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def fetch_duma_speeches(token: str, limit: int, client: httpx.AsyncClient) -> list[dict]:
    speeches = []
    offset = 0
    per_page = min(limit, 20)

    while len(speeches) < limit:
        url = DUMA_TRANSCRIPT_SEARCH.format(token=token, limit=per_page, offset=offset)
        try:
            r = await client.get(url, timeout=15)
            data = r.json()
            items = data.get("results", {}).get("items", [])
            if not items:
                break
            for item in items:
                text = clean_html(item.get("text", item.get("summary", "")))
                if "Жириновский" not in text and "Жириновский" not in item.get("name", ""):
                    continue
                speeches.append({
                    "year": int(item.get("date", "0000")[:4]) if item.get("date") else 0,
                    "title": item.get("name", "Выступление в Госдуме"),
                    "text": text[:2000],
                    "source": "duma.gov.ru",
                    "url": item.get("url", ""),
                })
            offset += per_page
        except Exception as e:
            print(f"Duma API error: {e}", file=sys.stderr)
            break

    return speeches


async def fetch_ldpr_speeches(limit: int, client: httpx.AsyncClient) -> list[dict]:
    import feedparser  # optional dependency
    speeches = []
    try:
        r = await client.get(LDPR_RSS, timeout=15)
        feed = feedparser.parse(r.text)
        for entry in feed.entries[:limit]:
            text = clean_html(entry.get("summary", entry.get("description", "")))
            if len(text) < 50:
                continue
            speeches.append({
                "year": int(entry.get("published", "")[:4]) if entry.get("published") else 0,
                "title": entry.get("title", "Выступление"),
                "text": text[:2000],
                "source": "ldpr.ru",
                "url": entry.get("link", ""),
            })
    except Exception as e:
        print(f"LDPR RSS error: {e}", file=sys.stderr)
    return speeches


def save_speeches(speeches: list[dict], filename: str = "duma_speeches.json") -> None:
    SPEECHES_DIR.mkdir(exist_ok=True)
    out = SPEECHES_DIR / filename
    out.write_text(json.dumps(speeches, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(speeches)} speeches to {out}")


async def main(duma_token: str | None, limit: int) -> None:
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        speeches = []

        if duma_token:
            print("Fetching from Duma API...")
            duma = await fetch_duma_speeches(duma_token, limit, client)
            print(f"  Got {len(duma)} speeches from Duma")
            speeches.extend(duma)

        print("Fetching from LDPR RSS...")
        ldpr = await fetch_ldpr_speeches(limit, client)
        print(f"  Got {len(ldpr)} speeches from LDPR")
        speeches.extend(ldpr)

    if speeches:
        save_speeches(speeches)
    else:
        print("No speeches fetched. Check API token or network connection.")
        print()
        print("To use Duma API register at: https://api.duma.gov.ru/")
        print("Then run: python scrape_duma.py --token YOUR_TOKEN --limit 200")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Zhirinovsky speeches")
    parser.add_argument("--token", default=None, help="Duma API token (optional)")
    parser.add_argument("--limit", type=int, default=50, help="Max speeches to fetch")
    args = parser.parse_args()

    asyncio.run(main(args.token, args.limit))
