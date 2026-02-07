"""
Fetch RSS/Atom feeds, extract article text, and normalize into rows for DB insertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import feedparser
import httpx
import pandas as pd
import trafilatura

from src.config.settings import settings

@dataclass
class FeedConfig:
    url: str
    source: str


DEFAULT_FEEDS: List[FeedConfig] = [
    FeedConfig(
        url="https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medical-products-feed",
        source="FDA Medical Products",
    ),
]


def _parse_published(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = parsedate_to_datetime(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _hash_event_id(url: str, title: str, published_at: Optional[datetime]) -> str:
    base = f"{url}|{title}|{published_at.isoformat() if published_at else ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _fetch_article_text(url: str, timeout_sec: int = 20) -> Optional[str]:
    try:
        with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return None

    # Try trafilatura extraction
    try:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return extracted
    except Exception:
        pass
    return None


def _extract_pubmed_id(url: str) -> Optional[str]:
    # PubMed URLs look like https://pubmed.ncbi.nlm.nih.gov/<pmid>/
    try:
        parts = urlparse(url).path.strip("/").split("/")
        if parts and parts[0].isdigit():
            return parts[0]
    except Exception:
        return None
    return None


def _fetch_pubmed_abstract(pmid: str, timeout_sec: int = 20) -> Optional[str]:
    """
    Fetch PubMed abstract via NCBI E-utilities.
    """
    try:
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        if settings.ENTREZ_EMAIL:
            params["email"] = settings.ENTREZ_EMAIL
        with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
            resp = client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params)
            resp.raise_for_status()
            xml = resp.text
    except Exception:
        return None

    # Minimal XML parsing for AbstractText
    try:
        import re
        parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, flags=re.DOTALL)
        if not parts:
            return None
        # Strip XML tags inside text
        clean = []
        for p in parts:
            p = re.sub(r"<[^>]+>", "", p)
            p = p.replace("\n", " ").strip()
            if p:
                clean.append(p)
        return " ".join(clean).strip() if clean else None
    except Exception:
        return None


def _looks_like_rss(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return "rss" in u or u.endswith(".xml") or "feed" in u


def load_feeds_from_csv(csv_path: str) -> List[FeedConfig]:
    """
    Load feed URLs from a CSV. Prefer `feed_url` column when present.
    Falls back to `url` if it looks like a feed.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip().lower() for c in df.columns]

    feed_col = "feed_url" if "feed_url" in df.columns else "url"
    flag_col = "is_feed" if "is_feed" in df.columns else None

    feeds: List[FeedConfig] = []
    for _, row in df.iterrows():
        raw_url = row.get(feed_col)
        if raw_url is None or (isinstance(raw_url, float) and pd.isna(raw_url)):
            continue
        url = str(raw_url).strip()
        if not url or url.lower() == "nan":
            continue
        # If is_feed column exists, require explicit true
        if flag_col:
            flag_val = str(row.get(flag_col) or "").strip().lower()
            if flag_val not in {"true", "1", "yes", "y"}:
                continue
        if not url:
            continue
        if feed_col == "url" and not _looks_like_rss(url):
            continue
        source = str(row.get("source") or "").strip() or "CSV Feed"
        feeds.append(FeedConfig(url=url, source=source))

    # De-dup by URL
    seen = set()
    unique = []
    for f in feeds:
        if f.url in seen:
            continue
        seen.add(f.url)
        unique.append(f)
    return unique


def fetch_feed_entries(feed: FeedConfig) -> List[dict]:
    """
    Fetch a single RSS/Atom feed and return normalized entries ready for DB insertion.
    """
    parsed = feedparser.parse(feed.url)
    if not parsed.entries:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG_PharmaChatBot/1.0)"}
            with httpx.Client(timeout=20, follow_redirects=True, headers=headers) as client:
                resp = client.get(feed.url)
                resp.raise_for_status()
                parsed = feedparser.parse(resp.content)
        except Exception:
            pass
    rows: List[dict] = []

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        url = (entry.get("link") or "").strip()
        if not url:
            continue

        summary = (entry.get("summary") or "").strip()
        author = (entry.get("author") or "").strip()
        published_at = _parse_published(entry.get("published") or entry.get("updated"))
        published_date = published_at.date() if published_at else None

        content_text = None
        pmid = _extract_pubmed_id(url)
        if pmid:
            content_text = _fetch_pubmed_abstract(pmid)
        if not content_text:
            content_text = _fetch_article_text(url)
        document_text = " ".join([p for p in [title, summary, content_text] if p])

        event_id = _hash_event_id(url, title, published_at)
        tags = []
        try:
            tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
        except Exception:
            tags = []

        row = {
            "event_id": event_id,
            "published_at": published_at,
            "published_date": published_date,
            "source_type": "rss_article",
            "source": feed.source,
            "domain": _extract_domain(url),
            "url": url,
            "title": title or None,
            "author": author or None,
            "summary": summary or None,
            "content_text": content_text or None,
            "document_text": document_text or None,
            "content_hash": event_id,
            "tags": tags or None,
            "therapeutic_area": None,
            "drug_name": None,
            "company": None,
            "content_type": None,
        }
        rows.append(row)

    return rows


def load_all_feeds(feeds: Optional[Iterable[FeedConfig]] = None) -> List[dict]:
    """
    Fetch all configured feeds and return a flat list of normalized rows.
    """
    if feeds is None:
        csv_path = getattr(settings, "WEB_CSV_INPUT_PATH", None)
        if csv_path:
            try:
                feeds = load_feeds_from_csv(csv_path)
            except Exception:
                feeds = []
        if not feeds:
            feeds = DEFAULT_FEEDS
    feeds = list(feeds)
    all_rows: List[dict] = []
    for f in feeds:
        all_rows.extend(fetch_feed_entries(f))
    return all_rows


if __name__ == "__main__":
    rows = load_all_feeds()
    print(f"Fetched {len(rows)} rows from RSS feeds")
