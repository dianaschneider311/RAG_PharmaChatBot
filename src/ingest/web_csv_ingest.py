"""
Load and normalize a web/RSS CSV into rows for public.web_suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import List, Dict, Any
from urllib.parse import urlparse

import pandas as pd


REQUIRED_COLUMNS = [
    "url",
    "title",
    "published_at",
    "source",
    "summary",
    "content_text",
]

KNOWN_COLUMNS = {
    "event_id",
    "published_at",
    "published_date",
    "source_type",
    "source",
    "domain",
    "url",
    "title",
    "author",
    "summary",
    "content_text",
    "document_text",
    "content_hash",
    "tags",
    "therapeutic_area",
    "drug_name",
    "company",
    "content_type",
}


def _hash_event_id(url: str, title: str, published_at: datetime | None) -> str:
    base = f"{url}|{title}|{published_at.isoformat() if published_at else ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _normalize_tags(value) -> List[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    # comma-separated
    return [p.strip() for p in s.split(",") if p.strip()]


def load_web_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load CSV, normalize columns, and return list of rows for web_suggestions.
    """
    df = pd.read_csv(path)

    # Drop unnamed/empty columns from trailing commas
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalize column names to snake_case (lower)
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in web CSV: {missing}")

    # Coerce published_at to datetime (UTC if possible)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    if "published_date" not in df.columns:
        df["published_date"] = df["published_at"].dt.date

    # Derive domain if missing
    if "domain" not in df.columns:
        df["domain"] = df["url"].apply(_extract_domain)

    # Default source_type for seed CSV rows
    if "source_type" not in df.columns:
        df["source_type"] = "seed"

    # Build document_text if missing or empty
    def _build_doc(row):
        doc = str(row.get("document_text") or "").strip()
        if doc:
            return doc
        parts = [row.get("title"), row.get("summary"), row.get("content_text")]
        parts = [str(p).strip() for p in parts if p is not None and str(p).strip()]
        return "\n".join(parts)

    df["document_text"] = df.apply(_build_doc, axis=1)

    # Generate event_id/content_hash if missing
    if "event_id" not in df.columns:
        df["event_id"] = df.apply(
            lambda r: _hash_event_id(
                str(r.get("url") or ""),
                str(r.get("title") or ""),
                r.get("published_at").to_pydatetime() if pd.notnull(r.get("published_at")) else None,
            ),
            axis=1,
        )

    if "content_hash" not in df.columns:
        df["content_hash"] = df["event_id"]

    # Normalize tags
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(_normalize_tags)

    # Ensure all known columns exist (fill missing with None)
    for col in KNOWN_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep only known columns for insert (stable order)
    keep_cols = [c for c in df.columns if c in KNOWN_COLUMNS]
    df = df[keep_cols]

    # Convert to list of dicts
    rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    return rows
