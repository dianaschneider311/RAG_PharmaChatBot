# src/ingest/row_builder.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from src.retrieval.embedding_schema import VECTOR_TEXT_FIELDS, OPTIONAL_CONTEXT_FIELDS


def _norm(v: Any) -> str:
    """Normalize values into a stable string for hashing / text building."""
    if v is None:
        return ""
    s = str(v).strip()
    # keep it simple & stable (no lowercasing unless you want case-insensitive IDs)
    return s


def make_row_id(row: dict, row_num: int) -> str:
    """
    Stable unique row identifier.
    row_num guarantees uniqueness even for identical business rows.
    """
    payload = {
        "row_num": row_num,
        "suggestion_date": row.get("suggestion_date"),
        "rep_name": row.get("rep_name"),
        "hcp_name": row.get("hcp_name"),
        "suggested_channel": row.get("suggested_channel"),
        "suggestion_reason": row.get("suggestion_reason"),
        "source": row.get("source"),
    }

    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def build_document_text(row: Dict[str, Any]) -> str:
    """
    Text that gets embedded.
    Base = VECTOR_TEXT_FIELDS
    Optionally append a compact context block for better semantic recall.
    """
    chunks = []

    # main semantic fields
    for k in VECTOR_TEXT_FIELDS:
        v = _norm(row.get(k))
        if v:
            chunks.append(v)

    # optional context
    ctx_lines = []
    for k in OPTIONAL_CONTEXT_FIELDS:
        v = _norm(row.get(k))
        if v:
            ctx_lines.append(f"{k}: {v}")

    if ctx_lines:
        chunks.append("\n\nContext:\n" + "\n".join(ctx_lines))

    return "\n\n---\n\n".join(chunks).strip()
