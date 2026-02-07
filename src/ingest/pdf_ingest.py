"""
Ingest local PDF files, extract text, and chunk into rows for web_suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class PdfDoc:
    path: Path
    title: str


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()


def load_pdf_folder(folder: str) -> List[Dict[str, Any]]:
    base = Path(folder)
    rows: List[Dict[str, Any]] = []
    for path in base.rglob("*.pdf"):
        title = path.stem
        full_text = _read_pdf_text(path)
        chunks = _chunk_text(full_text)
        parent_event_id = _hash_id(str(path.resolve()))
        for idx, chunk in enumerate(chunks):
            event_id = _hash_id(f"{parent_event_id}:{idx}")
            rows.append(
                {
                    "event_id": event_id,
                    "parent_event_id": parent_event_id,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "title": title,
                    "source_type": "pdf",
                    "source": "local_pdf",
                    "url": str(path.resolve()),
                    "content_text": chunk,
                    "document_text": chunk,
                    "content_hash": event_id,
                    "content_type": "pdf",
                }
            )
    return rows
