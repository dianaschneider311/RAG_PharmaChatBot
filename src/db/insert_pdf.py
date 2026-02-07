"""
Insert PDF chunk rows into public.web_suggestions.
"""

from __future__ import annotations

from typing import List, Dict, Any

from sqlalchemy import text

from src.config.settings import settings
from src.db.supabase import get_engine
from src.ingest.pdf_ingest import load_pdf_folder


def insert_pdf_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    engine = get_engine()
    insert_sql = text(
        """
        INSERT INTO public.web_suggestions (
            event_id,
            parent_event_id,
            chunk_index,
            chunk_count,
            source_type,
            source,
            url,
            title,
            content_text,
            document_text,
            content_hash,
            content_type,
            updated_at
        ) VALUES (
            :event_id,
            :parent_event_id,
            :chunk_index,
            :chunk_count,
            :source_type,
            :source,
            :url,
            :title,
            :content_text,
            :document_text,
            :content_hash,
            :content_type,
            now()
        )
        ON CONFLICT (event_id) DO UPDATE SET
            parent_event_id = EXCLUDED.parent_event_id,
            chunk_index = EXCLUDED.chunk_index,
            chunk_count = EXCLUDED.chunk_count,
            source_type = EXCLUDED.source_type,
            source = EXCLUDED.source,
            url = EXCLUDED.url,
            title = EXCLUDED.title,
            content_text = EXCLUDED.content_text,
            document_text = EXCLUDED.document_text,
            content_hash = EXCLUDED.content_hash,
            content_type = EXCLUDED.content_type,
            updated_at = now()
        """
    )

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)
    return len(rows)


def main() -> None:
    folder = getattr(settings, "PDF_INPUT_PATH", None)
    if not folder:
        raise ValueError("PDF_INPUT_PATH not set in .env")
    rows = load_pdf_folder(folder)
    count = insert_pdf_rows(rows)
    print(f"Inserted/updated {count} pdf chunks into web_suggestions")


if __name__ == "__main__":
    main()
