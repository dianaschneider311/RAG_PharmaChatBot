"""
Insert web/RSS CSV rows into public.web_suggestions.
"""

from __future__ import annotations

from typing import List, Dict, Any

from sqlalchemy import text

from src.config.settings import settings
from src.db.supabase import get_engine
from src.ingest.web_csv_ingest import load_web_csv


def insert_web_csv_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    engine = get_engine()
    insert_sql = text(
        """
        INSERT INTO public.web_suggestions (
            event_id,
            published_at,
            published_date,
            source_type,
            source,
            domain,
            url,
            title,
            author,
            summary,
            content_text,
            document_text,
            content_hash,
            tags,
            therapeutic_area,
            drug_name,
            company,
            content_type,
            updated_at
        ) VALUES (
            :event_id,
            :published_at,
            :published_date,
            :source_type,
            :source,
            :domain,
            :url,
            :title,
            :author,
            :summary,
            :content_text,
            :document_text,
            :content_hash,
            :tags,
            :therapeutic_area,
            :drug_name,
            :company,
            :content_type,
            now()
        )
        ON CONFLICT (event_id) DO UPDATE SET
            published_at = EXCLUDED.published_at,
            published_date = EXCLUDED.published_date,
            source_type = EXCLUDED.source_type,
            source = EXCLUDED.source,
            domain = EXCLUDED.domain,
            url = EXCLUDED.url,
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            summary = EXCLUDED.summary,
            content_text = EXCLUDED.content_text,
            document_text = EXCLUDED.document_text,
            content_hash = EXCLUDED.content_hash,
            tags = EXCLUDED.tags,
            therapeutic_area = EXCLUDED.therapeutic_area,
            drug_name = EXCLUDED.drug_name,
            company = EXCLUDED.company,
            content_type = EXCLUDED.content_type,
            updated_at = now()
        """
    )

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)
    return len(rows)


def main() -> None:
    path = getattr(settings, "WEB_CSV_INPUT_PATH", None)
    if not path:
        raise ValueError("WEB_CSV_INPUT_PATH not set in .env")
    rows = load_web_csv(path)
    count = insert_web_csv_rows(rows)
    print(f"Inserted/updated {count} web_suggestions rows from CSV")


if __name__ == "__main__":
    main()
