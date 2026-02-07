from src.db.supabase import get_engine
from sqlalchemy import text


def main() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        # Check pharma_suggestions exists
        pharma_cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='pharma_suggestions'"
            )
        ).fetchall()
        web_cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='web_suggestions'"
            )
        ).fetchall()

    pharma_colset = {r[0] for r in pharma_cols}
    web_colset = {r[0] for r in web_cols}

    required_web = {"event_id", "url", "title", "published_at", "source", "document_text", "source_type"}
    missing_web = required_web - web_colset

    print("pharma_suggestions columns:", len(pharma_colset))
    print("web_suggestions columns:", len(web_colset))
    if missing_web:
        print("Missing web_suggestions columns:", sorted(missing_web))
        raise SystemExit(1)
    print("Schema check OK")


if __name__ == "__main__":
    main()
