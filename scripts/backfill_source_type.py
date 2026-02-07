import os
import sys

sys.path.insert(0, os.getcwd())

from src.db.supabase import get_engine
from sqlalchemy import text


def main() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE public.web_suggestions SET source_type='seed' WHERE source_type IS NULL")
        )
    print("Backfilled source_type=seed for NULL rows")


if __name__ == "__main__":
    main()
