from src.db.supabase import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.begin() as conn:
    # Drop the existing table
    conn.execute(text("DROP TABLE IF EXISTS public.pharma_suggestions CASCADE"))
    print("✅ Dropped old table with 42 columns")

print("\nNow recreating clean schema...")
from src.db.create_tables import main as create_tables_main
create_tables_main()
print("✅ Created new clean table")

print("\nNow re-ingesting CSV data...")
from src.db.insert_csv import main as insert_csv_main
insert_csv_main()
print("✅ CSV ingestion complete")
