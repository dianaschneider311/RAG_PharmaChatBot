from src.db.supabase import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema='public' AND table_name='pharma_suggestions'
        ORDER BY ordinal_position
    """))
    cols = result.fetchall()
    print(f"Total columns: {len(cols)}\n")
    for i, (name, dtype) in enumerate(cols, 1):
        print(f"{i:2}. {name:40} {dtype}")
    
    # Check sample data
    print("\n" + "="*60)
    print("Sample data from first 3 rows:")
    print("="*60)
    data = conn.execute(text("SELECT * FROM pharma_suggestions LIMIT 3"))
    for row in data:
        print(row)
