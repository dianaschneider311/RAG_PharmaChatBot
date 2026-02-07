from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from src.config.settings import settings

def get_engine():
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=settings.SUPABASE_DB__USER,
        password=settings.SUPABASE_DB__PASSWORD,
        host=settings.SUPABASE_DB__HOST,
        port=int(settings.SUPABASE_DB__PORT),
        database=settings.SUPABASE_DB__NAME,
    )
    return create_engine(url, pool_pre_ping=True)

def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    print("Supabase DB connection OK")
