from sqlalchemy import text
from src.db.supabase import get_engine

# Expected column definitions (used to create table or add missing columns)
EXPECTED_COLUMNS = {
  "suggestion_date": "date",
  "suggestion_date_sec": "bigint",
  "rep_name": "text",
  "hcp_name": "text",
  "hcp_speciality": "text[]",
  "facility_address": "text",
  "suggested_channel": "text check (suggested_channel in ('SEND','VISIT'))",
  "suggestion_reason": "text",
  "suggestion_reason_clean": "text",
  "product_switch_from": "text",
  "product_switch_to": "text",
  "new_prescriber": "text",
  "repeat_prescriber": "text",
  "adjusted_expected_value": "text check (adjusted_expected_value in ('High','Medium','Low','Very Low'))",
  "aktana_hcp_value": "text check (aktana_hcp_value in ('High','Medium','Low','Very Low'))",
  "hcp_priority_trend_ocrevus": "text",
  "is_visit_channel_propensity_high": "boolean",
  "is_suggestion_recommended": "boolean",
  "source": "text default 'pharma_suggestions'",
  "rendered_text": "text",
  "content_hash": "text",
  "created_at": "timestamptz default now()",
  "updated_at": "timestamptz default now()",
  "row_id": "text",
  "document_text": "text",
}

INDEX_DEFS = [
  ("idx_pharma_suggestions_date_sec", "suggestion_date_sec"),
  ("idx_pharma_suggestions_rep_name", "rep_name"),
  ("idx_pharma_suggestions_hcp_name", "hcp_name"),
  ("idx_pharma_suggestions_channel", "suggested_channel"),
  ("idx_pharma_suggestions_product_to", "product_switch_to"),
]


def _create_table_if_missing(conn):
  # Build CREATE TABLE with expected columns (event_id is primary key)
  cols_sql = ["event_id text primary key"]
  for name, definition in EXPECTED_COLUMNS.items():
    cols_sql.append(f"{name} {definition}")

  create_sql = f"create table if not exists public.pharma_suggestions (\n  " + ",\n  ".join(cols_sql) + "\n);"
  conn.execute(text(create_sql))
  print("Table ensured (CREATE TABLE IF NOT EXISTS)")

  # Web/RSS ingestion table
  conn.execute(text(
    "create table if not exists public.web_suggestions ("
    "  event_id text primary key,"
    "  published_at timestamptz,"
    "  published_date date,"
    "  source_type text,"
    "  source text,"
    "  domain text,"
    "  url text,"
    "  title text,"
    "  author text,"
    "  summary text,"
    "  content_text text,"
    "  document_text text,"
    "  content_hash text,"
    "  tags text[],"
    "  parent_event_id text,"
    "  chunk_index integer,"
    "  chunk_count integer,"
    "  therapeutic_area text,"
    "  drug_name text,"
    "  company text,"
    "  content_type text,"
    "  created_at timestamptz default now(),"
    "  updated_at timestamptz default now()"
    ")"
  ))
  print("Table ensured: public.web_suggestions")

  # Synonym map for channel normalization (e.g., "email" -> "SEND")
  conn.execute(text(
    "create table if not exists public.channel_synonyms ("
    "  raw text primary key,"
    "  normalized text not null,"
    "  updated_at timestamptz default now()"
    ")"
  ))
  print("Table ensured: public.channel_synonyms")


def _ensure_missing_columns(conn):
  existing = {r[0] for r in conn.execute(text(
    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='pharma_suggestions'"
  )).fetchall()}

  for name, definition in EXPECTED_COLUMNS.items():
    if name not in existing:
      try:
        conn.execute(text(f"ALTER TABLE public.pharma_suggestions ADD COLUMN IF NOT EXISTS {name} {definition}"))
        print(f"Added missing column: {name}")
      except Exception as e:
        print(f"Warning: failed to add column {name}: {e}")

  # Ensure web_suggestions has source_type column if table already exists
  web_existing = {r[0] for r in conn.execute(text(
    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='web_suggestions'"
  )).fetchall()}
  if "source_type" not in web_existing:
    try:
      conn.execute(text("ALTER TABLE public.web_suggestions ADD COLUMN IF NOT EXISTS source_type text"))
      print("Added missing column: web_suggestions.source_type")
    except Exception as e:
      print(f"Warning: failed to add column web_suggestions.source_type: {e}")
  for col in ("parent_event_id", "chunk_index", "chunk_count"):
    if col not in web_existing:
      try:
        conn.execute(text(f"ALTER TABLE public.web_suggestions ADD COLUMN IF NOT EXISTS {col} text" if col == "parent_event_id" else f"ALTER TABLE public.web_suggestions ADD COLUMN IF NOT EXISTS {col} integer"))
        print(f"Added missing column: web_suggestions.{col}")
      except Exception as e:
        print(f"Warning: failed to add column web_suggestions.{col}: {e}")


def _create_indexes(conn):
  # Refresh column list
  existing = {r[0] for r in conn.execute(text(
    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='pharma_suggestions'"
  )).fetchall()}

  for idx_name, col in INDEX_DEFS:
    if col in existing:
      try:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON public.pharma_suggestions ({col})"))
      except Exception as e:
        print(f"Warning: failed to create index {idx_name} on {col}: {e}")
    else:
      print(f"Skipping index {idx_name}: column {col} does not exist yet")

  # Indexes for web_suggestions (create if table exists)
  web_cols = {r[0] for r in conn.execute(text(
    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='web_suggestions'"
  )).fetchall()}
  web_indexes = [
    ("idx_web_suggestions_published_at", "published_at"),
    ("idx_web_suggestions_source", "source"),
    ("idx_web_suggestions_domain", "domain"),
    ("idx_web_suggestions_url", "url"),
  ]
  for idx_name, col in web_indexes:
    if col in web_cols:
      try:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON public.web_suggestions ({col})"))
      except Exception as e:
        print(f"Warning: failed to create index {idx_name} on {col}: {e}")
    else:
      print(f"Skipping index {idx_name}: column {col} does not exist yet")

def _ensure_column_types(conn):
  # Ensure hcp_speciality is text[] (array) if it already exists as text
  col_info = conn.execute(text(
    "SELECT data_type, udt_name FROM information_schema.columns "
    "WHERE table_schema='public' AND table_name='pharma_suggestions' "
    "AND column_name='hcp_speciality'"
  )).fetchone()

  if col_info:
    data_type, udt_name = col_info
    is_array = (data_type == "ARRAY") or (udt_name or "").startswith("_")
    if not is_array:
      # Convert comma-separated strings into text[]
      conn.execute(text(
        "ALTER TABLE public.pharma_suggestions "
        "ALTER COLUMN hcp_speciality TYPE text[] "
        "USING CASE "
        "  WHEN hcp_speciality IS NULL THEN NULL "
        "  WHEN hcp_speciality = '' THEN ARRAY[]::text[] "
        "  ELSE string_to_array(hcp_speciality, ',') "
        "END"
      ))
      print("Altered column hcp_speciality to text[]")

def main():
  engine = get_engine()
  with engine.begin() as conn:
    _create_table_if_missing(conn)
    _ensure_missing_columns(conn)
    _ensure_column_types(conn)
    _create_indexes(conn)

  print("Tables/indexes ensured OK")


if __name__ == "__main__":
  main()
