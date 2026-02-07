# Copilot Instructions for RAG_PharmaChatBot

## Project Overview

RAG_PharmaChatBot is a **Retrieval-Augmented Generation** system for pharma sales enablement. It ingests pharmaceutical representative suggestions from CSV, stores them in Postgres (via Supabase), and will enable semantic search + LLM Q&A over this data. Current status: **data ingestion pipeline complete; RAG/API layers under development**.

## Architecture

```
CSV Data → Pandas Cleaner → Postgres/Supabase → [Qdrant Vector DB] → LLM Q&A → FastAPI
              (src/ingest/)      (src/db/)          (planned)           (planned)    (planned)
```

### Key Components

- **`src/config/settings.py`**: Pydantic Settings class—all env vars (Supabase, Qdrant, OpenRouter) are defined here with validation. Load via `from src.config.settings import settings`.
- **`src/db/supabase.py`**: SQLAlchemy engine factory. Use `get_engine()` to connect to Postgres.
- **`src/db/create_tables.py`**: Schema definition. Table: `public.suggestions` with pk=`event_id`, indexed on date_sec, rep_name, hcp_name, channel, product_to.
- **`src/ingest/csv_ingest.py`**: Strict CSV parser with regex-based newline collapse (handles embedded quotes/newlines). Validates 17 columns. Returns cleaned pandas DataFrame.
- **`src/db/insert_csv.py`**: Main ingestion script—loads CSV, connects to DB, inserts via `df.to_sql()` with chunksize=500.
- **`src/retrieval/rag.py`**: (empty, planned) Will contain vector embedding + Qdrant retriever + LLM chain.
- **`src/api/main.py`**: (empty, planned) FastAPI app for Q&A endpoints.

## Critical Developer Workflows

### 1. Environment Setup
- Copy `.env.example` to `.env` (not in git).
- Required vars: `SUPABASE_DB__*` (5 vars), `QDRANT__*` (3 vars), `OPENROUTER__API_KEY`.
- All settings are case-sensitive (Pydantic enforces via `case_sensitive=True`).

### 2. CSV Ingestion Pipeline
```powershell
# Verify DB connection
python -m src.db.supabase

# Create schema
python -m src.db.create_tables

# Ingest CSV (full pipeline)
python -m src.db.insert_csv
```

Ingestion is idempotent for new rows but will fail on duplicate `event_id` (primary key). CSV parsing is **very strict**—newlines inside field values are collapsed by regex; all quotes are stripped.

### 3. Testing DB Connectivity
```python
from src.db.supabase import get_engine, test_connection
test_connection()  # Prints "Supabase DB connection OK"
```

## Project-Specific Patterns & Conventions

1. **Configuration**: Use Pydantic Settings, never hardcode. Settings are singleton instance `settings` in `src/config/settings.py`.
2. **Database**: All DB access uses SQLAlchemy `create_engine()` from `src/db/supabase.py`. No raw psycopg2.
3. **CSV Parsing**: The input CSV has **embedded newlines within fields** and **quoted text**. Always use `load_clean_csv()` from `src/ingest/csv_ingest.py`—it handles stripping quotes and collapsing internal newlines via regex.
4. **Schema Validation**: CSV must have exactly 17 columns (listed in `EXPECTED_COLUMNS`). Mismatch raises `ValueError` immediately.
5. **Naming**: Table name is `pharma_suggestions` (in insert) but SQL schema defines `public.suggestions`—both are equivalent (default schema=public).

## Integration Points & External Dependencies

| Component | Purpose | Config Var(s) |
|-----------|---------|---------------|
| **Supabase Postgres** | Primary data store (suggestions table) | `SUPABASE_DB__*` (5 vars) |
| **Qdrant** | Vector DB (future: semantic search) | `QDRANT__URL`, `QDRANT__API_KEY`, `QDRANT__COLLECTION` |
| **OpenRouter API** | LLM calls (future: Q&A chain) | `OPENROUTER__API_KEY`, `OPENROUTER__API_URL` |
| **pandas** | DataFrame ops (CSV→table) | — |
| **SQLAlchemy** | ORM & connection pooling | — |
| **FastAPI** | REST API (planned) | — |

## Common Tasks & Examples

### Add a New Field to Suggestions Table
1. Update `EXPECTED_COLUMNS` in `src/ingest/csv_ingest.py`.
2. Update `CREATE_SQL` in `src/db/create_tables.py` to include the column.
3. Re-run `python -m src.db.create_tables` (idempotent, won't drop existing data).
4. Re-ingest data with updated CSV.

### Query Suggestions Directly
```python
from src.db.supabase import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM public.suggestions"))
    print(result.scalar())
```

### Add Environment Variable
1. Add field to `Settings` class in `src/config/settings.py` with proper type hints.
2. Add to `.env` and `.env.example`.
3. Access via `from src.config.settings import settings; settings.YOUR_VAR`.

## Notes on Incomplete Sections

- **`src/retrieval/rag.py`**: Plan to implement Qdrant retriever + LLM chain here. Will use LangChain or direct OpenRouter API calls.
- **`src/api/main.py`**: FastAPI app—likely will have `/query` endpoint that takes a question and returns LLM response + retrieved context.
- **`scripts/run_ingest_csv.ps1`**: PowerShell wrapper for ingestion (currently empty)—consider adding for Windows automation.

---

**Last Updated**: 2026-01-29 | **Status**: Data ingestion ✅, RAG/API in progress
