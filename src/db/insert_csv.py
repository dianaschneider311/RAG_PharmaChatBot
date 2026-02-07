import pandas as pd
from sqlalchemy import create_engine, text
import hashlib
from datetime import datetime

from src.ingest.csv_ingest import load_clean_csv
from src.config.settings import settings

TABLE_NAME = "pharma_suggestions"
SCHEMA_NAME = "public"


def get_engine():
    d = settings.model_dump()

    user = d["SUPABASE_DB__USER"]
    password = d["SUPABASE_DB__PASSWORD"]
    host = d["SUPABASE_DB__HOST"]
    port = d["SUPABASE_DB__PORT"]
    name = d["SUPABASE_DB__NAME"]

    url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True)


def _add_computed_columns(df):
    """Add missing columns that the DB schema expects but CSV doesn't have."""
    # Use row_id as event_id (it's already unique in the CSV)
    df["event_id"] = df["row_id"]
    
    # Add content_hash (hash of entire row)
    df["content_hash"] = df.apply(
        lambda row: hashlib.md5(str(row).encode()).hexdigest(),
        axis=1
    )
    
    # Add timestamps
    now = datetime.now()  # Use timezone-aware if possible
    df["created_at"] = now
    df["updated_at"] = now
    
    # Add missing optional fields (null/empty)
    if "rendered_text" not in df.columns:
        df["rendered_text"] = None
    if "source" not in df.columns:
        df["source"] = "pharma_suggestions"
    
    # Normalize hcp_speciality into list (comma-separated in CSV)
    if "hcp_speciality" in df.columns:
        def _to_list(v):
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return []
            return [p.strip() for p in s.split(",") if p.strip()]
        df["hcp_speciality"] = df["hcp_speciality"].apply(_to_list)

    # Convert suggestion_date to datetime if it's a string
    if "suggestion_date" in df.columns:
        try:
            df["suggestion_date"] = pd.to_datetime(df["suggestion_date"])
        except Exception as e:
            print(f"Warning: could not parse suggestion_date: {e}")
    
    # Add suggestion_date_sec (seconds since epoch)
    if "suggestion_date" in df.columns:
        df["suggestion_date_sec"] = df["suggestion_date"].apply(
            lambda x: int(pd.Timestamp(x).timestamp()) if pd.notna(x) else None
        )
    
    return df


def main():
    df = load_clean_csv(settings.CSV_INPUT_PATH)
    
    # Add computed columns that DB schema expects
    df = _add_computed_columns(df)
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"DataFrame columns: {list(df.columns)}")

    engine = get_engine()
    
    # Clean up dataframe - replace NaN with None for NULL handling
    df = df.where(pd.notna(df), None)
    
    # Use executemany with explicit column order to avoid aliases
    insert_sql = f"""
    INSERT INTO {SCHEMA_NAME}.{TABLE_NAME} (
        suggestion_date, rep_name, hcp_name, suggested_channel, source,
        facility_address, hcp_speciality, suggestion_reason, suggestion_reason_clean,
        hcp_priority_trend_ocrevus, adjusted_expected_value, aktana_hcp_value,
        is_suggestion_recommended, is_visit_channel_propensity_high,
        product_switch_from, product_switch_to, new_prescriber, repeat_prescriber,
        row_id, document_text, event_id, content_hash, created_at, updated_at,
        rendered_text, suggestion_date_sec
    ) VALUES (
        :suggestion_date, :rep_name, :hcp_name, :suggested_channel, :source,
        :facility_address, :hcp_speciality, :suggestion_reason, :suggestion_reason_clean,
        :hcp_priority_trend_ocrevus, :adjusted_expected_value, :aktana_hcp_value,
        :is_suggestion_recommended, :is_visit_channel_propensity_high,
        :product_switch_from, :product_switch_to, :new_prescriber, :repeat_prescriber,
        :row_id, :document_text, :event_id, :content_hash, :created_at, :updated_at,
        :rendered_text, :suggestion_date_sec
    )
    """
    
    # Convert rows to dicts for named parameter binding
    rows_as_dicts = df.to_dict('records')
    
    with engine.begin() as conn:
        for row_dict in rows_as_dicts:
            conn.execute(text(insert_sql), row_dict)
    
    print(f"Inserted {len(df)} rows into {SCHEMA_NAME}.{TABLE_NAME}")


if __name__ == "__main__":
    main()

