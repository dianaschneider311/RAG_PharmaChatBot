from src.ingest.csv_ingest import load_clean_csv
from src.config.settings import settings
import pandas as pd
import hashlib

df = load_clean_csv(settings.CSV_INPUT_PATH)
print('First 5 rows:')
print(df[['suggestion_date', 'rep_name', 'hcp_name']].head())

# Check for duplicates in key fields
print('\nChecking for duplicate rows...')
dup_check = df[['suggestion_date', 'rep_name', 'hcp_name', 'suggestion_reason']].duplicated().sum()
print(f'Duplicate rows (exact): {dup_check}')

# Check event_id generation
df['event_id'] = df.apply(
    lambda row: hashlib.md5(
        f"{row.get('suggestion_date', '')}|{row.get('rep_name', '')}|{row.get('hcp_name', '')}|{row.get('suggestion_reason', '')}".encode()
    ).hexdigest(),
    axis=1
)
print(f'\nEvent ID duplicates: {df["event_id"].duplicated().sum()}')
if df["event_id"].duplicated().sum() > 0:
    dups = df[df["event_id"].duplicated(keep=False)].sort_values('event_id')
    print(f'\nFirst few duplicate event_ids:')
    print(dups[['event_id', 'rep_name', 'hcp_name', 'suggestion_date']].head(10))
