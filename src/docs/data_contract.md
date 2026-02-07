# Data Contract — Pharma Suggestions RAG

## Source
- CSV ingestion → Supabase `pharma_suggestions`

## One row = one suggestion event

## Required columns (retrieval-critical)

| Column name | Purpose |
|------------|--------|
| row_id | Stable unique identifier (hash) |
| document_text | Text embedded + retrieved |
| suggestion_date | Temporal filtering |
| rep_name | Metadata filter |
| hcp_name | Metadata filter |
| suggested_channel | Metadata filter |
| hcp_speciality | Metadata filter |
| adjusted_expected_value | Ranking / boosting |
| aktana_hcp_value | Ranking / boosting |
| source | Provenance |

## Vectorization rules
- Only `document_text` is embedded
- All other fields are metadata only
- Metadata must NEVER be embedded

## Mutability rules
- `row_id` is immutable
- `document_text` changes → new `row_id`
