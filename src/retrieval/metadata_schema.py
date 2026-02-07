# src/retrieval/metadata_schema.py

DOCUMENT_CONTENT_DESCRIPTION = (
    "Suggestion for interaction between Pharmaceutical sales REP and Healthcare professional (HCP)."
)

METADATA_FIELD_INFO = [
    {"name": "suggestion_date", "description": "Date the suggestion was generated (YYYY-MM-DD).", "type": "date"},
    {"name": "suggestion_date_sec", "description": "Unix epoch seconds UTC for suggestion_date (int).", "type": "integer"},
    {"name": "hcp_name", "description": "Name of the HCP/doctor.", "type": "string"},
    {"name": "rep_name", "description": "Sales representative name.", "type": "string"},
    {"name": "suggested_channel", "description": "Recommended channel. Valid: SEND, VISIT.", "type": "string"},
    {"name": "product_switch_from", "description": "Switch from product. Valid list per business rules.", "type": "string"},
    {"name": "product_switch_to", "description": "Switch to product. Valid list per business rules.", "type": "string"},
    {"name": "new_prescriber", "description": "HCP is a new prescriber for a medication.", "type": "string"},
    {"name": "repeat_prescriber", "description": "HCP returns to prescribing the same medication.", "type": "string"},
    {"name": "hcp_speciality", "description": "HCP specialty (string or list).", "type": "string"},
    {"name": "source", "description": "Data source. Usually 'suggestions'.", "type": "string"},
    {"name": "adjusted_expected_value", "description": "Importance. Valid: High/Medium/Low/Very Low.", "type": "string"},
    {"name": "aktana_hcp_value", "description": "HCP importance. Valid: High/Medium/Low/Very Low.", "type": "string"},
    {"name": "hcp_priority_trend_ocrevus", "description": "Trend. Valid: Hold/Increase/Decline/Missing/Nan.", "type": "string"},
    {"name": "is_visit_channel_propensity_high", "description": "TRUE/FALSE.", "type": "boolean"},
    {"name": "is_suggestion_recommended", "description": "TRUE/FALSE.", "type": "boolean"},
    {"name": "suggestion_reason_clean", "description": "Suggestion reason with boilerplate removed.", "type": "string"},
]
