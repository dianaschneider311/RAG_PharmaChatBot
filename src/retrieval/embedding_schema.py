# src/retrieval/embedding_schema.py

# Fields that should become the main "document text" for embeddings
VECTOR_TEXT_FIELDS = [
    "suggestion_reason",
]

# Extra fields that should be appended to document_text when present
OPTIONAL_CONTEXT_FIELDS = [
    "suggestion_date",
    "rep_name",
    "hcp_name",
    "suggested_channel",
    "hcp_priority_trend_ocrevus",
    "adjusted_expected_value",
    "aktana_hcp_value",
    "facility_address",
    "hcp_speciality",
    "is_visit_channel_propensity_high",
    "source",
    "product_switch_from",
    "product_switch_to",
    "new_prescriber",
    "repeat_prescriber",
]


