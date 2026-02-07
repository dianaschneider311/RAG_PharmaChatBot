# src/ingest/csv_ingest.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from collections import Counter
from io import StringIO
import math
import re

import pandas as pd

from src.ingest.row_builder import make_row_id, build_document_text

# Map raw CSV headers -> canonical keys (snake_case)
CSV_TO_CANONICAL: Dict[str, str] = {
    "Suggestion Date": "suggestion_date",
    "Rep Name": "rep_name",
    "Suggested Channel": "suggested_channel",
    "HCP Priority Trend for Ocrevus": "hcp_priority_trend_ocrevus",
    "Adjusted Expected Value": "adjusted_expected_value",
    "Aktana HCP Value": "aktana_hcp_value",
    "Suggestion Reason": "suggestion_reason",
    "HCP Name": "hcp_name",
    "Is Suggestion Recommended": "is_suggestion_recommended",
    "Facility Address": "facility_address",
    "HCP Speciality": "hcp_speciality",
    "Is Visit Channel Propensity High": "is_visit_channel_propensity_high",
    "source": "source",
    "Product switch from": "product_switch_from",
    "Product switch to": "product_switch_to",
    "New prescriber": "new_prescriber",
    "Repeat prescriber": "repeat_prescriber",
}

# Your canonical schema (names only; order doesn't matter)
EXPECTED_COLUMNS = [
    "Suggestion Date",
    "Rep Name",
    "HCP Name",
    "Suggested Channel",
    "source",
    "Facility Address",
    "HCP Speciality",
    "Suggestion Reason",
    "HCP Priority Trend for Ocrevus",
    "Adjusted Expected Value",
    "Aktana HCP Value",
    "Is Suggestion Recommended",
    "Is Visit Channel Propensity High",
    "Product switch from",
    "Product switch to",
    "New prescriber",
    "Repeat prescriber",
]

def load_clean_csv(path: str) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # 1) Parse with quotes ENABLED so multiline fields stay in one row
    df = pd.read_csv(
        StringIO(text),
        sep="|",
        engine="python",
        dtype=str,
        quotechar='"',          # IMPORTANT
        escapechar="\\",        # safe default
        keep_default_na=False,
    )

    # 2) Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # 3) Validate columns by NAME (order-independent) — raw CSV headers
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 4) Reorder to canonical order (raw headers)
    df = df[EXPECTED_COLUMNS]

    # 5) Clean values
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.replace('"', '', regex=False)

    # 6) RENAME raw headers -> canonical snake_case  ✅ THIS IS THE FIX
    df = df.rename(columns=CSV_TO_CANONICAL)
    missing2 = set(CSV_TO_CANONICAL.values()) - set(df.columns)
    if missing2:
       raise ValueError(f"Missing canonical columns after rename: {missing2}")

    # 7) Build suggestion_reason_clean using generic sentence frequency + IDF
    if "suggestion_reason" in df.columns:
        def _split_sentences(text: str) -> List[str]:
            parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
            return [p.strip() for p in parts if p.strip()]

        def _tokenize(text: str) -> List[str]:
            return [t.lower() for t in re.findall(r"[A-Za-z]+", text)]

        reasons = [str(r or "") for r in df["suggestion_reason"].tolist()]
        per_doc_sents = []
        all_sents = []
        for r in reasons:
            sents = _split_sentences(r)
            per_doc_sents.append(sents)
            all_sents.extend([s.lower() for s in sents])

        sent_counts = Counter(all_sents)
        total_sents = len(all_sents) if all_sents else 1

        token_df = Counter()
        for s in set(all_sents):
            for t in set(_tokenize(s)):
                token_df[t] += 1

        def _avg_idf(sentence: str) -> float:
            toks = _tokenize(sentence)
            if not toks:
                return 0.0
            idfs = []
            for t in toks:
                df_count = token_df.get(t, 0)
                idfs.append(math.log((total_sents + 1) / (df_count + 1)) + 1.0)
            return sum(idfs) / len(idfs)

        freq_threshold = 0.2
        idf_threshold = 1.2
        cleaned = []
        for sents in per_doc_sents:
            kept = []
            for s in sents:
                s_l = s.lower()
                freq = sent_counts.get(s_l, 0) / total_sents
                if freq > freq_threshold:
                    continue
                if _avg_idf(s_l) < idf_threshold:
                    continue
                kept.append(s)
            cleaned_text = " ".join(kept).strip()
            cleaned.append(cleaned_text)
        df["suggestion_reason_clean"] = cleaned

    # 8) Build row_id + document_text using canonical keys
    df = df.reset_index(drop=True)

    df["row_id"] = [
        make_row_id(row.to_dict(), row_num=i)
        for i, row in df.iterrows()
    ]
    df["document_text"] = df.apply(build_document_text, axis=1)

    return df
