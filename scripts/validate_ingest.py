import json
import os
import sys
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, os.getcwd())

from src.config.settings import settings
from src.ingest.csv_ingest import load_clean_csv


def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def _validate_csv(path: str, required_cols: list[str], min_doc_len: int = 40, use_pharma_parser: bool = False):
    if use_pharma_parser:
        df = load_clean_csv(path)
    else:
        df = pd.read_csv(path, engine="python")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip().lower() for c in df.columns]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {
            "path": path,
            "error": f"missing required columns: {missing_cols}",
            "rows": 0,
        }, None

    # Basic checks
    report = {
        "path": path,
        "rows": len(df),
        "missing_values": {},
        "invalid_urls": 0,
        "short_document_text": 0,
        "warnings": [],
    }

    for col in required_cols:
        report["missing_values"][col] = int(df[col].isna().sum())

    if "url" in df.columns:
        report["invalid_urls"] = int(
            (~df["url"].fillna("").apply(_is_valid_url)).sum()
        )

    if "document_text" in df.columns:
        report["short_document_text"] = int(
            df["document_text"].fillna("").apply(lambda x: len(str(x).strip()) < min_doc_len).sum()
        )
    else:
        report["warnings"].append("document_text not present; cannot validate length")

    # Quarantine rows with fatal issues
    quarantine = df.copy()
    bad = pd.Series(False, index=quarantine.index)
    if "url" in df.columns:
        bad |= ~quarantine["url"].fillna("").apply(_is_valid_url)
    if "document_text" in df.columns:
        bad |= quarantine["document_text"].fillna("").apply(lambda x: len(str(x).strip()) < min_doc_len)
    quarantine = quarantine[bad]

    return report, quarantine


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/quarantine", exist_ok=True)

    reports = []

    # Pharma CSV validation
    pharma_required = [
        "rep_name",
        "hcp_name",
        "hcp_speciality",
        "suggested_channel",
        "suggestion_reason",
    ]
    rep, quarantine = _validate_csv(settings.CSV_INPUT_PATH, pharma_required, min_doc_len=20, use_pharma_parser=True)
    reports.append({"name": "pharma_csv", **rep})
    if quarantine is not None and len(quarantine) > 0:
        quarantine.to_csv("data/quarantine/pharma_bad_rows.csv", index=False)

    # Web CSV validation
    web_required = [
        "url",
        "title",
        "published_at",
        "source",
        "summary",
        "content_text",
    ]
    rep, quarantine = _validate_csv(settings.WEB_CSV_INPUT_PATH, web_required, min_doc_len=40)
    reports.append({"name": "web_csv", **rep})
    if quarantine is not None and len(quarantine) > 0:
        quarantine.to_csv("data/quarantine/web_bad_rows.csv", index=False)

    report_path = f"logs/validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"reports": reports}, f, indent=2)

    print(f"Wrote validation report: {report_path}")


if __name__ == "__main__":
    main()
