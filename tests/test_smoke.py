import os
import sys

sys.path.insert(0, os.getcwd())

import pytest

from src.ingest.web_csv_ingest import load_web_csv
from src.retrieval.rag import route_query


def test_route_query_pharma():
    route = route_query("Show HCP outreach suggestions for neurology", debug=False)
    assert route == "pharma"


def test_web_csv_load():
    path = os.path.join("data", "input", "pharma_external_sources_seed.csv")
    rows = load_web_csv(path)
    assert isinstance(rows, list)
    assert len(rows) >= 1
