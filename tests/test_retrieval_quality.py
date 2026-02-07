import os
import sys

sys.path.insert(0, os.getcwd())

from src.retrieval.retriever import search_suggestions, search_web_suggestions
from src.retrieval.rag import route_query


def test_route_query_web_keywords():
    q = "Latest FDA safety communication about ocrelizumab"
    assert route_query(q) == "web"


def test_route_query_pharma_keywords():
    q = "For REP John, list HCPs with neurology specialty"
    assert route_query(q) == "pharma"


def test_web_retrieval_returns_results():
    results = search_web_suggestions(
        query="ocrelizumab multiple sclerosis",
        source_type="rss_article",
        limit=3,
        score_threshold=0.0,
    )
    assert isinstance(results, list)
    assert len(results) >= 1


def test_pharma_retrieval_returns_results():
    results = search_suggestions(
        query="neurology outreach",
        limit=3,
        score_threshold=0.0,
    )
    assert isinstance(results, list)
