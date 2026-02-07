import os
import sys

sys.path.insert(0, os.getcwd())

from src.retrieval.retriever import search_web_suggestions


def main():
    query = "latest FDA drug safety communication"
    results = search_web_suggestions(
        query=query,
        limit=5,
        source=None,
        domain=None,
        content_type=None,
        tags=None,
        published_from=None,
        published_to=None,
        debug=False,
    )

    print(f"Query: {query}")
    print(f"Returned {len(results)} results")
    for r in results:
        print("-", r.get("published_at"), r.get("source"), r.get("title"), r.get("url"))


if __name__ == "__main__":
    main()
