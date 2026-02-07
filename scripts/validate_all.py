import os
import sys
from datetime import datetime

sys.path.insert(0, os.getcwd())

from sqlalchemy import text

from src.config.settings import settings
from src.db.supabase import get_engine
from src.retrieval.qdrant_client import get_qdrant_client
from src.retrieval.retriever import search_suggestions, search_web_suggestions
from src.retrieval.rag import _rerank_by_web_content


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    lines = []

    lines.append("Validation summary")
    lines.append("===================")
    lines.append("")

    # DB validation
    engine = get_engine()
    with engine.connect() as conn:
        pharma_count = conn.execute(text("SELECT COUNT(*) FROM public.pharma_suggestions")).scalar()
        web_count = conn.execute(text("SELECT COUNT(*) FROM public.web_suggestions")).scalar()
        pharma_nulls = conn.execute(text(
            "SELECT "
            "SUM(CASE WHEN rep_name IS NULL OR rep_name = '' THEN 1 ELSE 0 END) AS rep_name_nulls, "
            "SUM(CASE WHEN hcp_name IS NULL OR hcp_name = '' THEN 1 ELSE 0 END) AS hcp_name_nulls, "
            "SUM(CASE WHEN suggestion_reason IS NULL OR suggestion_reason = '' THEN 1 ELSE 0 END) AS reason_nulls "
            "FROM public.pharma_suggestions"
        )).mappings().fetchone()
        web_nulls = conn.execute(text(
            "SELECT "
            "SUM(CASE WHEN url IS NULL OR url = '' THEN 1 ELSE 0 END) AS url_nulls, "
            "SUM(CASE WHEN title IS NULL OR title = '' THEN 1 ELSE 0 END) AS title_nulls, "
            "SUM(CASE WHEN document_text IS NULL OR document_text = '' THEN 1 ELSE 0 END) AS doc_nulls, "
            "SUM(CASE WHEN source_type IS NULL OR source_type = '' THEN 1 ELSE 0 END) AS source_type_nulls "
            "FROM public.web_suggestions"
        )).mappings().fetchone()
        web_source_types = conn.execute(text(
            "SELECT COALESCE(source_type,'(null)') AS source_type, COUNT(*) AS cnt "
            "FROM public.web_suggestions GROUP BY source_type ORDER BY cnt DESC"
        )).fetchall()

    lines.append("DB post-ingest checks")
    lines.append("---------------------")
    lines.append(f"pharma_suggestions count: {pharma_count}")
    lines.append(f"web_suggestions count: {web_count}")
    lines.append(f"pharma nulls: {dict(pharma_nulls)}")
    lines.append(f"web nulls: {dict(web_nulls)}")
    lines.append("web source_type distribution:")
    for row in web_source_types:
        lines.append(f"- {row[0]} {row[1]}")
    lines.append("")

    # Qdrant validation
    client = get_qdrant_client()
    collection = settings.QDRANT__COLLECTION

    def _count_by_record_type(record_type: str) -> int:
        total = 0
        next_page = None
        while True:
            res = client.scroll(
                collection_name=collection,
                scroll_filter={"must": [{"key": "record_type", "match": {"value": record_type}}]},
                with_payload=False,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            points, next_page = res
            total += len(points)
            if not next_page:
                break
        return total

    pharma_vectors = _count_by_record_type("pharma")
    web_vectors = _count_by_record_type("web")
    sample = client.scroll(collection_name=collection, limit=1, with_payload=True, with_vectors=False)
    sample_payload = sample[0][0].payload if sample[0] else {}

    lines.append("Qdrant checks")
    lines.append("-------------")
    lines.append(f"collection: {collection}")
    lines.append(f"pharma vectors: {pharma_vectors}")
    lines.append(f"web vectors: {web_vectors}")
    lines.append(f"sample payload keys: {list(sample_payload.keys())}")
    lines.append("")

    # Retrieval validation
    lines.append("Retrieval checks")
    lines.append("----------------")
    pharma_queries = [
        "neurology outreach",
        "HCP neurology Ocrevus",
    ]
    web_queries = [
        ("ocrelizumab multiple sclerosis", "rss_article", 0.0),
        ("for OCREVUS give the resuls of Liver Function Tests", "pdf", 0.5),
    ]

    for pharma_q in pharma_queries:
        pharma_results = search_suggestions(query=pharma_q, limit=5, score_threshold=0.0)
        lines.append(f"Pharma query: {pharma_q}")
        lines.append(f"Count: {len(pharma_results)}")
        for r in pharma_results[:5]:
            lines.append(f"- {r.get('rep_name')} -> {r.get('hcp_name')} ({r.get('hcp_speciality')})")
        lines.append("")

    for web_q, source_type, threshold in web_queries:
        web_results = search_web_suggestions(
            query=web_q,
            source_type=source_type,
            limit=5,
            score_threshold=threshold,
        )
        lines.append(f"Web query: {web_q}")
        lines.append(f"Source type: {source_type}")
        lines.append(f"Score threshold: {threshold}")
        lines.append(f"Count: {len(web_results)}")
        for r in web_results[:5]:
            snippet = r.get("document_text") or r.get("content_text") or r.get("summary") or ""
            snippet = str(snippet).replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(
                f"- score={r.get('similarity_score')} | {r.get('published_at')} | {r.get('source')} | "
                f"{r.get('title')} | chunk {r.get('chunk_index')}/{r.get('chunk_count')}: {snippet}"
            )
        lines.append("")

    # Web rerank evaluation
    lines.append("Web rerank evaluation")
    lines.append("----------------------")
    query = "OCREVUS liver function tests"
    source_type = "pdf"
    base = search_web_suggestions(query=query, source_type=source_type, limit=8, score_threshold=0.0)
    reranked = _rerank_by_web_content(query, list(base), debug=False)
    lines.append(f"Query: {query}")
    lines.append("Before rerank:")
    for r in base[:5]:
        lines.append(
            f"- score={r.get('similarity_score')} | chunk {r.get('chunk_index')}/{r.get('chunk_count')} | {r.get('title')}"
        )
    lines.append("After rerank:")
    for r in reranked[:5]:
        lines.append(
            f"- hybrid={r.get('content_hybrid_score')} | score={r.get('similarity_score')} | "
            f"chunk {r.get('chunk_index')}/{r.get('chunk_count')} | {r.get('title')}"
        )

    report_path = f"logs/validation_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
