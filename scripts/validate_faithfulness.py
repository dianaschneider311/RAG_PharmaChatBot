import os
import sys
from datetime import datetime
import json
import re

sys.path.insert(0, os.getcwd())

from src.retrieval.rag import (
    answer_with_router,
    route_query,
    _build_context,
    _build_web_context,
    _build_multi_context,
)
from src.retrieval.retriever import search_suggestions, search_web_suggestions
from src.config.settings import settings


def _citation_coverage(answer: str) -> dict:
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    total = len(lines)
    cited = 0
    missing = []
    for l in lines:
        if re.search(r"\[[SW]\d+\]", l):
            cited += 1
        else:
            missing.append(l)
    return {
        "total_lines": total,
        "cited_lines": cited,
        "missing_lines": missing[:5],
    }


def _llm_judge(query: str, context: str, answer: str) -> dict:
    import openai

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )
    system = (
        "You are a strict evaluator. Determine if the ANSWER is fully supported by CONTEXT. "
        "If the answer explicitly says there are no matching records due to filters "
        "(e.g., 'No matching records were found' or 'I don't have enough information'), "
        "and the context does not contain relevant records, mark grounded=true. "
        "Return JSON with keys: grounded (true/false), missing_claims (list), notes (string)."
    )
    user = (
        f"QUERY:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return JSON only."
    )

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception:
        return {"grounded": False, "missing_claims": ["Invalid judge output"], "notes": content[:200]}


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    queries = [
        "What are recent findings about ocrelizumab in multiple sclerosis?",
        "For REP John, list HCPs with neurology specialty",
    ]

    lines = []
    lines.append("Faithfulness validation")
    lines.append("========================")

    for q in queries:
        resp = answer_with_router(
            query=q,
            limit=5,
            score_threshold=0.3,
            include_explanation=True,
            include_filters=True,
        )
        coverage = _citation_coverage(resp.answer or "")

        # Rebuild context for judging
        route = route_query(q)
        context = ""
        if route == "pharma":
            results = search_suggestions(query=q, limit=5, score_threshold=0.3, record_type="pharma")
            context, _ = _build_context(results, max_items=8)
        elif route == "web":
            results = search_web_suggestions(query=q, limit=5, score_threshold=0.3)
            context, _ = _build_web_context(results, max_items=8)
        else:
            p = search_suggestions(query=q, limit=3, score_threshold=0.3, record_type="pharma")
            w = search_web_suggestions(query=q, limit=3, score_threshold=0.3)
            context, _ = _build_multi_context(p, w, max_items=8)

        judge = _llm_judge(q, context=context, answer=resp.answer or "")
        lines.append(f"Query: {q}")
        lines.append(f"Route: {resp.route_used}")
        lines.append(f"Citation coverage: {coverage}")
        lines.append(f"LLM judge: {judge}")
        lines.append("")

    report_path = f"logs/faithfulness_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
