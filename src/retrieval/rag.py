"""RAG orchestration helpers (LLM filter parsing + retrieval + answering)."""

from __future__ import annotations

import json
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text

from src.config.settings import settings
from src.db.supabase import get_engine
from src.retrieval.retriever import (
    search_suggestions,
    search_best_match,
    search_web_best_match,
    search_web_suggestions,
)


class FilterSpec(BaseModel):
    """Structured filters extracted from user query."""

    rep_name: Optional[str] = None
    hcp_name: Optional[str] = None
    hcp_speciality: Optional[List[str]] = None
    suggested_channel: Optional[str] = None
    product_switch_to: Optional[str] = None
    product_switch_from: Optional[str] = None
    adjusted_expected_value: Optional[str] = None
    aktana_hcp_value: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=50)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unique_hcps: Optional[bool] = None
    output_fields: Optional[List[str]] = None
    date_from: Optional[str] = None  # ISO date
    date_to: Optional[str] = None    # ISO date


class ChatResponse(BaseModel):
    """Structured RAG response."""

    answer: str
    citations: List[str] = Field(default_factory=list)
    used_filters: Optional[FilterSpec] = None
    route_used: Optional[str] = None


def _call_llm_for_filters(query: str) -> FilterSpec:
    """Call LLM to parse filters as strict JSON and validate with Pydantic."""
    import openai

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )

    system = (
        "You are a strict parser that extracts search filters for a pharma REP chatbot. "
        "Return ONLY valid JSON with the exact keys: "
        "rep_name, hcp_name, hcp_speciality, suggested_channel, product_switch_to, "
        "product_switch_from, adjusted_expected_value, aktana_hcp_value, limit, "
        "score_threshold, unique_hcps, output_fields, "
        "date_from, date_to. "
        "Use null if unknown. Use a list of strings for hcp_speciality. "
        "For suggested_channel, ONLY return one of: SEND, VISIT, or null. "
        "For adjusted_expected_value or aktana_hcp_value, ONLY return one of: "
        "High, Medium, Low, Very Low, or null. "
        "For output_fields, ONLY use these values: hcp_name, facility_address, "
        "hcp_speciality, product_switch_from, product_switch_to, reason_summary, "
        "similarity_score, rerank_score, aktana_hcp_value, adjusted_expected_value. "
        "If the user asks for unique HCPs, set unique_hcps to true. "
        "For date ranges, use ISO format YYYY-MM-DD in date_from/date_to. "
        "Never add extra keys or text."
    )

    user = (
        "Query: " + query + "\n"
        "Return JSON only."
    )

    last_error: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content or ""
            data = json.loads(content)
            # Coerce common bad types before validation
            if isinstance(data, dict) and isinstance(data.get("score_threshold"), str):
                try:
                    data["score_threshold"] = float(data["score_threshold"])
                except ValueError:
                    data["score_threshold"] = None
            return FilterSpec.model_validate(data)
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            last_error = e
    raise RuntimeError(f"Failed to parse filters from LLM output: {last_error}")


def _call_llm_for_route(query: str) -> str:
    """Call LLM to classify query route: pharma or web."""
    import openai

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )

    system = (
        "You are a router for a pharma assistant. "
        "Return ONLY valid JSON with a single key: route. "
        "The route must be one of: pharma, web. "
        "Use pharma for internal rep/HCP suggestion questions. "
        "Use web for external news, publications, trials, FDA/EMA, press releases."
    )
    user = f"Query: {query}\nReturn JSON only."

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    data = json.loads(content)
    route = str(data.get("route") or "").strip().lower()
    return route if route in {"pharma", "web"} else "web"


def _route_query_rule(query: str) -> str:
    """Rule-based router: pharma, web, or ambiguous."""
    q = (query or "").lower()
    pharma_terms = {
        "hcp", "rep", "outreach", "suggestion", "suggested", "channel",
        "visit", "send", "aktana", "expected value", "product switch",
        "new prescriber", "repeat prescriber", "specialty", "speciality",
    }
    web_terms = {
        "news", "publication", "study", "trial", "fda", "ema",
        "press release", "label", "safety communication", "journal",
        "clinicaltrials", "pubmed",
        "prescribing information", "labeling", "guideline", "guidelines",
        "monitoring", "contraindication", "contraindications",
        "dose", "prior to", "assessment", "assessments",
    }
    pharma_hit = any(t in q for t in pharma_terms)
    web_hit = any(t in q for t in web_terms)

    if pharma_hit and not web_hit:
        return "pharma"
    if web_hit and not pharma_hit:
        return "web"
    return "ambiguous"


def route_query(query: str, debug: bool = False) -> str:
    """Hybrid router: rules first, LLM only if ambiguous."""
    rule = _route_query_rule(query)
    if debug:
        print("[router] rule:", rule)
    if rule != "ambiguous":
        return rule
    route = _call_llm_for_route(query)
    if debug:
        print("[router] llm:", route)
    return route


def _parse_filters(
    query: str,
    limit: int,
    score_threshold: float,
    debug: bool,
) -> FilterSpec:
    filters = _call_llm_for_filters(query)
    # Normalize HCP names: strip titles and credentials
    if filters.hcp_name:
        filters.hcp_name = _normalize_hcp_name(filters.hcp_name)
    # Normalize speciality values (match stored uppercase list)
    if filters.hcp_speciality:
        filters.hcp_speciality = _expand_specialities(filters.hcp_speciality)
    # Normalize suggested_channel via DB synonym map (if present)
    if filters.suggested_channel:
        normalized = _normalize_channel(filters.suggested_channel)
        filters.suggested_channel = normalized
    # Enforce allowed channel enum if LLM returns something unexpected
    if filters.suggested_channel not in {None, "SEND", "VISIT"}:
        filters.suggested_channel = None
    # Normalize date range if only one side provided
    if filters.date_from and not filters.date_to:
        filters.date_to = filters.date_from
    if filters.date_to and not filters.date_from:
        filters.date_from = filters.date_to
    # Normalize value enums
    filters.aktana_hcp_value = _normalize_value_enum(filters.aktana_hcp_value)
    filters.adjusted_expected_value = _normalize_value_enum(filters.adjusted_expected_value)
    # Fall back to provided defaults when LLM didn't specify.
    filters.limit = filters.limit if filters.limit is not None else limit
    # Coerce bad score_threshold values to None
    if isinstance(filters.score_threshold, str):
        try:
            filters.score_threshold = float(filters.score_threshold)
        except ValueError:
            filters.score_threshold = None
    filters.score_threshold = (
        filters.score_threshold
        if filters.score_threshold is not None
        else score_threshold
    )
    if debug:
        print("[rag] parsed filters:", filters.model_dump())
    return filters


def search_with_llm_filters(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.5,
    debug: bool = False,
):
    """
    Parse filters with an LLM (JSON schema) and run semantic search.
    """
    filters = _parse_filters(query=query, limit=limit, score_threshold=score_threshold, debug=debug)

    return search_suggestions(
        query=query,
        limit=filters.limit,
        score_threshold=filters.score_threshold,
        rep_name=filters.rep_name,
        hcp_name=filters.hcp_name,
        hcp_speciality=filters.hcp_speciality,
        suggested_channel=filters.suggested_channel,
        product_switch_to=filters.product_switch_to,
        product_switch_from=filters.product_switch_from,
        adjusted_expected_value=filters.adjusted_expected_value,
        aktana_hcp_value=filters.aktana_hcp_value,
        date_from=filters.date_from,
        date_to=filters.date_to,
        record_type="pharma",
        debug=debug,
    )


def _normalize_channel(value: str | None) -> str | None:
    """Map free-text channel to controlled enum via DB synonym table."""
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None

    # Simple normalization to reduce synonym table size
    raw = raw.replace(".", "").replace("-", " ")
    if raw.endswith("s") and len(raw) > 3:
        raw = raw[:-1]

    # Fast path: already a known enum
    up = raw.upper()
    if up in {"SEND", "VISIT"}:
        return up

    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT normalized FROM public.channel_synonyms WHERE raw = :raw"),
                {"raw": raw},
            ).fetchone()
        if row:
            return str(row[0]).strip().upper()
    except Exception:
        # If DB lookup fails, fall back to None (do not filter)
        pass
    return None


def _normalize_hcp_name(value: str) -> str:
    s = str(value).strip()
    if not s:
        return s
    # Remove common titles/prefixes and credentials
    for prefix in ("dr.", "dr ", "doctor "):
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
            break
    # Remove trailing credentials
    creds = ["md", "m.d.", "do", "d.o.", "phd", "ph.d.", "np", "pa", "rn"]
    parts = [p.strip() for p in s.replace(",", " ").split()]
    cleaned = []
    for p in parts:
        if p.lower().strip(".") in {c.replace(".", "") for c in creds}:
            continue
        cleaned.append(p)
    return " ".join(cleaned).strip()


def _normalize_speciality(value: str) -> str:
    s = str(value).strip()
    if not s:
        return s
    return s.upper()


def _normalize_value_enum(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"very low", "very_low", "very-low"}:
        return "Very Low"
    if s == "high":
        return "High"
    if s in {"medium", "med"}:
        return "Medium"
    if s == "low":
        return "Low"
    return None


def _expand_specialities(values: List[str]) -> List[str]:
    """Expand specialty values to closest DB labels using substring matching."""
    cleaned = [_normalize_speciality(v) for v in values if v]
    if not cleaned:
        return cleaned
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT unnest(hcp_speciality) AS spec "
                    "FROM public.pharma_suggestions "
                    "WHERE hcp_speciality IS NOT NULL"
                )
            ).fetchall()
        db_specs = [str(r[0]).upper() for r in rows if r and r[0]]
        expanded: List[str] = []
        for v in cleaned:
            # exact
            if v in db_specs:
                expanded.append(v)
                continue
            # substring matches
            matches = [s for s in db_specs if v in s or s in v]
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(v)
        # de-dupe
        return list(dict.fromkeys(expanded))
    except Exception:
        return cleaned


def _build_context(
    results: List[Dict[str, Any]],
    max_items: int = 8,
    include_reason: bool = True,
) -> Tuple[str, List[str]]:
    """Format retrieved rows into a numbered context block with citation IDs."""
    context_lines = []
    citations = []
    for idx, r in enumerate(results[:max_items], start=1):
        cite_id = f"S{idx}"
        citations.append(cite_id)
        line = (
            f"[{cite_id}] "
            f"HCP: {r.get('hcp_name')}; "
            f"Specialty: {r.get('hcp_speciality')}; "
            f"Facility: {r.get('facility_address')}; "
            f"Rep: {r.get('rep_name')}; "
            f"Channel: {r.get('suggested_channel')}; "
            f"Product To: {r.get('product_switch_to')}"
        )
        if include_reason:
            line += f"; Reason: {r.get('suggestion_reason')}"
        context_lines.append(line)
    return "\n".join(context_lines), citations


def _build_web_context(
    results: List[Dict[str, Any]],
    max_items: int = 8,
) -> Tuple[str, List[str]]:
    """Format web/RSS rows into a numbered context block with citation IDs."""
    context_lines = []
    citations = []
    for idx, r in enumerate(results[:max_items], start=1):
        cite_id = f"W{idx}"
        citations.append(cite_id)
        doc_text = r.get("document_text") or r.get("content_text") or r.get("summary")
        if doc_text:
            doc_text = str(doc_text).strip()
        line = (
            f"[{cite_id}] "
            f"Title: {r.get('title')}; "
            f"Source: {r.get('source')}; "
            f"Published: {r.get('published_at')}; "
            f"URL: {r.get('url')}; "
            f"Summary: {r.get('summary')}; "
            f"Content: {doc_text}"
        )
        context_lines.append(line)
    return "\n".join(context_lines), citations


def _build_web_source_details(
    results: List[Dict[str, Any]],
    max_items: int = 8,
) -> List[str]:
    lines = []
    for idx, r in enumerate(results[:max_items], start=1):
        cite_id = f"W{idx}"
        lines.append(
            f"[{cite_id}] Title: {r.get('title')}; Source: {r.get('source')}; "
            f"Author: {r.get('author')}; Event ID: {r.get('event_id')}; URL: {r.get('url')}"
        )
    return lines


def _dedupe_results(results: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    if not keys:
        return results
    best_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in results:
        k = tuple(r.get(f) for f in keys)
        if k not in best_by_key:
            best_by_key[k] = r
            continue
        existing = best_by_key[k]
        if (r.get("similarity_score") or 0) > (existing.get("similarity_score") or 0):
            best_by_key[k] = r
    # Preserve order by best score
    return sorted(best_by_key.values(), key=lambda x: x.get("similarity_score") or 0, reverse=True)


def _build_shaped_context(
    results: List[Dict[str, Any]],
    fields: List[str],
    max_items: int = 8,
) -> Tuple[str, List[str]]:
    context_lines = []
    citations = []
    for idx, r in enumerate(results[:max_items], start=1):
        cite_id = f"S{idx}"
        citations.append(cite_id)
        parts = []
        for f in fields:
            if f == "reason_summary":
                # Provide clean reason text for summarization
                parts.append(f"suggestion_reason_clean: {r.get('suggestion_reason_clean') or r.get('suggestion_reason')}")
            elif f == "rerank_score":
                parts.append(f"rerank_score: {r.get('reason_hybrid_score')}")
            else:
                parts.append(f"{f}: {r.get(f)}")
        context_lines.append(f"[{cite_id}] " + "; ".join(parts))
    return "\n".join(context_lines), citations


def answer_with_rag(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.5,
    max_context_items: int = 8,
    rerank_by_reason: bool = False,
    rerank_pool_multiplier: int = 5,
    include_explanation: bool = False,
    include_filters: bool = False,
    debug: bool = False,
) -> ChatResponse:
    """
    End-to-end RAG: parse filters, retrieve, and answer with citations.
    """
    import openai

    # 1) Retrieve
    filters = _parse_filters(query=query, limit=limit, score_threshold=score_threshold, debug=debug)
    pool_size = filters.limit
    if rerank_by_reason:
        pool_size = min(filters.limit * max(rerank_pool_multiplier, 1), 200)
    results = search_best_match(
        query=query,
        top_k=pool_size,
        rep_name=filters.rep_name,
        hcp_name=filters.hcp_name,
        hcp_speciality=filters.hcp_speciality,
        suggested_channel=filters.suggested_channel,
        product_switch_to=filters.product_switch_to,
        product_switch_from=filters.product_switch_from,
        adjusted_expected_value=filters.adjusted_expected_value,
        aktana_hcp_value=filters.aktana_hcp_value,
        date_from=filters.date_from,
        date_to=filters.date_to,
        record_type="pharma",
        debug=debug,
    )
    if debug:
        print(f"[rag] retrieved {len(results)} rows")
        if results:
            print("[rag] sample row keys:", list(results[0].keys()))
    # Optional rerank by suggestion_reason only
    if rerank_by_reason and results:
        rerank_query = _extract_reason_query(query, filters)
        if debug:
            print("[rag] rerank query:", rerank_query)
            print("[rag] rerank query (len):", len(rerank_query))
        results = _rerank_by_suggestion_reason(rerank_query, results, debug=debug)
        results = results[: filters.limit]

    # Optional shaping
    context: str
    citations: List[str]
    if filters.unique_hcps:
        # default dedupe by HCP + facility if available
        results = _dedupe_results(results, keys=["hcp_name", "facility_address"])
    if filters.output_fields:
        allowed = {
            "hcp_name",
            "facility_address",
            "hcp_speciality",
            "product_switch_from",
            "product_switch_to",
            "reason_summary",
            "similarity_score",
            "rerank_score",
            "aktana_hcp_value",
            "adjusted_expected_value",
        }
        fields = [f for f in filters.output_fields if f in allowed]
        if fields:
            context, citations = _build_shaped_context(results, fields=fields, max_items=max_context_items)
        else:
            context, citations = _build_context(results, max_items=max_context_items)
    else:
        context, citations = _build_context(results, max_items=max_context_items)
    if debug:
        print("[rag] context:\n", context if context else "(empty)")

    # 2) Generate answer
    if not context.strip():
        if include_explanation or include_filters:
            details = []
            if include_explanation:
                details.append("No matching records were found for the parsed filters.")
            if include_filters:
                details.append(f"Filters used: {filters.model_dump()}")
            extra = "\n".join(details)
            answer = "I don't have enough information based on the available context."
            if extra:
                answer = answer.rstrip() + "\n\n" + extra
        else:
            answer = "I don't have enough information based on the available context."
        return ChatResponse(
            answer=answer,
            citations=citations,
            used_filters=filters,
            route_used="pharma",
        )

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )

    system = (
        "You are a pharma REP assistant. Use ONLY the provided context. "
        "Provide concise, actionable answers. Cite sources as [S1], [S2], etc. "
        "Do not provide medical advice to patients; keep to REP outreach guidance. "
        "Do not claim missing information if the context contains relevant entries."
    )
    if filters.output_fields:
        if include_explanation or include_filters:
            system += (
                " Return the requested fields first in a short bullet list. "
                "Then add sections titled 'Why This Result' and 'Filters Used'."
            )
        else:
            system += (
                " Return ONLY the requested fields and nothing else. "
                "Use a short bullet list."
            )
        if "reason_summary" in filters.output_fields:
            system += " Summarize suggestion_reason_clean per HCP in 1 short sentence."

    filters_json = json.dumps(filters.model_dump(), ensure_ascii=True)
    user = f"User query: {query}\n\nContext:\n{context}"
    if include_explanation or include_filters:
        user += f"\n\nParsed filters (JSON): {filters_json}"

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content or ""
    return ChatResponse(answer=answer, citations=citations, used_filters=filters, route_used="pharma")


def answer_with_rag_web(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.5,
    max_context_items: int = 8,
    source_type: Optional[str] = None,
    source: Optional[str] = None,
    domain: Optional[str] = None,
    content_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    published_from: Optional[str] = None,
    published_to: Optional[str] = None,
    include_explanation: bool = False,
    include_source_details: bool = False,
    rerank_by_content: bool = False,
    debug: bool = False,
) -> ChatResponse:
    """
    End-to-end RAG for web/RSS: retrieve and answer with citations.
    """
    import openai

    results = search_web_best_match(
        query=query,
        top_k=limit,
        source_type=source_type,
        source=source,
        domain=domain,
        content_type=content_type,
        tags=tags,
        published_from=published_from,
        published_to=published_to,
        score_threshold=score_threshold,
        debug=debug,
    )

    if rerank_by_content and results:
        results = _rerank_by_web_content(query, results, debug=debug)
        results = results[:limit]

    context, citations = _build_web_context(results, max_items=max_context_items)
    if debug:
        print("[rag] web context:\n", context if context else "(empty)")

    if not context.strip():
        return ChatResponse(
            answer="I don't have enough information based on the available context.",
            citations=citations,
            used_filters=None,
            route_used="web",
        )

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )

    system = (
        "You are a pharma intelligence assistant. Use ONLY the provided context. "
        "Provide concise, factual summaries. Cite sources as [W1], [W2], etc. "
        "Do not provide medical advice to patients."
    )
    if include_explanation:
        system += " After the answer, add a short 'Why These Sources' section."

    user = f"User query: {query}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content or ""
    if include_source_details and results:
        lines = _build_web_source_details(results, max_items=max_context_items)
        if lines:
            answer = answer.rstrip() + "\n\nSources:\n" + "\n".join(lines)
    return ChatResponse(answer=answer, citations=citations, used_filters=None, route_used="web")


def _rerank_by_web_content(
    query: str,
    results: List[Dict[str, Any]],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Rerank web results using semantic similarity + lexical overlap on content."""
    from src.retrieval.embed_and_upsert import get_embeddings
    import numpy as np
    import re
    from difflib import SequenceMatcher

    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _lexical_score(q: str, t: str) -> float:
        q_tokens = set(re.findall(r"[a-zA-Z]+", q.lower()))
        t_tokens = set(re.findall(r"[a-zA-Z]+", t.lower()))
        if not q_tokens or not t_tokens:
            return 0.0
        return len(q_tokens & t_tokens) / len(q_tokens)

    q_clean = _normalize_text(query)
    if not q_clean:
        return results

    contents = []
    for r in results:
        c = r.get("document_text") or r.get("content_text") or r.get("summary") or ""
        contents.append(_normalize_text(str(c)))

    if not any(contents):
        return results

    q_vec = np.array(get_embeddings([q_clean])[0], dtype=float)
    r_vecs = np.array(get_embeddings(contents), dtype=float)

    q_norm = np.linalg.norm(q_vec) or 1.0
    r_norms = np.linalg.norm(r_vecs, axis=1)
    sims = (r_vecs @ q_vec) / (r_norms * q_norm + 1e-8)

    q_clean_lower = q_clean.lower()
    stop = {
        "the","and","for","with","that","this","have","has","most","often","is","are",
        "was","were","from","into","only","your","their","her","his","she","he","they",
        "to","of","in","on","a","an","or","as","at","by","be","been","being","it",
    }
    q_keywords = [t for t in re.findall(r"[a-zA-Z]+", q_clean_lower) if t not in stop and len(t) >= 4]
    q_kw_set = set(q_keywords)

    for r, s, content in zip(results, sims.tolist(), contents):
        lex = _lexical_score(q_clean, content)
        content_lower = content.lower()
        bonus = 0.2 if q_clean_lower and q_clean_lower in content_lower else 0.0
        ratio = SequenceMatcher(None, q_clean_lower, content_lower).ratio() if q_clean_lower and content_lower else 0.0
        ratio_bonus = 0.25 * ratio
        r_tokens = set(re.findall(r"[a-zA-Z]+", content_lower))
        kw_overlap = len(q_kw_set & r_tokens) / (len(q_kw_set) or 1)
        kw_bonus = 0.25 * kw_overlap

        r["content_similarity_score"] = float(s)
        r["content_lexical_score"] = float(lex)
        r["content_hybrid_score"] = float(0.3 * s + 0.7 * lex + bonus + ratio_bonus + kw_bonus)

    ranked = sorted(results, key=lambda x: x.get("content_hybrid_score") or 0, reverse=True)
    if debug:
        print("[rag] rerank web: top hybrid scores:", [r.get("content_hybrid_score") for r in ranked[:5]])
    return ranked

def _build_multi_context(
    pharma_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    max_items: int = 8,
) -> Tuple[str, List[str]]:
    context_lines = []
    citations = []

    # Pharma results with S* citations
    for idx, r in enumerate(pharma_results, start=1):
        if len(context_lines) >= max_items:
            break
        cite_id = f"S{idx}"
        citations.append(cite_id)
        line = (
            f"[{cite_id}] "
            f"HCP: {r.get('hcp_name')}; "
            f"Specialty: {r.get('hcp_speciality')}; "
            f"Facility: {r.get('facility_address')}; "
            f"Rep: {r.get('rep_name')}; "
            f"Channel: {r.get('suggested_channel')}; "
            f"Product To: {r.get('product_switch_to')}; "
            f"Reason: {r.get('suggestion_reason')}"
        )
        context_lines.append(line)

    # Web results with W* citations
    for idx, r in enumerate(web_results, start=1):
        if len(context_lines) >= max_items:
            break
        cite_id = f"W{idx}"
        citations.append(cite_id)
        doc_text = r.get("document_text") or r.get("content_text") or r.get("summary")
        if doc_text:
            doc_text = str(doc_text).strip()
        line = (
            f"[{cite_id}] "
            f"Title: {r.get('title')}; "
            f"Source: {r.get('source')}; "
            f"Published: {r.get('published_at')}; "
            f"URL: {r.get('url')}; "
            f"Summary: {r.get('summary')}; "
            f"Content: {doc_text}"
        )
        context_lines.append(line)

    return "\n".join(context_lines), citations


def answer_with_rag_multi(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.5,
    max_context_items: int = 8,
    source_type: Optional[str] = None,
    debug: bool = False,
) -> ChatResponse:
    """
    Two-phase fusion: retrieve pharma + web, then build a combined context.
    """
    import openai

    per_source = max(1, limit // 2)

    pharma_results = search_suggestions(
        query=query,
        limit=per_source,
        score_threshold=score_threshold,
        record_type="pharma",
        debug=debug,
    )
    web_results = search_web_suggestions(
        query=query,
        limit=per_source,
        score_threshold=score_threshold,
        source_type=source_type,
        debug=debug,
    )

    context, citations = _build_multi_context(
        pharma_results=pharma_results,
        web_results=web_results,
        max_items=max_context_items,
    )

    if not context.strip():
        return ChatResponse(
            answer="I don't have enough information based on the available context.",
            citations=citations,
            used_filters=None,
            route_used="multi",
        )

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )

    system = (
        "You are a pharma assistant. Use ONLY the provided context. "
        "Provide concise, actionable answers. Cite sources as [S1], [W1], etc. "
        "Do not provide medical advice to patients."
    )

    user = f"User query: {query}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )

    answer = resp.choices[0].message.content or ""
    return ChatResponse(answer=answer, citations=citations, used_filters=None, route_used="multi")


def answer_with_router(
    query: str,
    debug: bool = False,
    **kwargs,
) -> ChatResponse:
    """
    Hybrid router: rules first, LLM if ambiguous.
    Routes to pharma or web RAG automatically.
    """
    if kwargs.get("source_type"):
        route = "web"
    else:
        rule = _route_query_rule(query)
        if rule == "ambiguous":
            return answer_with_rag_multi(
                query=query,
                limit=kwargs.get("limit", 5),
                score_threshold=kwargs.get("score_threshold", 0.5),
                max_context_items=kwargs.get("max_context_items", 8),
                source_type=kwargs.get("source_type"),
                debug=debug,
            )
        route = rule
    if route == "pharma":
        # Strip web-only args before calling pharma path
        kwargs.pop("include_source_details", None)
        kwargs.pop("source_type", None)
        kwargs.pop("source", None)
        kwargs.pop("domain", None)
        kwargs.pop("content_type", None)
        kwargs.pop("tags", None)
        kwargs.pop("published_from", None)
        kwargs.pop("published_to", None)
        resp = answer_with_rag(query=query, debug=debug, **kwargs)
        # Optional fallback to web if explicitly requested
        if kwargs.get("allow_web_fallback"):
            if not resp.citations or "don't have enough information" in (resp.answer or "").lower():
                # Strip pharma-only args before calling web path
                kwargs.pop("rerank_by_reason", None)
                kwargs.pop("rerank_pool_multiplier", None)
                kwargs.pop("include_filters", None)
                return answer_with_rag_web(query=query, debug=debug, **kwargs)
        return resp
    # Strip pharma-only args before calling web path
    kwargs.pop("rerank_by_reason", None)
    kwargs.pop("rerank_pool_multiplier", None)
    kwargs.pop("include_filters", None)
    return answer_with_rag_web(query=query, debug=debug, **kwargs)


def _rerank_by_suggestion_reason(
    query: str,
    results: List[Dict[str, Any]],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Rerank results using semantic similarity + lexical overlap on cleaned suggestion_reason."""
    from src.retrieval.embed_and_upsert import get_embeddings
    import numpy as np
    import re
    from difflib import SequenceMatcher

    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _lexical_score(q: str, t: str) -> float:
        # simple token overlap (lowercase, alpha tokens)
        q_tokens = set(re.findall(r"[a-zA-Z]+", q.lower()))
        t_tokens = set(re.findall(r"[a-zA-Z]+", t.lower()))
        if not q_tokens or not t_tokens:
            return 0.0
        return len(q_tokens & t_tokens) / len(q_tokens)

    q_clean = _normalize_text(query)
    if not q_clean:
        if debug:
            print("[rag] rerank by reason: empty query after cleaning; skip rerank")
        return results

    reasons = [
        _normalize_text(str(r.get("suggestion_reason_clean") or r.get("suggestion_reason") or ""))
        for r in results
    ]
    if not any(reasons):
        return results

    # Build embeddings in one batch for consistent scaling
    q_vec = np.array(get_embeddings([q_clean])[0], dtype=float)
    r_vecs = np.array(get_embeddings(reasons), dtype=float)

    # Cosine similarity
    q_norm = np.linalg.norm(q_vec) or 1.0
    r_norms = np.linalg.norm(r_vecs, axis=1)
    sims = (r_vecs @ q_vec) / (r_norms * q_norm + 1e-8)

    q_lower = query.lower()
    q_clean = _normalize_text(query)
    q_clean_lower = q_clean.lower()
    stop = {
        "the","and","for","with","that","this","have","has","most","often","is","are",
        "was","were","from","into","only","your","their","her","his","she","he","they",
        "to","of","in","on","a","an","or","as","at","by","be","been","being","it",
    }
    q_keywords = [t for t in re.findall(r"[a-zA-Z]+", q_clean_lower) if t not in stop and len(t) >= 4]
    q_kw_set = set(q_keywords)
    # Key phrase anchors from query (2-4 word phrases)
    phrase_tokens = [t for t in re.findall(r"[a-zA-Z]+", q_clean_lower) if len(t) >= 4]
    phrases = []
    for n in (2, 3, 4):
        for i in range(0, max(0, len(phrase_tokens) - n + 1)):
            phrases.append(" ".join(phrase_tokens[i:i+n]))

    for r, s, reason in zip(results, sims.tolist(), reasons):
        lex = _lexical_score(q_clean, reason)
        reason_lower = reason.lower()
        # Bonus for exact substring match (after cleaning)
        bonus = 0.2 if q_clean_lower and q_clean_lower in reason_lower else 0.0
        # Fuzzy ratio bonus for near-exact matches
        ratio = SequenceMatcher(None, q_clean_lower, reason_lower).ratio() if q_clean_lower and reason_lower else 0.0
        ratio_bonus = 0.25 * ratio
        # Keyword overlap bonus for distinctive terms
        r_tokens = set(re.findall(r"[a-zA-Z]+", reason_lower))
        kw_overlap = len(q_kw_set & r_tokens) / (len(q_kw_set) or 1)
        kw_bonus = 0.25 * kw_overlap
        # Strong boost when most distinctive keywords are present
        strong_match_bonus = 0.0
        if kw_overlap >= 0.6 and len(q_kw_set) >= 4:
            strong_match_bonus = 0.5
        # Phrase anchor bonus for near-exact matching sections
        phrase_hits = 0
        for p in phrases:
            if p and p in reason_lower:
                phrase_hits += 1
        phrase_bonus = 0.0
        if phrase_hits >= 2:
            phrase_bonus = 1.5
        elif phrase_hits == 1:
            phrase_bonus = 0.5

        # Weighted hybrid score (semantic + lexical + bonuses)
        r["reason_similarity_score"] = float(s)
        r["reason_lexical_score"] = float(lex)
        r["reason_hybrid_score"] = float(0.3 * s + 0.7 * lex + bonus + ratio_bonus + kw_bonus + strong_match_bonus + phrase_bonus)

    ranked = sorted(results, key=lambda x: x.get("reason_hybrid_score") or 0, reverse=True)

    if debug:
        print("[rag] rerank by reason: top hybrid scores:", [r.get("reason_hybrid_score") for r in ranked[:5]])
        for r in ranked[:5]:
            name = r.get("hcp_name")
            score = r.get("reason_hybrid_score")
            snippet = (r.get("suggestion_reason") or "")[:160]
            print(f"[rag] top rerank: {name} score={score} snippet={snippet}")
        # Removed HCP-specific debug output
    return ranked


def _extract_reason_query(query: str, filters: FilterSpec) -> str:
    """Strip filter-like phrases to keep only the descriptive reason text."""
    q = query.strip()
    # Remove rep name mentions
    if filters.rep_name:
        q = q.replace(filters.rep_name, "")
    # Remove channel words
    if filters.suggested_channel:
        for term in ("visit", "visits", "send", "sending", "email", "emails"):
            q = q.replace(term, "")
            q = q.replace(term.capitalize(), "")
    # Drop leading instruction clause up to first period
    if "." in q:
        head, tail = q.split(".", 1)
        if "provide" in head.lower() or "rep" in head.lower():
            q = tail
    return " ".join(q.split()).strip()
