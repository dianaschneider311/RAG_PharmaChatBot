"""Semantic search retriever for pharma suggestions using Qdrant + SQL filtering."""

from typing import Optional, Dict, Any, List
from datetime import date

from src.config.settings import settings
from src.db.supabase import get_engine
from src.retrieval.qdrant_client import get_qdrant_client
from sqlalchemy import text


def search_suggestions(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.5,
    rep_name: Optional[str] = None,
    hcp_name: Optional[str] = None,
    hcp_speciality: Optional[str] = None,
    suggested_channel: Optional[str] = None,
    product_switch_to: Optional[str] = None,
    product_switch_from: Optional[str] = None,
    adjusted_expected_value: Optional[str] = None,
    aktana_hcp_value: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    source: Optional[str] = None,
    record_type: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search with optional SQL filtering.

    Args:
        query: Natural language query to search for.
        limit: Maximum number of results to return.
        score_threshold: Minimum similarity score (0-1) for results.
        rep_name: Optional filter by rep name (exact match).
        hcp_name: Optional filter by HCP name (exact match).
        hcp_speciality: Optional filter by HCP speciality (single string or list).
        suggested_channel: Optional filter by channel (SEND, VISIT).
        product_switch_to: Optional filter by product being switched to.

    Returns:
        List of suggestion dictionaries with full details and similarity scores.
    """
    from src.retrieval.embed_and_upsert import get_embeddings

    # Get Qdrant client
    qdrant_client = get_qdrant_client()
    collection_name = settings.QDRANT__COLLECTION

    # Generate embedding for query
    query_embedding = get_embeddings([query])[0]

    # Build filter conditions for Qdrant
    filter_conditions = None
    should_conditions = []
    filter_keys = set()
    # Normalize channel values to match stored data
    if suggested_channel:
        def _norm_channel(v: str) -> str:
            s = str(v).strip().upper()
            if s in {"VISIT", "VISITS", "IN PERSON", "IN-PERSON", "MEETING"}:
                return "VISIT"
            if s in {"SEND", "SENDS", "EMAIL", "E-MAIL", "MAIL", "SMS", "TEXT", "CALL", "PHONE"}:
                return "SEND"
            return s

        if isinstance(suggested_channel, (list, tuple)):
            suggested_channel = [_norm_channel(v) for v in suggested_channel]
        else:
            suggested_channel = _norm_channel(suggested_channel)
    if rep_name or hcp_name or hcp_speciality or suggested_channel or product_switch_to or product_switch_from or adjusted_expected_value or aktana_hcp_value or date_from or date_to or source or record_type:
        from qdrant_client.http import models

        def _cond(key: str, value: str):
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))

        must_conditions = []
        if rep_name:
            filter_keys.add("rep_name")
            if isinstance(rep_name, (list, tuple)):
                should_conditions.extend([_cond("rep_name", v) for v in rep_name])
            else:
                must_conditions.append(_cond("rep_name", rep_name))
        if hcp_name:
            filter_keys.add("hcp_name")
            if isinstance(hcp_name, (list, tuple)):
                should_conditions.extend([_cond("hcp_name", v) for v in hcp_name])
            else:
                must_conditions.append(_cond("hcp_name", hcp_name))
        # Support filtering by HCP speciality (match any of an array)
        if hcp_speciality:
            filter_keys.add("hcp_speciality")
            # accept either a single string or a list of strings
            if isinstance(hcp_speciality, (list, tuple)):
                # build should conditions (match any)
                should_conditions.extend([_cond("hcp_speciality", s) for s in hcp_speciality])
            else:
                should_conditions.append(_cond("hcp_speciality", hcp_speciality))
        if suggested_channel:
            filter_keys.add("suggested_channel")
            if isinstance(suggested_channel, (list, tuple)):
                should_conditions.extend([_cond("suggested_channel", v) for v in suggested_channel])
            else:
                must_conditions.append(_cond("suggested_channel", suggested_channel))
        if product_switch_to:
            filter_keys.add("product_switch_to")
            if isinstance(product_switch_to, (list, tuple)):
                should_conditions.extend([_cond("product_switch_to", v) for v in product_switch_to])
            else:
                must_conditions.append(_cond("product_switch_to", product_switch_to))
        if product_switch_from:
            filter_keys.add("product_switch_from")
            if isinstance(product_switch_from, (list, tuple)):
                should_conditions.extend([_cond("product_switch_from", v) for v in product_switch_from])
            else:
                must_conditions.append(_cond("product_switch_from", product_switch_from))
        if adjusted_expected_value:
            filter_keys.add("adjusted_expected_value")
            if isinstance(adjusted_expected_value, (list, tuple)):
                should_conditions.extend([_cond("adjusted_expected_value", v) for v in adjusted_expected_value])
            else:
                must_conditions.append(_cond("adjusted_expected_value", adjusted_expected_value))
        if aktana_hcp_value:
            filter_keys.add("aktana_hcp_value")
            if isinstance(aktana_hcp_value, (list, tuple)):
                should_conditions.extend([_cond("aktana_hcp_value", v) for v in aktana_hcp_value])
            else:
                must_conditions.append(_cond("aktana_hcp_value", aktana_hcp_value))
        if date_from or date_to:
            filter_keys.add("suggestion_date")
            df = date.fromisoformat(date_from) if date_from else None
            dt = date.fromisoformat(date_to) if date_to else None
            must_conditions.append(
                models.FieldCondition(
                    key="suggestion_date",
                    range=models.DatetimeRange(
                        gte=df,
                        lte=dt,
                    ),
                )
            )
        if source:
            filter_keys.add("source")
            if isinstance(source, (list, tuple)):
                should_conditions.extend([_cond("source", v) for v in source])
            else:
                must_conditions.append(_cond("source", source))
        if record_type:
            filter_keys.add("record_type")
            must_conditions.append(_cond("record_type", record_type))

        if must_conditions or should_conditions:
            from qdrant_client.http import models

            filter_kwargs = {}
            if must_conditions:
                filter_kwargs["must"] = must_conditions
            if should_conditions:
                filter_kwargs["should"] = should_conditions
            filter_conditions = models.Filter(**filter_kwargs)

    # Ensure payload indexes for filterable keys (required by Qdrant for keyword filters)
    if filter_keys:
        from qdrant_client.http import models
        for key in sorted(filter_keys):
            try:
                if key in {"suggestion_date", "published_at"}:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=key,
                        field_schema=models.PayloadSchemaType.DATETIME,
                    )
                else:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=key,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            except Exception:
                # Index already exists or server rejects duplicate creation
                pass

    # Perform semantic search in Qdrant (filters are applied server-side)
    if hasattr(qdrant_client, "search"):
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold,
        )
    else:
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold,
        ).points

    if debug:
        print(f"[retriever] qdrant returned {len(search_result)} points")
        if search_result:
            try:
                print("[retriever] sample payload keys:", list(search_result[0].payload.keys()))
            except Exception:
                print("[retriever] sample payload unavailable")
            try:
                print("[retriever] sample scores:", [getattr(p, "score", None) for p in search_result[:5]])
            except Exception:
                print("[retriever] sample scores unavailable")
            try:
                sample_clean = search_result[0].payload.get("suggestion_reason_clean")
                if sample_clean:
                    print("[retriever] sample reason_clean preview:", str(sample_clean)[:300])
            except Exception:
                pass

    # Build results with full suggestion details from DB
    db_engine = get_engine()
    results = []

    for scored_point in search_result:
        event_id = scored_point.payload.get("event_id")
        payload_reason_clean = scored_point.payload.get("suggestion_reason_clean")

        # Fetch full details from database
        with db_engine.connect() as conn:
            # Inspect available columns and build a safe SELECT/WHERE
            col_rows = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='pharma_suggestions'"
                )
            ).fetchall()
            existing_cols = {r[0] for r in col_rows}

            select_fields = []
            # Choose id expression
            if 'event_id' in existing_cols and 'content_hash' in existing_cols:
                select_fields.append('COALESCE(event_id, content_hash) AS event_id')
                where_clause = '(event_id = :event_id OR content_hash = :event_id)'
            elif 'event_id' in existing_cols:
                select_fields.append('event_id')
                where_clause = 'event_id = :event_id'
            elif 'content_hash' in existing_cols:
                select_fields.append('content_hash AS event_id')
                where_clause = 'content_hash = :event_id'
            else:
                # no id columns exist; cannot fetch by id
                row = None
                where_clause = None

            # other expected fields (only add if present)
            other_fields = [
                'suggestion_date',
                'rep_name',
                'hcp_name',
                'hcp_speciality',
                'facility_address',
                'suggested_channel',
                'suggestion_reason',
                'suggestion_reason_clean',
                'product_switch_from',
                'product_switch_to',
                'new_prescriber',
                'repeat_prescriber',
                'adjusted_expected_value',
                'aktana_hcp_value',
                'hcp_priority_trend_for_ocrevus',
                'is_visit_channel_propensity_high',
                'is_suggestion_recommended',
                'source',
            ]
            for f in other_fields:
                if f in existing_cols:
                    select_fields.append(f)

            if where_clause:
                query = f"SELECT {', '.join(select_fields)} FROM public.pharma_suggestions WHERE {where_clause}"
                row = conn.execute(text(query), {"event_id": event_id}).mappings().fetchone()

        if row:
            result = {
                "similarity_score": scored_point.score,
                "event_id": row.get("event_id"),
                "suggestion_date": row.get("suggestion_date"),
                "rep_name": row.get("rep_name"),
                "hcp_name": row.get("hcp_name"),
                "hcp_speciality": row.get("hcp_speciality"),
                "facility_address": row.get("facility_address"),
                "suggested_channel": row.get("suggested_channel"),
                "suggestion_reason": row.get("suggestion_reason"),
                "suggestion_reason_clean": row.get("suggestion_reason_clean") or payload_reason_clean,
                "product_switch_from": row.get("product_switch_from"),
                "product_switch_to": row.get("product_switch_to"),
                "new_prescriber": row.get("new_prescriber"),
                "repeat_prescriber": row.get("repeat_prescriber"),
                "adjusted_expected_value": row.get("adjusted_expected_value"),
                "aktana_hcp_value": row.get("aktana_hcp_value"),
                "hcp_priority_trend_for_ocrevus": row.get("hcp_priority_trend_for_ocrevus"),
                "is_visit_channel_propensity_high": row.get("is_visit_channel_propensity_high"),
                "is_suggestion_recommended": row.get("is_suggestion_recommended"),
                "source": row.get("source"),
            }
            results.append(result)

    if debug:
        top = results[:3]
        for i, r in enumerate(top, start=1):
            reason_clean = r.get("suggestion_reason_clean")
            reason_preview = str(reason_clean)[:300] if reason_clean else None
            print(f"[retriever] top {i} score={r.get('similarity_score')}")
            print(f"[retriever] top {i} reason_clean preview: {reason_preview}")

        # Explicitly report rank for a specific HCP only if needed (removed Zadon-specific debug)

    return results


def search_best_match(
    query: str,
    top_k: int = 5,
    rep_name: Optional[str] = None,
    hcp_name: Optional[str] = None,
    hcp_speciality: Optional[str] = None,
    suggested_channel: Optional[str] = None,
    product_switch_to: Optional[str] = None,
    product_switch_from: Optional[str] = None,
    adjusted_expected_value: Optional[str] = None,
    aktana_hcp_value: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    source: Optional[str] = None,
    record_type: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return the closest semantic matches within the given filters (no score threshold).
    """
    return search_suggestions(
        query=query,
        limit=top_k,
        score_threshold=0.0,
        rep_name=rep_name,
        hcp_name=hcp_name,
        hcp_speciality=hcp_speciality,
        suggested_channel=suggested_channel,
        product_switch_to=product_switch_to,
        product_switch_from=product_switch_from,
        adjusted_expected_value=adjusted_expected_value,
        aktana_hcp_value=aktana_hcp_value,
        date_from=date_from,
        date_to=date_to,
        source=source,
        record_type=record_type,
        debug=debug,
    )


def search_by_filters(
    rep_name: Optional[str] = None,
    hcp_name: Optional[str] = None,
    hcp_speciality: Optional[str] = None,
    suggested_channel: Optional[str] = None,
    product_switch_to: Optional[str] = None,
    adjusted_expected_value: Optional[str] = None,
    aktana_hcp_value: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Perform pure SQL filtering (no semantic search).

    Args:
        rep_name: Filter by rep name.
        hcp_name: Filter by HCP name.
        hcp_speciality: Filter by HCP speciality (single string or list).
        suggested_channel: Filter by channel (SEND, VISIT).
        product_switch_to: Filter by product.
        limit: Maximum number of results.

    Returns:
        List of matching suggestions.
    """
    db_engine = get_engine()

    # Detect whether hcp_speciality is stored as an array or text
    hcp_speciality_is_array = False
    with db_engine.connect() as conn:
        col_info = conn.execute(
            text(
                "SELECT data_type, udt_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='pharma_suggestions' "
                "AND column_name='hcp_speciality'"
            )
        ).fetchone()
        if col_info:
            data_type, udt_name = col_info
            hcp_speciality_is_array = (data_type == "ARRAY") or (udt_name or "").startswith("_")

    # Build WHERE clause dynamically
    where_clauses = []
    params = {}

    if rep_name:
        where_clauses.append("rep_name = :rep_name")
        params["rep_name"] = rep_name

    if hcp_name:
        where_clauses.append("hcp_name = :hcp_name")
        params["hcp_name"] = hcp_name

    # HCP speciality can be a single value or a list. Handle both text and array columns.
    if hcp_speciality:
        if isinstance(hcp_speciality, (list, tuple)):
            if hcp_speciality_is_array:
                where_clauses.append("hcp_speciality && :hcp_specialities")
                params["hcp_specialities"] = list(hcp_speciality)
            else:
                where_clauses.append("hcp_speciality IN :hcp_specialities")
                params["hcp_specialities"] = tuple(hcp_speciality)
        else:
            if hcp_speciality_is_array:
                where_clauses.append(":hcp_speciality = ANY(hcp_speciality)")
            else:
                where_clauses.append("hcp_speciality = :hcp_speciality")
            params["hcp_speciality"] = hcp_speciality

    if suggested_channel:
        where_clauses.append("suggested_channel = :suggested_channel")
        params["suggested_channel"] = suggested_channel

    if product_switch_to:
        where_clauses.append("product_switch_to = :product_switch_to")
        params["product_switch_to"] = product_switch_to
    if adjusted_expected_value:
        where_clauses.append("adjusted_expected_value = :adjusted_expected_value")
        params["adjusted_expected_value"] = adjusted_expected_value
    if aktana_hcp_value:
        where_clauses.append("aktana_hcp_value = :aktana_hcp_value")
        params["aktana_hcp_value"] = aktana_hcp_value

    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
        SELECT 
            event_id,
            suggestion_date,
            rep_name,
            hcp_name,
            hcp_speciality,
            facility_address,
            suggested_channel,
            suggestion_reason,
            product_switch_from,
            product_switch_to,
            adjusted_expected_value,
            aktana_hcp_value,
            is_suggestion_recommended
        FROM public.pharma_suggestions
        WHERE {where_clause}
        ORDER BY suggestion_date DESC
        LIMIT :limit
    """

    params["limit"] = limit

    with db_engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    results = [
        {
            "event_id": row[0],
            "suggestion_date": row[1],
            "rep_name": row[2],
            "hcp_name": row[3],
            "hcp_speciality": row[4],
            "facility_address": row[5],
            "suggested_channel": row[6],
            "suggestion_reason": row[7],
            "product_switch_from": row[8],
            "product_switch_to": row[9],
            "adjusted_expected_value": row[10],
            "aktana_hcp_value": row[11],
            "is_suggestion_recommended": row[12],
        }
        for row in rows
    ]

    return results


def search_web_suggestions(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.5,
    source_type: Optional[str] = None,
    source: Optional[str] = None,
    domain: Optional[str] = None,
    content_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    published_from: Optional[str] = None,
    published_to: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search over web/RSS content with optional filters.
    """
    from src.retrieval.embed_and_upsert import get_embeddings

    qdrant_client = get_qdrant_client()
    collection_name = settings.QDRANT__COLLECTION

    query_embedding = get_embeddings([query])[0]

    filter_conditions = None
    should_conditions = []
    filter_keys = set()

    from qdrant_client.http import models

    def _cond(key: str, value: str):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    must_conditions = []

    # Ensure we only hit web records
    filter_keys.add("record_type")
    must_conditions.append(_cond("record_type", "web"))

    if source_type:
        filter_keys.add("source_type")
        if isinstance(source_type, (list, tuple)):
            should_conditions.extend([_cond("source_type", v) for v in source_type])
        else:
            must_conditions.append(_cond("source_type", source_type))
    if source:
        filter_keys.add("source")
        if isinstance(source, (list, tuple)):
            should_conditions.extend([_cond("source", v) for v in source])
        else:
            must_conditions.append(_cond("source", source))
    if domain:
        filter_keys.add("domain")
        if isinstance(domain, (list, tuple)):
            should_conditions.extend([_cond("domain", v) for v in domain])
        else:
            must_conditions.append(_cond("domain", domain))
    if content_type:
        filter_keys.add("content_type")
        if isinstance(content_type, (list, tuple)):
            should_conditions.extend([_cond("content_type", v) for v in content_type])
        else:
            must_conditions.append(_cond("content_type", content_type))
    if tags:
        filter_keys.add("tags")
        if isinstance(tags, (list, tuple)):
            should_conditions.extend([_cond("tags", t) for t in tags])
        else:
            should_conditions.append(_cond("tags", str(tags)))
    if published_from or published_to:
        filter_keys.add("published_at")
        df = date.fromisoformat(published_from) if published_from else None
        dt = date.fromisoformat(published_to) if published_to else None
        must_conditions.append(
            models.FieldCondition(
                key="published_at",
                range=models.DatetimeRange(
                    gte=df,
                    lte=dt,
                ),
            )
        )

    if must_conditions or should_conditions:
        filter_kwargs = {}
        if must_conditions:
            filter_kwargs["must"] = must_conditions
        if should_conditions:
            filter_kwargs["should"] = should_conditions
        filter_conditions = models.Filter(**filter_kwargs)

    # Ensure payload indexes for filterable keys
    if filter_keys:
        for key in sorted(filter_keys):
            try:
                if key in {"suggestion_date", "published_at"}:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=key,
                        field_schema=models.PayloadSchemaType.DATETIME,
                    )
                else:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=key,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            except Exception:
                pass

    if hasattr(qdrant_client, "search"):
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold,
        )
    else:
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold,
        ).points

    if debug:
        print(f"[retriever] web qdrant returned {len(search_result)} points")

    db_engine = get_engine()
    results = []

    for scored_point in search_result:
        event_id = scored_point.payload.get("event_id")
        with db_engine.connect() as conn:
            col_rows = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='web_suggestions'"
                )
            ).fetchall()
            existing_cols = {r[0] for r in col_rows}

            select_fields = []
            if 'event_id' in existing_cols and 'content_hash' in existing_cols:
                select_fields.append('COALESCE(event_id, content_hash) AS event_id')
                where_clause = '(event_id = :event_id OR content_hash = :event_id)'
            elif 'event_id' in existing_cols:
                select_fields.append('event_id')
                where_clause = 'event_id = :event_id'
            elif 'content_hash' in existing_cols:
                select_fields.append('content_hash AS event_id')
                where_clause = 'content_hash = :event_id'
            else:
                row = None
                where_clause = None

            other_fields = [
                'published_at',
                'source',
                'domain',
                'url',
                'title',
                'author',
                'summary',
                'content_text',
                'document_text',
                'tags',
                'parent_event_id',
                'chunk_index',
                'chunk_count',
                'therapeutic_area',
                'drug_name',
                'company',
                'content_type',
            ]
            for f in other_fields:
                if f in existing_cols:
                    select_fields.append(f)

            if where_clause:
                query_sql = f"SELECT {', '.join(select_fields)} FROM public.web_suggestions WHERE {where_clause}"
                row = conn.execute(text(query_sql), {"event_id": event_id}).mappings().fetchone()

        if row:
            result = {
                "similarity_score": scored_point.score,
                "event_id": row.get("event_id"),
                "published_at": row.get("published_at"),
                "source": row.get("source"),
                "domain": row.get("domain"),
                "url": row.get("url"),
                "title": row.get("title"),
                "author": row.get("author"),
                "summary": row.get("summary"),
                "content_text": row.get("content_text"),
                "document_text": row.get("document_text"),
                "tags": row.get("tags"),
                "parent_event_id": row.get("parent_event_id"),
                "chunk_index": row.get("chunk_index"),
                "chunk_count": row.get("chunk_count"),
                "therapeutic_area": row.get("therapeutic_area"),
                "drug_name": row.get("drug_name"),
                "company": row.get("company"),
                "content_type": row.get("content_type"),
            }
            results.append(result)

    return results


def search_web_best_match(
    query: str,
    top_k: int = 5,
    source_type: Optional[str] = None,
    source: Optional[str] = None,
    domain: Optional[str] = None,
    content_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    published_from: Optional[str] = None,
    published_to: Optional[str] = None,
    score_threshold: float = 0.0,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    return search_web_suggestions(
        query=query,
        limit=top_k,
        score_threshold=score_threshold,
        source_type=source_type,
        source=source,
        domain=domain,
        content_type=content_type,
        tags=tags,
        published_from=published_from,
        published_to=published_to,
        debug=debug,
    )


if __name__ == "__main__":
    # Example: semantic search
    results = search_suggestions(
        query="How should we approach high-value HCPs for Ocrevus?",
        limit=5,
    )

    print(f"\nFound {len(results)} results:")
    for result in results:
        print(
            f"  - {result['rep_name']} → {result['hcp_name']}: {result['suggested_channel']} ({result['similarity_score']:.3f})"
        )
