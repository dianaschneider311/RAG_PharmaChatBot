"""Generate embeddings and upsert suggestions into Qdrant vector database."""

from typing import Optional
import uuid

import numpy as np
from qdrant_client.http import models

from src.config.settings import settings
from src.db.supabase import get_engine
from src.retrieval.qdrant_client import get_qdrant_client
from sqlalchemy import text


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using OpenRouter API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (1536-dim for text-embedding-3-small).

    Raises:
        ValueError: If OpenRouter API key is missing.
    """
    import time
    import json

    if not settings.OPENROUTER__API_KEY:
        raise ValueError("OPENROUTER__API_KEY not set in .env")

    if not texts or all((t is None or str(t).strip() == "") for t in texts):
        raise ValueError("No texts provided for embedding")

    import openai

    client = openai.OpenAI(
        api_key=settings.OPENROUTER__API_KEY,
        base_url=settings.OPENROUTER__API_URL,
    )

    max_retries = 3
    backoff = 1.0
    last_resp = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )

            last_resp = response

            # Validate response
            data = getattr(response, 'data', None)
            if data and len(data) > 0 and hasattr(data[0], 'embedding'):
                return [item.embedding for item in data]

            # If we reach here, the response didn't contain embeddings
            print(f"[embed_and_upsert] Attempt {attempt}: embedding response missing data; will retry")
            try:
                # Try to show raw response if possible
                print("[embed_and_upsert] raw response:", json.dumps(response.__dict__, default=str))
            except Exception:
                pass

        except Exception as e:
            print(f"[embed_and_upsert] Attempt {attempt}: embeddings API error: {e}")

        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2

    # Exhausted retries
    raise RuntimeError(f"Embeddings request failed after {max_retries} attempts. Last response: {getattr(last_resp, '__dict__', last_resp)}")




def embed_and_upsert_suggestions() -> None:
    """
    Fetch suggestions from Postgres, generate embeddings, and upsert to Qdrant.

    Flow:
    1. Query all suggestions from public.pharma_suggestions
    2. Build text representation for each suggestion
    3. Generate embeddings in batches
    4. Upsert vectors + metadata to Qdrant collection
    """
    # Initialize clients
    db_engine = get_engine()
    qdrant_client = get_qdrant_client()
    collection_name = settings.QDRANT__COLLECTION

    # Fetch suggestions from DB
    with db_engine.connect() as conn:
        # Inspect table columns so we only reference existing columns
        col_rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='pharma_suggestions'"
            )
        ).fetchall()
        existing_cols = {r[0] for r in col_rows}

        # Build event_id expression depending on available columns
        event_id_expr_parts = []
        if 'event_id' in existing_cols:
            event_id_expr_parts.append('event_id')
        if 'content_hash' in existing_cols:
            event_id_expr_parts.append('content_hash')

        # Fallback to md5 of key columns
        md5_expr = "md5(COALESCE(suggestion_date::text,'') || '|' || COALESCE(rep_name,'') || '|' || COALESCE(hcp_name,'') || '|' || COALESCE(suggestion_reason,''))"
        # Build COALESCE expression safely
        if event_id_expr_parts:
            coalesce_expr = 'COALESCE(' + ','.join(event_id_expr_parts) + f", {md5_expr})"
        else:
            coalesce_expr = f"{md5_expr}"

        select_fields = [f"{coalesce_expr} AS event_id"]
        # other fields we expect; include only those present
        field_order = [
            'suggestion_date',
            'rep_name',
            'hcp_name',
            'hcp_speciality',
            'suggested_channel',
            'suggestion_reason',
            'suggestion_reason_clean',
            'product_switch_from',
            'product_switch_to',
            'adjusted_expected_value',
            'aktana_hcp_value',
            'hcp_priority_trend_for_ocrevus',
            'is_suggestion_recommended',
            'source',
        ]
        for f in field_order:
            if f in existing_cols:
                select_fields.append(f)
        # include content_hash if present
        if 'content_hash' in existing_cols:
            select_fields.append('content_hash')

        select_sql = 'SELECT ' + ',\n                    '.join(select_fields) + '\n                FROM public.pharma_suggestions'
        # Order by suggestion_date if present
        if 'suggestion_date' in existing_cols:
            select_sql += '\n                ORDER BY suggestion_date DESC'

        result = conn.execute(text(select_sql))
        # fetch as mappings so we can reference columns by name even if schema varies
        rows = result.mappings().fetchall()

    if not rows:
        print("No suggestions found in database.")
        return

    print(f"Fetched {len(rows)} suggestions from database.")

    # Build text representations and metadata
    texts = []
    metadata_list = []
    skipped_empty = 0

    for row_idx, row in enumerate(rows):
        # row is a Mapping; pull values safely
        event_id = row.get('event_id')
        suggestion_date = row.get('suggestion_date')
        rep_name = row.get('rep_name')
        hcp_name = row.get('hcp_name')
        hcp_speciality = row.get('hcp_speciality')
        suggested_channel = row.get('suggested_channel')
        suggestion_reason = row.get('suggestion_reason')
        suggestion_reason_clean = row.get('suggestion_reason_clean')
        product_switch_from = row.get('product_switch_from')
        product_switch_to = row.get('product_switch_to')
        adjusted_expected_value = row.get('adjusted_expected_value')
        aktana_hcp_value = row.get('aktana_hcp_value')
        hcp_priority_trend_for_ocrevus = row.get('hcp_priority_trend_for_ocrevus')
        is_suggestion_recommended = row.get('is_suggestion_recommended')

        # Build a comprehensive text for embedding from available fields
        parts = []
        if rep_name:
            parts.append(f"Rep: {rep_name}")
        if hcp_name:
            parts.append(f"HCP: {hcp_name}")
        if hcp_speciality is not None:
            parts.append(f"Specialty: {hcp_speciality}")
        if suggested_channel:
            parts.append(f"Channel: {suggested_channel}")
        if suggestion_reason or suggestion_reason_clean:
            reason_text = suggestion_reason_clean or suggestion_reason
            parts.append(f"Reason: {reason_text}")
        if product_switch_from:
            parts.append(f"Product From: {product_switch_from}")
        if product_switch_to:
            parts.append(f"Product To: {product_switch_to}")
        if adjusted_expected_value:
            parts.append(f"Expected Value: {adjusted_expected_value}")
        if aktana_hcp_value:
            parts.append(f"Aktana HCP Value: {aktana_hcp_value}")
        if hcp_priority_trend_for_ocrevus:
            parts.append(f"Priority Trend: {hcp_priority_trend_for_ocrevus}")
        if is_suggestion_recommended is not None:
            parts.append(f"Recommended: {is_suggestion_recommended}")

        doc_text = "\n".join(parts).strip()
        
        # Skip rows with completely empty text (all fields null/missing)
        if not doc_text:
            skipped_empty += 1
            continue

        texts.append(doc_text)

        # normalize hcp_speciality: could be Postgres array returned as list or string
        hcp_spec_val = hcp_speciality
        if isinstance(hcp_speciality, str):
            # try to parse simple comma-separated string
            hcp_spec_val = [s.strip() for s in hcp_speciality.split(',')] if hcp_speciality else []

        metadata = {
            "event_id": str(event_id) if event_id is not None else None,
            "suggestion_date": str(suggestion_date) if suggestion_date is not None else None,
            "rep_name": str(rep_name) if rep_name is not None else None,
            "hcp_name": str(hcp_name) if hcp_name is not None else None,
            "hcp_speciality": list(hcp_spec_val) if hcp_spec_val is not None else [],
            "suggested_channel": str(suggested_channel) if suggested_channel is not None else None,
            "product_switch_from": str(product_switch_from) if product_switch_from is not None else None,
            "product_switch_to": str(product_switch_to) if product_switch_to is not None else None,
            "adjusted_expected_value": str(adjusted_expected_value) if adjusted_expected_value is not None else None,
            "aktana_hcp_value": str(aktana_hcp_value) if aktana_hcp_value is not None else None,
            "is_suggestion_recommended": bool(is_suggestion_recommended) if is_suggestion_recommended is not None else None,
            "suggestion_reason_clean": suggestion_reason_clean if suggestion_reason_clean is not None else None,
            "source": str(row.get("source")) if row.get("source") is not None else None,
            "record_type": "pharma",
        }
        metadata_list.append(metadata)

    if skipped_empty > 0:
        print(f"Skipped {skipped_empty} rows with empty texts.")

    # Generate embeddings in batches
    batch_size = 50
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = get_embeddings(batch_texts)
        vectors.extend(batch_embeddings)
        print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    # Prepare points for Qdrant
    points = []
    for idx, (event_id_str, embedding, metadata) in enumerate(
        zip([m["event_id"] for m in metadata_list], vectors, metadata_list)
    ):
        if event_id_str:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(event_id_str)))
        else:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pharma_{idx}"))
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata,
        )
        points.append(point)

    # Ensure Qdrant collection exists
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists")
    except:
        # Collection doesn't exist, create it
        vector_size = len(vectors[0]) if vectors else 1536  # Default to OpenRouter embedding size
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"Created new Qdrant collection '{collection_name}' with vector size {vector_size}")

    # Upsert to Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print(f"âœ… Upserted {len(points)} vectors to Qdrant collection '{collection_name}'")



def embed_and_upsert_web_suggestions() -> None:
    """
    Fetch web/RSS suggestions from Postgres, generate embeddings, and upsert to Qdrant.
    """
    db_engine = get_engine()
    qdrant_client = get_qdrant_client()
    collection_name = settings.QDRANT__COLLECTION

    with db_engine.connect() as conn:
        col_rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='web_suggestions'"
            )
        ).fetchall()
        existing_cols = {r[0] for r in col_rows}

        event_id_expr_parts = []
        if 'event_id' in existing_cols:
            event_id_expr_parts.append('event_id')
        if 'content_hash' in existing_cols:
            event_id_expr_parts.append('content_hash')

        md5_expr = "md5(COALESCE(url,'') || '|' || COALESCE(title,'') || '|' || COALESCE(published_at::text,''))"
        if event_id_expr_parts:
            coalesce_expr = 'COALESCE(' + ','.join(event_id_expr_parts) + f", {md5_expr})"
        else:
            coalesce_expr = f"{md5_expr}"

        select_fields = [f"{coalesce_expr} AS event_id"]
        field_order = [
            'published_at',
            'source_type',
            'source',
            'domain',
            'url',
            'title',
            'author',
            'summary',
            'content_text',
            'document_text',
            'tags',
            'therapeutic_area',
            'drug_name',
            'company',
            'content_type',
        ]
        for f in field_order:
            if f in existing_cols:
                select_fields.append(f)
        if 'content_hash' in existing_cols:
            select_fields.append('content_hash')

        select_sql = 'SELECT ' + ',\n                    '.join(select_fields) + '\n                FROM public.web_suggestions'
        if 'published_at' in existing_cols:
            select_sql += '\n                ORDER BY published_at DESC'

        result = conn.execute(text(select_sql))
        rows = result.mappings().fetchall()

    if not rows:
        print("No web suggestions found in database.")
        return

    print(f"Fetched {len(rows)} web suggestions from database.")

    texts = []
    metadata_list = []
    skipped_empty = 0

    for row in rows:
        event_id = row.get('event_id')
        published_at = row.get('published_at')
        source = row.get('source')
        domain = row.get('domain')
        url = row.get('url')
        title = row.get('title')
        author = row.get('author')
        summary = row.get('summary')
        content_text = row.get('content_text')
        document_text = row.get('document_text')
        tags = row.get('tags')
        therapeutic_area = row.get('therapeutic_area')
        drug_name = row.get('drug_name')
        company = row.get('company')
        content_type = row.get('content_type')

        if document_text:
            doc_text = str(document_text).strip()
        else:
            parts = [p for p in [title, summary, content_text] if p]
            doc_text = "\n".join(parts).strip()

        if not doc_text:
            skipped_empty += 1
            continue

        texts.append(doc_text)

        metadata = {
            "event_id": str(event_id) if event_id is not None else None,
            "published_at": str(published_at) if published_at is not None else None,
            "source_type": str(row.get("source_type")) if row.get("source_type") is not None else None,
            "source": str(source) if source is not None else None,
            "domain": str(domain) if domain is not None else None,
            "url": str(url) if url is not None else None,
            "title": str(title) if title is not None else None,
            "author": str(author) if author is not None else None,
            "summary": str(summary) if summary is not None else None,
            "tags": list(tags) if tags is not None else [],
            "therapeutic_area": str(therapeutic_area) if therapeutic_area is not None else None,
            "drug_name": str(drug_name) if drug_name is not None else None,
            "company": str(company) if company is not None else None,
            "content_type": str(content_type) if content_type is not None else None,
            "record_type": "web",
        }
        metadata_list.append(metadata)

    if skipped_empty > 0:
        print(f"Skipped {skipped_empty} rows with empty texts.")

    batch_size = 50
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = get_embeddings(batch_texts)
        vectors.extend(batch_embeddings)
        print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    points = []
    for idx, (event_id_str, embedding, metadata) in enumerate(
        zip([m["event_id"] for m in metadata_list], vectors, metadata_list)
    ):
        if event_id_str:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(event_id_str)))
        else:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"web_{idx}"))
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata,
        )
        points.append(point)

    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists")
    except:
        vector_size = len(vectors[0]) if vectors else 1536
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"Created new Qdrant collection '{collection_name}' with vector size {vector_size}")

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print(f"Upserted {len(points)} vectors to Qdrant collection '{collection_name}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed and upsert data into Qdrant.")
    parser.add_argument("--pharma", action="store_true", help="Embed pharma_suggestions (default)")
    parser.add_argument("--web", action="store_true", help="Embed web_suggestions")
    args = parser.parse_args()

    if not args.pharma and not args.web:
        args.pharma = True

    if args.pharma:
        embed_and_upsert_suggestions()
    if args.web:
        embed_and_upsert_web_suggestions()

