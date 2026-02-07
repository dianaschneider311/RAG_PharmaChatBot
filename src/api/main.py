from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config.settings import settings
from src.retrieval.rag import answer_with_router


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("api")
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))

limiter = Limiter(key_func=get_remote_address)


class AskRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_context_items: int = Field(default=8, ge=1, le=20)
    include_explanation: bool = False
    include_filters: bool = False
    include_source_details: bool = False
    debug: bool = False
    # Web/RSS filters
    source_type: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    content_type: Optional[str] = None
    tags: Optional[List[str]] = None
    published_from: Optional[str] = None
    published_to: Optional[str] = None
    # Rerank
    rerank_by_reason: bool = False
    rerank_pool_multiplier: int = 5


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    route_used: Optional[str] = None


def _log_request(payload: Dict[str, Any]) -> None:
    path = os.path.join(LOG_DIR, "api_requests.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _verify_api_key(request: Request) -> None:
    api_key = getattr(settings, "API_KEY", None)
    if not api_key:
        return
    provided = request.headers.get("x-api-key")
    if provided != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="Pharma RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(getattr(settings, "API_RATE_LIMIT", "20/minute"))
async def ask(request: Request, _: None = Depends(_verify_api_key)) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        req = AskRequest.model_validate(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    started = time.time()
    resp = answer_with_router(
        query=req.query,
        limit=req.limit,
        score_threshold=req.score_threshold,
        max_context_items=req.max_context_items,
        rerank_by_reason=req.rerank_by_reason,
        rerank_pool_multiplier=req.rerank_pool_multiplier,
        include_explanation=req.include_explanation,
        include_filters=req.include_filters,
        include_source_details=req.include_source_details,
        source_type=req.source_type,
        source=req.source,
        domain=req.domain,
        content_type=req.content_type,
        tags=req.tags,
        published_from=req.published_from,
        published_to=req.published_to,
        debug=req.debug,
    )

    duration_ms = int((time.time() - started) * 1000)
    _log_request(
        {
            "request_id": str(uuid.uuid4()),
            "query": req.query,
            "route": resp.route_used,
            "limit": req.limit,
            "score_threshold": req.score_threshold,
            "duration_ms": duration_ms,
            "citations": resp.citations,
        }
    )

    return JSONResponse(
        status_code=200,
        content=AskResponse(
            answer=resp.answer,
            citations=resp.citations,
            route_used=resp.route_used,
        ).model_dump(),
    )
