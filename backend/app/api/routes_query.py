"""
api/routes_query.py — Query endpoints.

Endpoints:
  POST /api/v1/query    Run GraphRAG and/or VectorRAG, return dual answers.

Design:
  - mode=both runs both engines concurrently via asyncio.gather().
  - mode=graphrag or mode=vectorrag runs only the requested engine.
  - Token usage is accumulated across both engines when mode=both.
  - If the index has not been built, returns HTTP 503 with a helpful message.
  - All query parameters map directly onto the engine .query() methods.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, status

from app.dependencies import AuthDep
from app.models.request_models import QueryMode, QueryRequest
from app.models.response_models import (
    QueryResponse,
    TokenUsage,
)
from app.utils.logger import get_logger, get_request_id

log = get_logger(__name__)
router = APIRouter()


def _build_graphrag_engine():
    """Build a GraphRAGEngine from application settings."""
    from app.core.query.graphrag_engine import GraphRAGEngine
    from app.config import get_settings
    from app.services.llm_service import LLMService
    from app.services.tokenizer_service import TokenizerService
    from app.storage.summary_store import SummaryStore

    settings = get_settings()
    llm_svc = LLMService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_tokens=settings.openai_max_tokens,
        temperature=settings.openai_temperature,
    )
    tokenizer = TokenizerService(model=settings.gemini_model)
    summary_store = SummaryStore(artifacts_dir=settings.artifacts_dir)
    return GraphRAGEngine(
        openai_service=llm_svc,
        summary_store=summary_store,
        tokenizer=tokenizer,
    )


def _build_vectorrag_engine():
    """Build a VectorRAGEngine from application settings."""
    from app.core.query.vectorrag_engine import VectorRAGEngine
    from app.config import get_settings
    from app.services.llm_service import LLMService
    from app.services.embedding_service import EmbeddingService
    from app.services.faiss_service import FAISSService
    from app.services.tokenizer_service import TokenizerService

    settings = get_settings()
    llm_svc = LLMService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_tokens=settings.openai_max_tokens,
        temperature=settings.openai_temperature,
    )
    embedding_svc = EmbeddingService(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
    )
    faiss_svc = FAISSService(embedding_dim=1536)
    tokenizer = TokenizerService(model=settings.gemini_model)

    return VectorRAGEngine(
        openai_service=llm_svc,
        embedding_service=embedding_svc,
        faiss_service=faiss_svc,
        tokenizer=tokenizer,
        faiss_index_path=settings.faiss_index_path,
        embeddings_metadata_path=settings.embeddings_path.with_suffix(".json"),
        context_window=settings.context_window_size,
    )


def _check_indexed(settings) -> None:
    """Raise HTTP 503 if the corpus has not been indexed yet."""
    from app.storage.summary_store import SummaryStore
    from app.storage.graph_store import GraphStore

    summary_store = SummaryStore(artifacts_dir=settings.artifacts_dir)
    graph_store   = GraphStore(artifacts_dir=settings.artifacts_dir)

    if not summary_store.summaries_exist() and not graph_store.graph_exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "not_indexed",
                "message": (
                    "The corpus has not been indexed yet. "
                    "Run POST /api/v1/index first and wait for completion."
                ),
            },
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        200: {"description": "Query answered successfully."},
        503: {"description": "Corpus not indexed. Run POST /api/v1/index first."},
    },
    summary="Query the RAG systems",
    description=(
        "Submit a global sensemaking question to GraphRAG, VectorRAG, or both. "
        "When `mode=both`, both engines run concurrently and their answers are "
        "returned side-by-side with combined token usage for cost comparison."
    ),
)
async def query(
    request: QueryRequest,
    _auth: AuthDep,
) -> QueryResponse:
    """
    Run one or both RAG engines and return answers.

    - mode=graphrag  → map-reduce over community summaries
    - mode=vectorrag → FAISS top-k retrieval + 8k context window
    - mode=both      → both engines run concurrently (recommended)
    """
    from app.config import get_settings
    settings = get_settings()
    request_id = get_request_id()
    t0 = time.monotonic()

    # Validate index exists before building engines
    _check_indexed(settings)

    graphrag_answer = None
    vectorrag_answer = None

    try:
        if request.mode == QueryMode.BOTH:
            # Run both engines concurrently
            graphrag_engine  = _build_graphrag_engine()
            vectorrag_engine = _build_vectorrag_engine()

            graphrag_answer, vectorrag_answer = await asyncio.gather(
                graphrag_engine.query(
                    question=request.query,
                    community_level=request.community_level.value,
                    helpfulness_threshold=request.helpfulness_threshold,
                    include_context=request.include_context,
                    include_token_usage=request.include_token_usage,
                ),
                vectorrag_engine.query(
                    question=request.query,
                    top_k=request.top_k,
                    include_context=request.include_context,
                    include_token_usage=request.include_token_usage,
                ),
            )

        elif request.mode == QueryMode.GRAPHRAG:
            graphrag_engine = _build_graphrag_engine()
            graphrag_answer = await graphrag_engine.query(
                question=request.query,
                community_level=request.community_level.value,
                helpfulness_threshold=request.helpfulness_threshold,
                include_context=request.include_context,
                include_token_usage=request.include_token_usage,
            )

        else:  # VECTORRAG
            vectorrag_engine = _build_vectorrag_engine()
            vectorrag_answer = await vectorrag_engine.query(
                question=request.query,
                top_k=request.top_k,
                include_context=request.include_context,
                include_token_usage=request.include_token_usage,
            )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "index_missing",
                "message": str(exc),
            },
        )
    except Exception as exc:
        log.error("Query failed", error=str(exc), query=request.query[:100])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "query_failed",
                "message": f"Query execution failed: {str(exc)}",
            },
        )

    total_ms = round((time.monotonic() - t0) * 1000, 1)

    # Aggregate token usage across both engines
    total_token_usage: TokenUsage | None = None
    if request.include_token_usage:
        usage = TokenUsage.zero()
        if graphrag_answer and graphrag_answer.token_usage:
            usage = usage + graphrag_answer.token_usage
        if vectorrag_answer and vectorrag_answer.token_usage:
            usage = usage + vectorrag_answer.token_usage
        total_token_usage = usage if usage.total_tokens > 0 else None

    log.info(
        "Query completed",
        mode=request.mode.value,
        community_level=request.community_level.value,
        total_ms=total_ms,
        total_tokens=total_token_usage.total_tokens if total_token_usage else 0,
    )

    return QueryResponse(
        query=request.query,
        mode=request.mode.value,
        request_id=request_id,
        graphrag=graphrag_answer,
        vectorrag=vectorrag_answer,
        total_latency_ms=total_ms,
        total_token_usage=total_token_usage,
    )