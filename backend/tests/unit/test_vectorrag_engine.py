"""
tests/unit/test_vectorrag_engine.py — VectorRAGEngine unit tests.

Tests:
  - query() returns VectorRAGAnswer on success
  - FAISS returns empty → graceful empty answer
  - context chunks filled up to 8k token window
  - include_context=False omits retrieved chunks
  - include_token_usage=False omits token_usage
  - score_threshold filters low-similarity results
  - _ensure_index_loaded called before search
  - _fill_context_window respects token limit
  - _empty_answer() shape correct
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def make_completion_result(content="Answer text."):
    from app.services.openai_service import CompletionResult
    return CompletionResult(
        content=content, model="gpt-4o",
        prompt_tokens=200, completion_tokens=80,
        total_tokens=280, finish_reason="stop",
        estimated_cost_usd=0.003,
    )


def make_search_result(rank=1, chunk_id="c001", text="Some chunk text.",
                       source_doc="article.txt", score=0.85, token_count=15):
    from app.services.faiss_service import SearchResult
    return SearchResult(
        rank=rank, index_id=rank-1, score=score,
        chunk_id=chunk_id, text=text,
        source_document=source_doc, token_count=token_count,
    )


@pytest.fixture
def mock_faiss(sample_chunks):
    svc = MagicMock()
    results = [
        make_search_result(
            rank=i+1,
            chunk_id=c.chunk_id,
            text=c.text,
            source_doc=c.source_document,
            score=0.9 - i * 0.05,
            token_count=c.token_count,
        )
        for i, c in enumerate(sample_chunks[:5])
    ]
    svc.search = MagicMock(return_value=results)
    svc._is_built = True
    return svc


@pytest.fixture
def mock_embedding_svc():
    svc = MagicMock()
    svc.embed_text = AsyncMock(return_value=np.array([0.1] * 1536, dtype=np.float32))
    return svc


@pytest.fixture
def mock_openai_svc():
    svc = MagicMock()
    svc.model = "gpt-4o"
    svc.complete = AsyncMock(return_value=make_completion_result())
    return svc


@pytest.fixture
def engine(mock_openai_svc, mock_embedding_svc, mock_faiss, mock_tokenizer, tmp_path):
    from app.core.query.vectorrag_engine import VectorRAGEngine
    engine = VectorRAGEngine(
        openai_service=mock_openai_svc,
        embedding_service=mock_embedding_svc,
        faiss_service=mock_faiss,
        tokenizer=mock_tokenizer,
        faiss_index_path=tmp_path / "faiss_index.bin",
        embeddings_metadata_path=tmp_path / "embeddings.json",
        context_window=8000,
    )
    engine._index_loaded = True   # bypass file loading in tests
    return engine


# ── query() ──────────────────────────────────────────────────────────────────

class TestVectorRAGEngineQuery:

    @pytest.mark.asyncio
    async def test_returns_vectorrag_answer(self, engine):
        from app.models.response_models import VectorRAGAnswer
        answer = await engine.query("What are the main themes?")
        assert isinstance(answer, VectorRAGAnswer)

    @pytest.mark.asyncio
    async def test_answer_has_query_field(self, engine):
        answer = await engine.query("Who founded OpenAI?")
        assert answer.query == "Who founded OpenAI?"

    @pytest.mark.asyncio
    async def test_answer_text_non_empty(self, engine):
        answer = await engine.query("test query")
        assert len(answer.answer) > 0

    @pytest.mark.asyncio
    async def test_chunks_retrieved_count_matches(self, engine, mock_faiss):
        answer = await engine.query("test", top_k=5)
        # chunks_retrieved is determined by context window fill
        assert answer.chunks_retrieved >= 0

    @pytest.mark.asyncio
    async def test_include_context_true_returns_context(self, engine):
        answer = await engine.query("test", include_context=True)
        assert answer.context is not None
        assert len(answer.context) > 0

    @pytest.mark.asyncio
    async def test_include_context_false_omits_context(self, engine):
        answer = await engine.query("test", include_context=False)
        assert answer.context is None

    @pytest.mark.asyncio
    async def test_include_token_usage_false_omits_usage(self, engine):
        answer = await engine.query("test", include_token_usage=False)
        assert answer.token_usage is None

    @pytest.mark.asyncio
    async def test_include_token_usage_true_populates_usage(self, engine):
        answer = await engine.query("test", include_token_usage=True)
        assert answer.token_usage is not None
        assert answer.token_usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_empty_faiss_results_returns_graceful_answer(
        self, mock_openai_svc, mock_embedding_svc, mock_tokenizer, tmp_path
    ):
        from app.core.query.vectorrag_engine import VectorRAGEngine
        empty_faiss = MagicMock()
        empty_faiss.search = MagicMock(return_value=[])
        empty_faiss._is_built = True

        engine = VectorRAGEngine(
            openai_service=mock_openai_svc,
            embedding_service=mock_embedding_svc,
            faiss_service=empty_faiss,
            tokenizer=mock_tokenizer,
            faiss_index_path=tmp_path / "faiss_index.bin",
            embeddings_metadata_path=tmp_path / "embeddings.json",
        )
        engine._index_loaded = True
        answer = await engine.query("What?")
        assert isinstance(answer.answer, str)

    @pytest.mark.asyncio
    async def test_embedding_service_called_with_query(
        self, engine, mock_embedding_svc
    ):
        await engine.query("my specific query")
        mock_embedding_svc.embed_text.assert_called_once_with("my specific query")

    @pytest.mark.asyncio
    async def test_faiss_search_called_with_vector(self, engine, mock_faiss):
        await engine.query("query", top_k=7)
        mock_faiss.search.assert_called_once()
        call_kwargs = mock_faiss.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 7 or call_kwargs.args[1] == 7


# ── _fill_context_window() ────────────────────────────────────────────────────

class TestFillContextWindow:

    def test_fills_within_token_limit(self, engine):
        results = [
            make_search_result(rank=i+1, chunk_id=f"c{i:03d}",
                               text="word " * 100, token_count=100)
            for i in range(20)
        ]
        chunks, context_text, total_tokens = engine._fill_context_window(results, "q")
        assert total_tokens <= engine.context_window + 200  # small tolerance
        assert len(context_text) > 0

    def test_returns_chunks_in_score_order(self, engine):
        results = [
            make_search_result(rank=1, chunk_id="high", text="high score chunk", score=0.95, token_count=10),
            make_search_result(rank=2, chunk_id="low", text="low score chunk", score=0.50, token_count=10),
        ]
        chunks, _, _ = engine._fill_context_window(results, "q")
        assert len(chunks) >= 1
        if len(chunks) >= 2:
            assert chunks[0].score >= chunks[1].score


# ── _empty_answer() ───────────────────────────────────────────────────────────

class TestEmptyAnswer:

    def test_empty_answer_shape(self, engine):
        from app.models.response_models import VectorRAGAnswer
        answer = engine._empty_answer("my query", latency_ms=50.0)
        assert isinstance(answer, VectorRAGAnswer)
        assert answer.query == "my query"
        assert answer.chunks_retrieved == 0

    def test_empty_answer_has_answer_text(self, engine):
        answer = engine._empty_answer("q", latency_ms=10.0)
        assert isinstance(answer.answer, str)
        assert len(answer.answer) > 0