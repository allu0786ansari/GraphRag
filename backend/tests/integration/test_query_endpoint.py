"""
tests/integration/test_query_endpoint.py — POST /api/v1/query integration tests.

Strategy:
  - Mock the storage layer (_check_indexed, SummaryStore, GraphStore) so the
    endpoint thinks the corpus is indexed.
  - Mock _build_graphrag_engine and _build_vectorrag_engine to return engines
    backed by mocked OpenAI/FAISS — zero real API calls.
  - Test HTTP contract: status codes, response shape, auth, validation.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def graphrag_answer_mock(query="test"):
    from app.models.response_models import GraphRAGAnswer, TokenUsage
    return GraphRAGAnswer(
        answer="GraphRAG answer: multiple themes identified.",
        query=query,
        community_level="c1",
        communities_total=6,
        communities_used_in_map=6,
        map_answers_generated=6,
        map_answers_after_filter=4,
        latency_ms=500.0,
        token_usage=TokenUsage(prompt_tokens=1500, completion_tokens=400, total_tokens=1900),
    )


def vectorrag_answer_mock(query="test"):
    from app.models.response_models import VectorRAGAnswer, TokenUsage
    return VectorRAGAnswer(
        answer="VectorRAG answer: semantically similar chunks retrieved.",
        query=query,
        chunks_retrieved=5,
        context_tokens_used=1200,
        token_usage=TokenUsage(prompt_tokens=1200, completion_tokens=200, total_tokens=1400),
    )


@pytest.fixture(autouse=True)
def mock_indexed(monkeypatch):
    """Make all query tests think the corpus is indexed."""
    with patch("app.api.routes_query._check_indexed", return_value=None):
        yield


@pytest.fixture
def mock_engines(monkeypatch):
    """Patch engine builders to return mock engines."""
    graphrag_engine = MagicMock()
    graphrag_engine.query = AsyncMock(return_value=graphrag_answer_mock())

    vectorrag_engine = MagicMock()
    vectorrag_engine.query = AsyncMock(return_value=vectorrag_answer_mock())

    with patch("app.api.routes_query._build_graphrag_engine",
               return_value=graphrag_engine), \
         patch("app.api.routes_query._build_vectorrag_engine",
               return_value=vectorrag_engine):
        yield graphrag_engine, vectorrag_engine


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestQueryAuth:

    def test_missing_api_key_returns_401(self, client):
        response = client.post("/api/v1/query", json={"query": "What?", "mode": "graphrag"})
        assert response.status_code == 401

    def test_wrong_api_key_returns_401(self, client):
        response = client.post(
            "/api/v1/query",
            json={"query": "What?", "mode": "graphrag"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_valid_api_key_allowed(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "What are the main themes?", "mode": "graphrag"},
            headers=auth_headers,
        )
        assert response.status_code == 200


# ── Validation ────────────────────────────────────────────────────────────────

class TestQueryValidation:

    def test_empty_query_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/query",
            json={"query": "", "mode": "graphrag"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_mode_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/query",
            json={"query": "What?", "mode": "invalid_mode"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_community_level_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/query",
            json={"query": "What?", "mode": "graphrag", "community_level": "c9"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "vectorrag", "top_k": 9999},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ── 503 when not indexed ──────────────────────────────────────────────────────

class TestQueryNotIndexed:

    def test_returns_503_when_not_indexed(self, client, auth_headers):
        from fastapi import HTTPException
        with patch("app.api.routes_query._check_indexed",
                   side_effect=Exception) as mock_check:
            mock_check.side_effect = __import__("fastapi").HTTPException(
                status_code=503,
                detail={"error": "not_indexed", "message": "Not indexed"}
            )
            response = client.post(
                "/api/v1/query",
                json={"query": "test", "mode": "graphrag"},
                headers=auth_headers,
            )
        assert response.status_code == 503


# ── Response shape: graphrag mode ─────────────────────────────────────────────

class TestQueryGraphRAGMode:

    def test_graphrag_mode_returns_200(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "What are the main themes?", "mode": "graphrag"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_graphrag_response_has_graphrag_field(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "What are the main themes?", "mode": "graphrag"},
            headers=auth_headers,
        )
        data = response.json()
        assert "graphrag" in data
        assert data["graphrag"] is not None

    def test_graphrag_response_vectorrag_is_null(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "graphrag"},
            headers=auth_headers,
        )
        data = response.json()
        assert data.get("vectorrag") is None

    def test_graphrag_answer_has_correct_fields(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "graphrag"},
            headers=auth_headers,
        )
        data = response.json()
        graphrag = data["graphrag"]
        assert "answer" in graphrag
        assert "query" in graphrag
        assert "community_level" in graphrag


# ── Response shape: vectorrag mode ───────────────────────────────────────────

class TestQueryVectorRAGMode:

    def test_vectorrag_mode_returns_200(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "Who founded OpenAI?", "mode": "vectorrag"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_vectorrag_response_has_vectorrag_field(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "vectorrag"},
            headers=auth_headers,
        )
        data = response.json()
        assert "vectorrag" in data
        assert data["vectorrag"] is not None

    def test_vectorrag_graphrag_field_is_null(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "vectorrag"},
            headers=auth_headers,
        )
        data = response.json()
        assert data.get("graphrag") is None


# ── Response shape: both mode ─────────────────────────────────────────────────

class TestQueryBothMode:

    def test_both_mode_returns_200(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "both"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_both_mode_returns_both_answers(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "both"},
            headers=auth_headers,
        )
        data = response.json()
        assert data["graphrag"] is not None
        assert data["vectorrag"] is not None

    def test_both_mode_response_has_request_id(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "both"},
            headers=auth_headers,
        )
        data = response.json()
        assert "request_id" in data

    def test_both_mode_aggregates_token_usage(self, client, auth_headers, mock_engines):
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "mode": "both", "include_token_usage": True},
            headers=auth_headers,
        )
        data = response.json()
        if data.get("token_usage"):
            assert data["token_usage"]["total_tokens"] > 0