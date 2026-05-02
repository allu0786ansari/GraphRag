"""
tests/conftest.py — Shared pytest fixtures.

Fixtures defined here are available to ALL tests without importing.
Add fixtures here that are needed across unit and integration tests.

Stage 1: settings override, app client, log capture.
Later stages add: mock LLM client, sample chunks, test graph, etc.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


# ── Override settings for tests ───────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def set_test_environment(tmp_path_factory):
    """
    Force test environment variables before any module loads settings.
    Uses a session-scoped temp directory for all file I/O.
    """
    tmp = tmp_path_factory.mktemp("graphrag_test")

    env_overrides = {
        "APP_ENV":          "development",
        "DEBUG":            "true",
        "API_KEY":          "test-api-key-12345",
        "OPENAI_API_KEY":   "sk-test-000000000000000000000000000000000000000000000000",
        "LOG_LEVEL":        "DEBUG",
        "LOG_FORMAT":       "text",
        "RATE_LIMIT_ENABLED": "false",
        "DATA_DIR":         str(tmp / "data"),
        "ARTIFACTS_DIR":    str(tmp / "data" / "processed"),
        "RAW_DATA_DIR":     str(tmp / "data" / "raw"),
        "EVALUATION_DIR":   str(tmp / "data" / "evaluation"),
        "LOGS_DIR":         str(tmp / "logs"),
        "FAISS_INDEX_PATH": str(tmp / "data" / "processed" / "faiss_index.bin"),
        "EMBEDDINGS_PATH":  str(tmp / "data" / "processed" / "embeddings.npy"),
    }

    for key, val in env_overrides.items():
        os.environ[key] = val

    # Clear cached settings so test env vars are picked up
    from app.config import get_settings
    get_settings.cache_clear()

    yield tmp

    # Cleanup env overrides after session
    for key in env_overrides:
        os.environ.pop(key, None)

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def settings(set_test_environment):
    """Return test settings instance."""
    from app.config import get_settings
    return get_settings()


# ── FastAPI test client ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def app(settings):
    """Return the FastAPI application configured for testing."""
    from app.main import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """
    Return a synchronous TestClient for the FastAPI app.

    Usage:
        def test_health(client):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Return valid API key headers for authenticated requests."""
    return {"X-API-Key": "test-api-key-12345"}


# ── Mock OpenAI client (added to by later stages) ─────────────────────────────
@pytest.fixture
def mock_openai_client():
    """
    Mock OpenAI client that never makes real API calls.

    Stages 3+ extend this fixture with specific return values
    for extraction, summarization, and evaluation prompts.
    """
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Mock LLM response"),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )
    )
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)],
            usage=MagicMock(total_tokens=10),
        )
    )
    return client


@pytest.fixture
def mock_tokenizer():
    """Return a real tokenizer service for token-counting and chunking tests."""
    from app.services.tokenizer_service import TokenizerService
    return TokenizerService(model="gpt-4o")


@pytest.fixture
def sample_chunks(sample_chunk):
    """Return multiple sample chunks for FAISS and vector retrieval tests."""
    return [
        SimpleNamespace(
            chunk_id=f"chunk_{i:04d}",
            source_document=sample_chunk["source_document"],
            text=sample_chunk["text"] + f" extra {i}",
            token_count=sample_chunk["token_count"],
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_summaries():
    """Return synthetic CommunitySummary objects for GraphRAG tests."""
    from app.models.graph_models import CommunityLevel, CommunityFinding, CommunitySummary

    return [
        CommunitySummary(
            community_id="comm_c1_0001",
            level=CommunityLevel.C1,
            title="AI ecosystem overview",
            summary="A summary of the AI ecosystem focusing on investment and major players.",
            impact_rating=7.5,
            rating_explanation="This community captures the key growth drivers in AI.",
            findings=[
                CommunityFinding(
                    finding_id=0,
                    summary="OpenAI and Microsoft are central to the AI investment landscape.",
                    explanation="OpenAI's partnership with Microsoft and large investments establish a dominant industry position.",
                )
            ],
            node_ids=["OpenAI", "Microsoft"],
            context_tokens_used=300,
            was_truncated=False,
            used_sub_community_substitution=False,
        ),
        CommunitySummary(
            community_id="comm_c1_0002",
            level=CommunityLevel.C1,
            title="AI regulation discussion",
            summary="A summary of regulatory debates around AI safety and oversight.",
            impact_rating=6.0,
            rating_explanation="Regulation is an important emerging theme in AI governance.",
            findings=[
                CommunityFinding(
                    finding_id=0,
                    summary="AI safety is a growing concern among policymakers.",
                    explanation="Discussions focus on transparency, privacy, and responsible deployment.",
                )
            ],
            node_ids=["AI safety", "policy"],
            context_tokens_used=250,
            was_truncated=False,
            used_sub_community_substitution=False,
        ),
    ]


# ── Sample data fixtures (populated by later stages) ─────────────────────────
@pytest.fixture
def sample_text():
    """Short text sample for chunking and extraction tests."""
    return (
        "OpenAI was founded in 2015 by Sam Altman, Elon Musk, and others. "
        "The company created GPT-4, a large language model used globally. "
        "Microsoft invested $10 billion in OpenAI in 2023. "
        "Google responded by investing in Anthropic, an AI safety company. "
        "Anthropic was founded by Dario Amodei and Daniela Amodei in 2021. "
        "The AI industry saw rapid growth with companies like Meta releasing Llama. "
        "Sam Altman became CEO of OpenAI after a brief board dispute in November 2023. "
    ) * 10  # repeat to get a realistic chunk size


@pytest.fixture
def sample_chunk():
    """A single pre-chunked text unit as produced by chunking.py."""
    return {
        "chunk_id": "chunk_0001",
        "source_document": "news_article_001.json",
        "text": "OpenAI was founded in 2015 by Sam Altman and Elon Musk. "
                "Microsoft invested $10 billion in the company in 2023.",
        "token_count": 32,
        "start_token": 0,
        "end_token": 32,
    }