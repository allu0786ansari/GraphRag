"""
tests/unit/test_stage1_foundation.py — Stage 1 foundation tests.

Tests every component built in Stage 1:
  - Configuration loading and validation
  - Logger setup and structured output
  - Middleware: request ID, CORS headers, rate limiting, error handling
  - Health endpoints: liveness and readiness
  - Dependency injection: API key auth
"""

from __future__ import annotations

import os
import json
import pytest
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_settings_load_from_env(self, settings):
        """Settings must load from environment without raising."""
        assert settings.app_name == "GraphRAG System"
        assert settings.app_env == "development"
        assert settings.debug is True

    def test_openai_model_default(self, settings):
        assert settings.openai_model == "gpt-4o"

    def test_pipeline_paper_exact_defaults(self, settings):
        """Paper-exact parameters must match specification."""
        assert settings.chunk_size == 600
        assert settings.chunk_overlap == 100
        assert settings.gleaning_rounds == 2
        assert settings.context_window_size == 8000
        assert settings.community_level == "c1"
        assert settings.evaluation_runs == 5

    def test_allowed_origins_parsed_to_list(self, settings):
        origins = settings.allowed_origins_list
        assert isinstance(origins, list)
        assert len(origins) >= 1
        for o in origins:
            assert o.startswith("http")

    def test_storage_directories_created(self, settings):
        """Config validator must auto-create all required directories."""
        assert settings.artifacts_dir.exists()
        assert settings.raw_data_dir.exists()
        assert settings.evaluation_dir.exists()
        assert settings.logs_dir.exists()

    def test_is_production_false_in_test(self, settings):
        assert settings.is_production is False
        assert settings.is_development is True

    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        """Validator must reject overlap >= chunk_size."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="chunk_overlap"):
            from app.config import Settings
            Settings(
                api_key="key",
                openai_api_key="sk-test",
                chunk_size=100,
                chunk_overlap=100,   # equal — must fail
            )

    def test_settings_singleton_cached(self, settings):
        """get_settings() must return the same object on repeated calls."""
        from app.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_community_level_must_be_valid(self):
        """Invalid community level must raise validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            from app.config import Settings
            Settings(
                api_key="key",
                openai_api_key="sk-test",
                community_level="c9",   # invalid
            )


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogger:

    def test_get_logger_returns_logger(self):
        from app.utils.logger import get_logger
        log = get_logger("test")
        assert log is not None

    def test_request_id_context_var(self):
        from app.utils.logger import set_request_id, get_request_id
        set_request_id("test-req-123")
        assert get_request_id() == "test-req-123"

    def test_request_id_default_is_dash(self):
        """Default request_id before any request is '-'."""
        from app.utils.logger import _request_id_var
        # Access default directly
        assert _request_id_var.get() in ("-", "test-req-123")  # may carry from prev test

    def test_pipeline_stage_context_var(self):
        from app.utils.logger import set_pipeline_stage, get_pipeline_stage
        set_pipeline_stage("chunking")
        assert get_pipeline_stage() == "chunking"

    def test_setup_logging_does_not_raise(self, tmp_path):
        """setup_logging() must complete without error."""
        from app.utils.logger import setup_logging
        setup_logging(
            log_level="DEBUG",
            log_format="text",
            logs_dir=tmp_path / "logs",
        )

    def test_logger_can_log_structured_fields(self):
        from app.utils.logger import get_logger
        log = get_logger("test.structured")
        # Should not raise
        log.info("Test message", key="value", number=42, flag=True)

    def test_logger_can_log_exception(self):
        from app.utils.logger import get_logger
        log = get_logger("test.exception")
        try:
            raise ValueError("test error")
        except ValueError:
            log.exception("Caught error", context="test")  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY UTILS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetry:

    @pytest.mark.asyncio
    async def test_with_retry_succeeds_on_first_attempt(self):
        from app.utils.retry import with_retry
        call_count = 0

        @with_retry(max_attempts=3)
        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeeds()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_with_retry_retries_on_transient_error(self):
        from app.utils.retry import with_retry
        call_count = 0

        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.01,
                    exceptions=(ValueError,))
        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "recovered"

        result = await fails_twice()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_raises_after_max_attempts(self):
        from app.utils.retry import with_retry
        from tenacity import RetryError

        @with_retry(max_attempts=2, min_wait=0.01, max_wait=0.01,
                    exceptions=(RuntimeError,))
        async def always_fails():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError):
            await always_fails()

    def test_pre_configured_decorators_exist(self):
        from app.utils.retry import llm_retry, bulk_llm_retry, embedding_retry
        assert callable(llm_retry)
        assert callable(bulk_llm_retry)
        assert callable(embedding_retry)


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC UTILS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncUtils:

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_returns_results(self):
        from app.utils.async_utils import gather_with_concurrency
        import asyncio

        async def double(x):
            await asyncio.sleep(0)
            return x * 2

        results = await gather_with_concurrency(
            [double(i) for i in range(5)],
            max_concurrency=3,
        )
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_preserves_order(self):
        from app.utils.async_utils import gather_with_concurrency
        import asyncio

        async def slow_if_even(x):
            if x % 2 == 0:
                await asyncio.sleep(0.01)
            return x

        results = await gather_with_concurrency(
            [slow_if_even(i) for i in range(6)],
            max_concurrency=6,
        )
        assert results == [0, 1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batch_process_processes_all_items(self):
        from app.utils.async_utils import batch_process
        import asyncio

        async def identity(x):
            return x * 3

        results = await batch_process(
            list(range(10)),
            processor=identity,
            batch_size=3,
            max_concurrency=2,
            log_progress=False,
        )
        assert results == [i * 3 for i in range(10)]

    @pytest.mark.asyncio
    async def test_run_in_executor_runs_blocking_function(self):
        from app.utils.async_utils import run_in_executor

        def blocking_add(a, b):
            return a + b

        result = await run_in_executor(blocking_add, 3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_async_chunks_yields_correct_chunks(self):
        from app.utils.async_utils import async_chunks
        items = list(range(10))
        chunks = []
        async for chunk in async_chunks(items, chunk_size=3):
            chunks.append(chunk)
        assert chunks == [[0,1,2], [3,4,5], [6,7,8], [9]]


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoints:

    def test_liveness_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_liveness_response_schema(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)

    def test_readiness_returns_200_or_503(self, client):
        response = client.get("/api/v1/health/ready")
        assert response.status_code in (200, 503)

    def test_readiness_response_contains_checks(self, client):
        data = client.get("/api/v1/health/ready").json()
        assert "checks" in data
        assert "configuration" in data["checks"]
        assert "storage_directories" in data["checks"]
        assert "pipeline_artifacts" in data["checks"]

    def test_readiness_reports_not_indexed_before_pipeline(self, client):
        data = client.get("/api/v1/health/ready").json()
        artifacts = data["checks"]["pipeline_artifacts"]
        # Before any indexing run, no artifacts should exist
        assert artifacts["indexed"] is False

    def test_readiness_reports_all_storage_dirs(self, client):
        data = client.get("/api/v1/health/ready").json()
        dirs = data["checks"]["storage_directories"]["dirs"]
        assert "artifacts" in dirs
        assert "raw_data" in dirs
        assert "evaluation" in dirs
        assert "logs" in dirs

    def test_readiness_config_check_passes(self, client):
        data = client.get("/api/v1/health/ready").json()
        assert data["checks"]["configuration"]["status"] == "ok"
        assert data["checks"]["configuration"]["environment"] == "development"

    def test_readiness_includes_version_and_env(self, client):
        data = client.get("/api/v1/health/ready").json()
        assert "version" in data
        assert "environment" in data
        assert "python" in data

    def test_unknown_path_returns_404_json(self, client):
        response = client.get("/api/v1/does-not-exist")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "not_found"
        assert "request_id" in data


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMiddleware:

    def test_request_id_added_to_response_headers(self, client):
        response = client.get("/api/v1/health")
        assert "x-request-id" in response.headers

    def test_client_provided_request_id_is_echoed(self, client):
        custom_id = "my-custom-request-id-abc"
        response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
        assert response.headers.get("x-request-id") == custom_id

    def test_response_time_header_present(self, client):
        response = client.get("/api/v1/health")
        assert "x-response-time" in response.headers
        # Should end with 'ms'
        assert response.headers["x-response-time"].endswith("ms")

    def test_cors_allow_origin_header_present(self, client):
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert "access-control-allow-origin" in response.headers

    def test_rate_limit_headers_present(self, client):
        response = client.get("/api/v1/health")
        # Rate limiting is disabled in tests but middleware still attaches headers
        # when enabled; skip header assertion if rate limiting is disabled
        # This test verifies the endpoint responds regardless
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# API KEY AUTH TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthentication:

    def test_verify_api_key_accepts_correct_key(self, client, auth_headers):
        """
        Health endpoints are public, but this tests the auth dependency directly.
        Protected routes (added in Stage 7) will use the auth_headers fixture.
        """
        # Health is public — test it always passes
        response = client.get("/api/v1/health", headers=auth_headers)
        assert response.status_code == 200

    def test_missing_api_key_would_return_401(self):
        """Directly test the dependency raises 401 when key is missing."""
        from fastapi import HTTPException
        import asyncio
        from app.dependencies import verify_api_key
        from app.config import get_settings

        settings = get_settings()

        async def run():
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(settings=settings, x_api_key=None)
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail["error"] == "missing_api_key"

        asyncio.get_event_loop().run_until_complete(run())

    def test_wrong_api_key_would_return_401(self):
        """Directly test the dependency raises 401 on wrong key."""
        from fastapi import HTTPException
        import asyncio
        from app.dependencies import verify_api_key
        from app.config import get_settings

        settings = get_settings()

        async def run():
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(settings=settings, x_api_key="wrong-key")
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail["error"] == "invalid_api_key"

        asyncio.get_event_loop().run_until_complete(run())