"""
tests/unit/test_stage8_workers.py — Stage 8 Background Worker tests.

Test strategy:
  - Zero real LLM/API calls. All OpenAI/pipeline calls are mocked.
  - indexing_worker tests verify job store state transitions directly
    (no HTTP layer needed — pure unit tests of the worker module).
  - extraction_worker tests verify pool lifecycle, concurrency cap,
    callback invocation, batch splitting, and cancellation.
  - Integration tests verify routes_indexing still works correctly
    after being refactored to delegate to the worker module.
  - All tests restore global worker state in finally blocks to avoid
    cross-test contamination of the shared _JOB_STORE / _ACTIVE_JOB_ID.

Coverage targets:
  workers/indexing_worker.py    95%+
  workers/extraction_worker.py  90%+
  api/routes_indexing.py        90%+   (refactored thin wrapper)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.request_models import IndexRequest
from app.models.response_models import PipelineStage, PipelineStatus


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_index_request(**overrides) -> IndexRequest:
    """Build a minimal valid IndexRequest."""
    defaults = {
        "chunk_size": 600,
        "chunk_overlap": 100,
        "gleaning_rounds": 2,
        "context_window_size": 8000,
        "max_community_levels": 3,
        "force_reindex": False,
        "skip_claims": False,
        "max_chunks": None,
    }
    defaults.update(overrides)
    return IndexRequest(**defaults)


def make_pipeline_result(
    success: bool = True,
    chunks: int = 100,
    nodes: int = 500,
    edges: int = 800,
    summaries: int = 55,
    error_stage: str | None = None,
    error_message: str | None = None,
) -> Any:
    """Build a mock PipelineResult."""
    from app.core.pipeline.pipeline_runner import PipelineResult
    r = PipelineResult(
        success=success,
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        total_elapsed_seconds=120.0,
    )
    r.chunks_count = chunks
    r.graph_nodes  = nodes
    r.graph_edges  = edges
    r.summaries_count = summaries
    r.error_stage  = error_stage
    r.error_message = error_message
    return r


def _reset_worker_state():
    """Reset global indexing_worker state between tests."""
    import app.workers.indexing_worker as iw
    iw._JOB_STORE.clear()
    iw._ACTIVE_JOB_ID = None


# ═══════════════════════════════════════════════════════════════════════════════
# INDEXING WORKER — JOB STORE UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestJobStore:
    """Direct unit tests of the job-store functions in indexing_worker."""

    def setup_method(self):
        _reset_worker_state()

    def teardown_method(self):
        _reset_worker_state()

    def test_make_blank_stages_returns_all_stages(self):
        from app.workers.indexing_worker import make_blank_stages, STAGE_ORDER
        stages = make_blank_stages()
        assert len(stages) == len(STAGE_ORDER)
        for sp, name in zip(stages, STAGE_ORDER):
            assert sp.stage.value == name
            assert sp.status == PipelineStatus.QUEUED

    def test_submit_indexing_job_creates_job_record(self):
        from app.workers.indexing_worker import submit_indexing_job, get_job, STAGE_ORDER
        request = make_index_request()
        job_id = submit_indexing_job(request)

        assert job_id.startswith("idx_")
        job = get_job(job_id)
        assert job is not None
        assert job["status"] == PipelineStatus.QUEUED
        assert job["started_at"] is None
        assert job["error_message"] is None
        assert len(job["stages"]) == len(STAGE_ORDER)

    def test_submit_indexing_job_sets_active_job_id(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, get_active_job_id, _JOB_STORE, _ACTIVE_JOB_ID
        )
        import app.workers.indexing_worker as iw
        request = make_index_request()
        job_id = submit_indexing_job(request)
        assert iw._ACTIVE_JOB_ID == job_id

    def test_submit_job_while_running_raises(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, _JOB_STORE, STAGE_ORDER, make_blank_stages
        )
        import app.workers.indexing_worker as iw

        # Simulate a running job
        running_id = "idx_fakerunning"
        iw._JOB_STORE[running_id] = {
            "job_id": running_id,
            "status": PipelineStatus.RUNNING,
            "stages": make_blank_stages(),
        }
        iw._ACTIVE_JOB_ID = running_id

        with pytest.raises(RuntimeError, match="already running"):
            submit_indexing_job(make_index_request())

    def test_is_job_running_false_when_no_active(self):
        from app.workers.indexing_worker import is_job_running
        assert is_job_running() is False

    def test_is_job_running_true_when_running(self):
        import app.workers.indexing_worker as iw
        from app.workers.indexing_worker import is_job_running, make_blank_stages
        iw._ACTIVE_JOB_ID = "idx_running"
        iw._JOB_STORE["idx_running"] = {
            "status": PipelineStatus.RUNNING,
            "stages": make_blank_stages(),
        }
        assert is_job_running() is True

    def test_is_job_running_false_when_completed(self):
        import app.workers.indexing_worker as iw
        from app.workers.indexing_worker import is_job_running, make_blank_stages
        iw._ACTIVE_JOB_ID = "idx_done"
        iw._JOB_STORE["idx_done"] = {
            "status": PipelineStatus.COMPLETED,
            "stages": make_blank_stages(),
        }
        # _ACTIVE_JOB_ID is set but status is COMPLETED
        assert is_job_running() is False

    def test_get_job_returns_none_for_unknown(self):
        from app.workers.indexing_worker import get_job
        assert get_job("idx_doesnotexist") is None

    def test_get_most_recent_job_returns_last_submitted(self):
        from app.workers.indexing_worker import submit_indexing_job, get_most_recent_job
        import app.workers.indexing_worker as iw

        iw._ACTIVE_JOB_ID = None
        id1 = submit_indexing_job(make_index_request())
        iw._ACTIVE_JOB_ID = None
        id2 = submit_indexing_job(make_index_request())

        most_recent = get_most_recent_job()
        assert most_recent["job_id"] == id2

    def test_get_most_recent_job_none_when_empty(self):
        from app.workers.indexing_worker import get_most_recent_job
        assert get_most_recent_job() is None

    def test_list_jobs_returns_most_recent_first(self):
        from app.workers.indexing_worker import submit_indexing_job, list_jobs
        import app.workers.indexing_worker as iw

        iw._ACTIVE_JOB_ID = None
        id1 = submit_indexing_job(make_index_request())
        iw._ACTIVE_JOB_ID = None
        id2 = submit_indexing_job(make_index_request())

        jobs = list_jobs()
        assert jobs[0]["job_id"] == id2
        assert jobs[1]["job_id"] == id1

    def test_list_jobs_respects_limit(self):
        from app.workers.indexing_worker import submit_indexing_job, list_jobs
        import app.workers.indexing_worker as iw

        for _ in range(5):
            iw._ACTIVE_JOB_ID = None
            submit_indexing_job(make_index_request())

        assert len(list_jobs(limit=3)) == 3

    def test_current_stage_of_pending(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, get_job, current_stage_of
        )
        job_id = submit_indexing_job(make_index_request())
        job = get_job(job_id)
        assert current_stage_of(job) == PipelineStage.PENDING

    def test_current_stage_of_running_stage(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, get_job, current_stage_of
        )
        job_id = submit_indexing_job(make_index_request())
        job = get_job(job_id)
        # Manually set extraction stage to RUNNING
        job["stages"][1].status = PipelineStatus.RUNNING
        assert current_stage_of(job) == PipelineStage.EXTRACTION

    def test_current_stage_of_completed_job(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, get_job, current_stage_of
        )
        import app.workers.indexing_worker as iw
        job_id = submit_indexing_job(make_index_request())
        iw._ACTIVE_JOB_ID = None
        job = get_job(job_id)
        job["status"] = PipelineStatus.COMPLETED
        assert current_stage_of(job) == PipelineStage.COMPLETED

    def test_current_stage_of_failed_job(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, get_job, current_stage_of
        )
        import app.workers.indexing_worker as iw
        job_id = submit_indexing_job(make_index_request())
        iw._ACTIVE_JOB_ID = None
        job = get_job(job_id)
        job["status"] = PipelineStatus.FAILED
        assert current_stage_of(job) == PipelineStage.FAILED


# ═══════════════════════════════════════════════════════════════════════════════
# INDEXING WORKER — BACKGROUND COROUTINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunIndexingJob:
    """Tests for the run_indexing_job() background coroutine."""

    def setup_method(self):
        _reset_worker_state()

    def teardown_method(self):
        _reset_worker_state()

    @pytest.mark.asyncio
    async def test_successful_run_updates_job_to_completed(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, run_indexing_job, get_job
        )
        import app.workers.indexing_worker as iw

        request = make_index_request()
        job_id = submit_indexing_job(request)

        mock_result = make_pipeline_result(success=True)

        with patch("app.workers.indexing_worker.get_settings") as mock_settings:
            mock_settings.return_value.raw_data_dir = Path("/tmp/raw")
            mock_settings.return_value.artifacts_dir = Path("/tmp/artifacts")
            mock_settings.return_value.openai_api_key = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"

            with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
                runner_instance = AsyncMock()
                runner_instance.run = AsyncMock(return_value=mock_result)
                MockRunner.return_value = runner_instance

                with patch("app.workers.indexing_worker.GraphStore") as MockGS:
                    MockGS.return_value.get_community_counts.return_value = {
                        "c0": 10, "c1": 55
                    }

                    await run_indexing_job(job_id, request)

        job = get_job(job_id)
        assert job["status"] == PipelineStatus.COMPLETED
        assert job["total_chunks"] == mock_result.chunks_count
        assert job["total_nodes"] == mock_result.graph_nodes
        assert job["total_edges"] == mock_result.graph_edges
        assert job["total_summaries"] == mock_result.summaries_count
        assert job["total_communities"] == {"c0": 10, "c1": 55}
        assert job["completed_at"] is not None
        assert iw._ACTIVE_JOB_ID is None   # always cleared

    @pytest.mark.asyncio
    async def test_failed_pipeline_updates_job_to_failed(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, run_indexing_job, get_job
        )
        import app.workers.indexing_worker as iw

        request = make_index_request()
        job_id = submit_indexing_job(request)

        failed_result = make_pipeline_result(
            success=False,
            error_stage="extraction",
            error_message="OpenAI rate limit exceeded",
        )

        with patch("app.workers.indexing_worker.get_settings") as mock_settings:
            mock_settings.return_value.raw_data_dir = Path("/tmp/raw")
            mock_settings.return_value.artifacts_dir = Path("/tmp/artifacts")
            mock_settings.return_value.openai_api_key = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"

            with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
                runner_instance = AsyncMock()
                runner_instance.run = AsyncMock(return_value=failed_result)
                MockRunner.return_value = runner_instance

                await run_indexing_job(job_id, request)

        job = get_job(job_id)
        assert job["status"] == PipelineStatus.FAILED
        assert job["error_message"] == "OpenAI rate limit exceeded"
        assert iw._ACTIVE_JOB_ID is None

    @pytest.mark.asyncio
    async def test_unexpected_exception_sets_failed_status(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, run_indexing_job, get_job
        )
        import app.workers.indexing_worker as iw

        request = make_index_request()
        job_id = submit_indexing_job(request)

        with patch("app.workers.indexing_worker.get_settings") as mock_settings:
            mock_settings.return_value.raw_data_dir = Path("/tmp/raw")
            mock_settings.return_value.artifacts_dir = Path("/tmp/artifacts")
            mock_settings.return_value.openai_api_key = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"

            with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
                MockRunner.side_effect = RuntimeError("Cannot connect to OpenAI")

                await run_indexing_job(job_id, request)

        job = get_job(job_id)
        assert job["status"] == PipelineStatus.FAILED
        assert "Cannot connect to OpenAI" in job["error_message"]
        assert iw._ACTIVE_JOB_ID is None   # cleared even on crash

    @pytest.mark.asyncio
    async def test_active_job_id_cleared_after_success(self):
        from app.workers.indexing_worker import (
            submit_indexing_job, run_indexing_job
        )
        import app.workers.indexing_worker as iw

        request = make_index_request()
        job_id = submit_indexing_job(request)
        assert iw._ACTIVE_JOB_ID == job_id

        with patch("app.workers.indexing_worker.get_settings") as mock_settings:
            mock_settings.return_value.raw_data_dir = Path("/tmp/raw")
            mock_settings.return_value.artifacts_dir = Path("/tmp/artifacts")
            mock_settings.return_value.openai_api_key = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"

            with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
                runner_instance = AsyncMock()
                runner_instance.run = AsyncMock(
                    return_value=make_pipeline_result(success=True)
                )
                MockRunner.return_value = runner_instance
                with patch("app.workers.indexing_worker.GraphStore"):
                    await run_indexing_job(job_id, request)

        assert iw._ACTIVE_JOB_ID is None

    @pytest.mark.asyncio
    async def test_on_progress_callback_updates_stage_progress(self):
        """on_progress updates stage progress_pct in the job store."""
        from app.workers.indexing_worker import (
            submit_indexing_job, run_indexing_job, get_job
        )

        request = make_index_request()
        job_id = submit_indexing_job(request)

        # We'll intercept the on_progress call by capturing what the runner
        # receives and calling it ourselves mid-run
        captured_callback = [None]

        async def fake_run(force_reindex=False, max_chunks=None, on_progress=None):
            captured_callback[0] = on_progress
            if on_progress:
                on_progress("chunking", 1.0)
                on_progress("extraction", 0.5)    # sub-step progress
                on_progress("extraction", 1.0)
            return make_pipeline_result(success=True)

        with patch("app.workers.indexing_worker.get_settings") as mock_settings:
            mock_settings.return_value.raw_data_dir = Path("/tmp/raw")
            mock_settings.return_value.artifacts_dir = Path("/tmp/artifacts")
            mock_settings.return_value.openai_api_key = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"

            with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
                runner_instance = AsyncMock()
                runner_instance.run = fake_run
                MockRunner.return_value = runner_instance
                with patch("app.workers.indexing_worker.GraphStore"):
                    await run_indexing_job(job_id, request)

        job = get_job(job_id)
        # Chunking should be COMPLETED
        chunking_stage = next(
            sp for sp in job["stages"] if sp.stage.value == "chunking"
        )
        assert chunking_stage.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_skipped_stages_marked_completed_on_success(self):
        """Stages skipped due to checkpointing are still marked COMPLETED."""
        from app.workers.indexing_worker import (
            submit_indexing_job, run_indexing_job, get_job
        )

        request = make_index_request()
        job_id = submit_indexing_job(request)

        with patch("app.workers.indexing_worker.get_settings") as mock_settings:
            mock_settings.return_value.raw_data_dir = Path("/tmp/raw")
            mock_settings.return_value.artifacts_dir = Path("/tmp/artifacts")
            mock_settings.return_value.openai_api_key = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"

            with patch("app.workers.indexing_worker.PipelineRunner") as MockRunner:
                runner_instance = AsyncMock()
                # Pipeline succeeds without calling on_progress at all (all cached)
                async def fake_run_all_skipped(**kwargs):
                    return make_pipeline_result(success=True)
                runner_instance.run = fake_run_all_skipped
                MockRunner.return_value = runner_instance
                with patch("app.workers.indexing_worker.GraphStore"):
                    await run_indexing_job(job_id, request)

        job = get_job(job_id)
        assert job["status"] == PipelineStatus.COMPLETED
        # All stages should be COMPLETED or QUEUED→COMPLETED
        for sp in job["stages"]:
            assert sp.status == PipelineStatus.COMPLETED


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION WORKER POOL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractionWorkerPool:
    """Tests for ExtractionWorkerPool and run_extraction_workers()."""

    def _make_mock_services(self):
        openai_svc = MagicMock()
        tokenizer  = MagicMock()
        return openai_svc, tokenizer

    def _make_mock_chunk(self, chunk_id: str = "chunk_001"):
        chunk = MagicMock()
        chunk.chunk_id = chunk_id
        chunk.text = "Some text content."
        return chunk

    def _make_mock_extraction(self, chunk_id: str, success: bool = True):
        extraction = MagicMock()
        extraction.chunk_id = chunk_id
        extraction.extraction_completed = success
        extraction.entities = [MagicMock()] if success else []
        extraction.relationships = [MagicMock()] if success else []
        extraction.error_message = None if success else "LLM error"
        return extraction

    @pytest.mark.asyncio
    async def test_pool_as_context_manager_builds_pipeline(self):
        """Pool __aenter__ builds ExtractionPipeline."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        openai_svc, tokenizer = self._make_mock_services()

        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                    gleaning_rounds=2,
                    max_concurrency=5,
                ) as pool:
                    assert pool._pipeline is not None
                    assert pool.max_concurrency == 5
                MockPipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_batch_empty_chunks_returns_empty(self):
        """Empty chunk list returns ExtractionBatchResult with zeros."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline"):
            with patch("app.workers.extraction_worker.GleaningLoop"):
                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                ) as pool:
                    result = await pool.extract_batch(chunks=[])

        assert result.total_chunks == 0
        assert result.successful == 0
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_extract_batch_calls_pipeline_per_chunk(self):
        """extract_batch calls extract_chunk once per chunk."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        chunks = [self._make_mock_chunk(f"chunk_{i:03d}") for i in range(3)]
        extractions = [
            self._make_mock_extraction(f"chunk_{i:03d}", success=True)
            for i in range(3)
        ]

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                pipeline_instance.extract_chunk = AsyncMock(
                    side_effect=extractions
                )
                MockPipeline.return_value = pipeline_instance

                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                    gleaning_rounds=2,
                ) as pool:
                    result = await pool.extract_batch(chunks=chunks)

        assert result.total_chunks == 3
        assert result.successful == 3
        assert result.failed == 0
        assert pipeline_instance.extract_chunk.call_count == 3

    @pytest.mark.asyncio
    async def test_extract_batch_invokes_on_chunk_complete(self):
        """on_chunk_complete is called once per chunk."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        chunks = [self._make_mock_chunk(f"chunk_{i:03d}") for i in range(4)]
        extractions = [
            self._make_mock_extraction(f"chunk_{i:03d}", success=True)
            for i in range(4)
        ]
        callback_calls = []

        async def on_complete(extraction):
            callback_calls.append(extraction.chunk_id)

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                pipeline_instance.extract_chunk = AsyncMock(side_effect=extractions)
                MockPipeline.return_value = pipeline_instance

                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                ) as pool:
                    result = await pool.extract_batch(
                        chunks=chunks,
                        on_chunk_complete=on_complete,
                    )

        assert len(callback_calls) == 4
        assert result.total_chunks == 4

    @pytest.mark.asyncio
    async def test_extract_batch_handles_failed_extractions(self):
        """Failed extractions are counted but don't stop the batch."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        chunks = [self._make_mock_chunk(f"chunk_{i:03d}") for i in range(3)]
        extractions = [
            self._make_mock_extraction("chunk_000", success=True),
            self._make_mock_extraction("chunk_001", success=False),  # fails
            self._make_mock_extraction("chunk_002", success=True),
        ]

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                pipeline_instance.extract_chunk = AsyncMock(side_effect=extractions)
                MockPipeline.return_value = pipeline_instance

                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                ) as pool:
                    result = await pool.extract_batch(chunks=chunks)

        assert result.total_chunks == 3
        assert result.successful == 2
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_extract_batch_handles_exception_per_chunk(self):
        """LLM exception on one chunk doesn't crash the whole batch."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        chunks = [self._make_mock_chunk(f"chunk_{i:03d}") for i in range(2)]

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                # First chunk raises, second succeeds
                pipeline_instance.extract_chunk = AsyncMock(
                    side_effect=[
                        RuntimeError("Timeout"),
                        self._make_mock_extraction("chunk_001", success=True),
                    ]
                )
                MockPipeline.return_value = pipeline_instance

                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                ) as pool:
                    result = await pool.extract_batch(chunks=chunks)

        # Both chunks processed — one via error path, one normally
        assert result.total_chunks == 2

    @pytest.mark.asyncio
    async def test_extract_batch_without_context_manager_raises(self):
        """Calling extract_batch without entering context raises RuntimeError."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        openai_svc, tokenizer = self._make_mock_services()
        pool = ExtractionWorkerPool(
            openai_service=openai_svc,
            tokenizer=tokenizer,
        )

        with pytest.raises(RuntimeError, match="context manager"):
            await pool.extract_batch(chunks=[MagicMock()])

    def test_cancel_sets_cancelled_flag(self):
        from app.workers.extraction_worker import ExtractionWorkerPool

        openai_svc, tokenizer = self._make_mock_services()
        pool = ExtractionWorkerPool(openai_service=openai_svc, tokenizer=tokenizer)
        assert pool.is_cancelled is False
        pool.cancel()
        assert pool.is_cancelled is True

    @pytest.mark.asyncio
    async def test_gleaning_rounds_zero_skips_gleaning_loop(self):
        """gleaning_rounds=0 means no GleaningLoop is created."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop") as MockGleaning:
                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                    gleaning_rounds=0,
                ) as pool:
                    pass
                # GleaningLoop should NOT have been instantiated
                MockGleaning.assert_not_called()
                # ExtractionPipeline called with gleaning_loop=None
                call_kwargs = MockPipeline.call_args[1]
                assert call_kwargs.get("gleaning_loop") is None

    @pytest.mark.asyncio
    async def test_sync_callback_also_supported(self):
        """Synchronous callbacks (non-async) are also supported."""
        from app.workers.extraction_worker import ExtractionWorkerPool

        chunks = [self._make_mock_chunk("chunk_sync")]
        sync_calls = []

        def sync_callback(extraction):
            sync_calls.append(extraction.chunk_id)

        openai_svc, tokenizer = self._make_mock_services()
        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                pipeline_instance.extract_chunk = AsyncMock(
                    return_value=self._make_mock_extraction("chunk_sync")
                )
                MockPipeline.return_value = pipeline_instance

                async with ExtractionWorkerPool(
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                ) as pool:
                    await pool.extract_batch(chunks=chunks, on_chunk_complete=sync_callback)

        assert "chunk_sync" in sync_calls


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH SPLITTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchSplitter:

    def test_split_into_batches_even(self):
        from app.workers.extraction_worker import split_into_batches
        items = list(range(9))
        batches = split_into_batches(items, batch_size=3)
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2]
        assert batches[2] == [6, 7, 8]

    def test_split_into_batches_uneven(self):
        from app.workers.extraction_worker import split_into_batches
        items = list(range(10))
        batches = split_into_batches(items, batch_size=3)
        assert len(batches) == 4
        assert batches[-1] == [9]  # last batch has 1 item

    def test_split_into_batches_single_item(self):
        from app.workers.extraction_worker import split_into_batches
        batches = split_into_batches(["only"], batch_size=10)
        assert len(batches) == 1
        assert batches[0] == ["only"]

    def test_split_into_batches_empty(self):
        from app.workers.extraction_worker import split_into_batches
        batches = split_into_batches([], batch_size=5)
        assert batches == []

    def test_split_into_batches_zero_size_raises(self):
        from app.workers.extraction_worker import split_into_batches
        with pytest.raises(ValueError, match="batch_size"):
            split_into_batches([1, 2, 3], batch_size=0)

    def test_split_into_batches_preserves_order(self):
        from app.workers.extraction_worker import split_into_batches
        items = list(range(100))
        batches = split_into_batches(items, batch_size=10)
        flat = [item for batch in batches for item in batch]
        assert flat == items


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunExtractionWorkers:

    @pytest.mark.asyncio
    async def test_run_extraction_workers_success(self):
        """run_extraction_workers() convenience function works end-to-end."""
        from app.workers.extraction_worker import run_extraction_workers

        openai_svc = MagicMock()
        tokenizer  = MagicMock()
        chunks     = [MagicMock() for _ in range(3)]

        mock_extraction = MagicMock()
        mock_extraction.chunk_id = "chunk_x"
        mock_extraction.extraction_completed = True
        mock_extraction.entities = [MagicMock()]
        mock_extraction.relationships = [MagicMock()]

        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                pipeline_instance.extract_chunk = AsyncMock(
                    return_value=mock_extraction
                )
                MockPipeline.return_value = pipeline_instance

                result = await run_extraction_workers(
                    chunks=chunks,
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                    gleaning_rounds=1,
                    max_concurrency=3,
                )

        assert result.total_chunks == 3
        assert result.successful == 3

    @pytest.mark.asyncio
    async def test_run_extraction_workers_with_callback(self):
        """Callback is forwarded through the convenience function."""
        from app.workers.extraction_worker import run_extraction_workers

        openai_svc = MagicMock()
        tokenizer  = MagicMock()
        chunks = [MagicMock()]
        calls = []

        def cb(extraction):
            calls.append(1)

        mock_extraction = MagicMock()
        mock_extraction.extraction_completed = True
        mock_extraction.entities = []
        mock_extraction.relationships = []

        with patch("app.workers.extraction_worker.ExtractionPipeline") as MockPipeline:
            with patch("app.workers.extraction_worker.GleaningLoop"):
                pipeline_instance = AsyncMock()
                pipeline_instance.extract_chunk = AsyncMock(return_value=mock_extraction)
                MockPipeline.return_value = pipeline_instance

                await run_extraction_workers(
                    chunks=chunks,
                    openai_service=openai_svc,
                    tokenizer=tokenizer,
                    on_chunk_complete=cb,
                )

        assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES INTEGRATION — refactored thin route still passes all Stage 7 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutesIndexingIntegration:
    """
    Verify that routes_indexing.py still works correctly after being
    refactored to delegate to indexing_worker.py.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        return TestClient(create_app(), raise_server_exceptions=True)

    def _headers(self):
        from app.config import get_settings
        return {"X-API-Key": get_settings().api_key}

    def setup_method(self):
        _reset_worker_state()

    def teardown_method(self):
        _reset_worker_state()

    def test_post_index_returns_202(self, client):
        response = client.post(
            "/api/v1/index",
            json={"force_reindex": False},
            headers=self._headers(),
        )
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"].startswith("idx_")
        assert data["status"] == "queued"

    def test_post_index_conflict_when_running(self, client):
        from app.workers.indexing_worker import make_blank_stages
        import app.workers.indexing_worker as iw

        iw._ACTIVE_JOB_ID = "idx_fakerunning"
        iw._JOB_STORE["idx_fakerunning"] = {
            "status": PipelineStatus.RUNNING,
            "stages": make_blank_stages(),
        }
        try:
            response = client.post(
                "/api/v1/index",
                json={"force_reindex": False},
                headers=self._headers(),
            )
            assert response.status_code == 409
        finally:
            _reset_worker_state()

    def test_get_index_status_no_jobs(self, client):
        response = client.get("/api/v1/index/status", headers=self._headers())
        assert response.status_code == 200
        assert response.json()["job_id"] == "none"

    def test_get_index_status_known_job(self, client):
        from app.workers.indexing_worker import submit_indexing_job, make_blank_stages
        import app.workers.indexing_worker as iw

        job_id = submit_indexing_job(make_index_request())
        iw._JOB_STORE[job_id]["status"] = PipelineStatus.COMPLETED
        iw._JOB_STORE[job_id]["total_chunks"] = 200

        response = client.get(
            f"/api/v1/index/status?job_id={job_id}",
            headers=self._headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"
        assert data["total_chunks"] == 200

    def test_get_index_status_unknown_returns_404(self, client):
        response = client.get(
            "/api/v1/index/status?job_id=idx_doesnotexist",
            headers=self._headers(),
        )
        assert response.status_code == 404

    def test_invalid_chunk_overlap_returns_422(self, client):
        response = client.post(
            "/api/v1/index",
            json={"chunk_size": 600, "chunk_overlap": 700},  # overlap >= size
            headers=self._headers(),
        )
        assert response.status_code == 422

    def test_post_index_missing_auth_returns_401(self, client):
        response = client.post("/api/v1/index", json={})
        assert response.status_code == 401

    def test_get_status_returns_most_recent_without_job_id(self, client):
        from app.workers.indexing_worker import submit_indexing_job
        import app.workers.indexing_worker as iw

        job_id = submit_indexing_job(make_index_request())
        iw._ACTIVE_JOB_ID = None  # job not running

        response = client.get("/api/v1/index/status", headers=self._headers())
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id