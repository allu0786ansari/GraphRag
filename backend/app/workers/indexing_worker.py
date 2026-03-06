"""
workers/indexing_worker.py — Background indexing pipeline worker.

Responsibilities:
  1. Maintain the in-memory job store (_JOB_STORE) — the source of truth for
     all indexing job states.
  2. Provide submit_indexing_job() — called by routes_indexing to enqueue work.
  3. Run the full PipelineRunner.run() coroutine as a FastAPI BackgroundTask,
     updating the job store in real-time so GET /index/status reflects live
     progress at sub-stage granularity.
  4. Enforce single-job-at-a-time semantics (one active pipeline per process).

Architecture note — why in-memory?
  Single uvicorn process (workers=1) means a plain dict is safe and has zero
  overhead.  In a multi-process or multi-node deployment, replace _JOB_STORE
  with a Redis hash and use asyncio.Lock for CAS on _ACTIVE_JOB_ID.  The
  public API surface (submit_indexing_job / get_job / list_jobs) is the same
  regardless of backing store.

Progress tracking:
  PipelineRunner.run() accepts an on_progress(stage_name, pct_complete)
  callback.  The extraction and summarization stages call it after every chunk/
  community completes, so status updates are near-continuous during the two
  longest stages.  The other four stages (chunking, graph construction,
  community detection, embedding) call it once at 100% because they're
  single-threaded operations.

Thread safety:
  All mutations happen inside the single asyncio event loop — no threads, no
  locking required.  The FastAPI BackgroundTask scheduler ensures the pipeline
  coroutine runs concurrently with HTTP handlers on the same loop.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from app.models.request_models import IndexRequest
from app.models.response_models import (
    PipelineStage,
    PipelineStatus,
    StageProgress,
    TokenUsage,
)
from app.utils.logger import get_logger

# Imported at module level so unittest.mock.patch() can intercept them.
# patch("app.workers.indexing_worker.get_settings") etc. all work correctly.
from app.config import get_settings
from app.core.pipeline.pipeline_runner import PipelineRunner
from app.storage.graph_store import GraphStore

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

STAGE_ORDER: list[str] = [
    "chunking",
    "extraction",
    "gleaning",
    "graph_construction",
    "community_detection",
    "summarization",
]

# ── Job store ──────────────────────────────────────────────────────────────────

# Keyed by job_id → job dict.  Public read; mutations via helpers below.
_JOB_STORE: dict[str, dict[str, Any]] = {}

# ID of the currently running job, or None.  Only one pipeline runs at a time.
_ACTIVE_JOB_ID: str | None = None


# ── Job record helpers ─────────────────────────────────────────────────────────

def make_blank_stages() -> list[StageProgress]:
    """Create a fresh StageProgress list — all stages queued, no progress."""
    return [
        StageProgress(
            stage=PipelineStage(s),
            status=PipelineStatus.QUEUED,
        )
        for s in STAGE_ORDER
    ]


def _update_stage(job: dict, stage_name: str, **kwargs: Any) -> None:
    """
    Mutate a single StageProgress entry inside a job dict.

    Finds the entry whose stage.value matches stage_name and applies
    all keyword arguments as attribute updates.  No-op if not found.
    """
    for sp in job["stages"]:
        if sp.stage.value == stage_name:
            for k, v in kwargs.items():
                setattr(sp, k, v)
            return
    log.warning("_update_stage: stage not found", stage_name=stage_name)


def _new_job_record(job_id: str) -> dict[str, Any]:
    """Return a freshly initialised job dict for the given job_id."""
    return {
        "job_id":            job_id,
        "status":            PipelineStatus.QUEUED,
        "stages":            make_blank_stages(),
        "started_at":        None,
        "completed_at":      None,
        "elapsed_seconds":   None,
        "total_chunks":      None,
        "total_nodes":       None,
        "total_edges":       None,
        "total_communities": None,
        "total_summaries":   None,
        "token_usage":       None,
        "error_message":     None,
    }


# ── Public query API ───────────────────────────────────────────────────────────

def get_job(job_id: str) -> dict[str, Any] | None:
    """Return the job dict for job_id, or None if not found."""
    return _JOB_STORE.get(job_id)


def get_active_job_id() -> str | None:
    """Return the ID of the currently running job, or None."""
    return _ACTIVE_JOB_ID


def is_job_running() -> bool:
    """True if a pipeline job is currently executing."""
    if _ACTIVE_JOB_ID is None:
        return False
    job = _JOB_STORE.get(_ACTIVE_JOB_ID, {})
    return job.get("status") == PipelineStatus.RUNNING


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    """
    Return a list of job dicts, most recent first.

    Args:
        limit: Maximum number of jobs to return.  Default: 50.
    """
    all_jobs = list(_JOB_STORE.values())
    return list(reversed(all_jobs))[:limit]


def get_most_recent_job() -> dict[str, Any] | None:
    """Return the most recently submitted job dict, or None."""
    if not _JOB_STORE:
        return None
    return _JOB_STORE[list(_JOB_STORE.keys())[-1]]


def current_stage_of(job: dict[str, Any]) -> PipelineStage:
    """
    Derive the current active PipelineStage from a job dict.

    Scans the stages list for the first RUNNING stage.
    Falls back to COMPLETED or FAILED based on overall job status.
    """
    for sp in job["stages"]:
        if sp.status == PipelineStatus.RUNNING:
            return sp.stage

    status = job["status"]
    if status == PipelineStatus.COMPLETED:
        return PipelineStage.COMPLETED
    if status == PipelineStatus.FAILED:
        return PipelineStage.FAILED
    return PipelineStage.PENDING


# ── Submission ─────────────────────────────────────────────────────────────────

def submit_indexing_job(request: IndexRequest) -> str:
    """
    Register a new indexing job in the job store and return its job_id.

    Does NOT start the background task — that's the caller's responsibility
    (FastAPI BackgroundTasks.add_task).  This separation makes the function
    fully synchronous and trivially testable.

    Raises:
        RuntimeError: If a pipeline job is already running.

    Returns:
        The new job_id string (e.g. "idx_a1b2c3d4").
    """
    global _ACTIVE_JOB_ID

    if is_job_running():
        raise RuntimeError(
            f"Indexing job '{_ACTIVE_JOB_ID}' is already running. "
            "Wait for it to complete before submitting a new job."
        )

    job_id = f"idx_{uuid.uuid4().hex[:8]}"
    _JOB_STORE[job_id] = _new_job_record(job_id)
    _ACTIVE_JOB_ID = job_id

    log.info(
        "Indexing job registered",
        job_id=job_id,
        force_reindex=request.force_reindex,
        max_chunks=request.max_chunks,
    )
    return job_id


# ── Background coroutine ───────────────────────────────────────────────────────

async def run_indexing_job(job_id: str, request: IndexRequest) -> None:
    """
    Execute the full indexing pipeline as a FastAPI BackgroundTask.

    This coroutine is designed to be passed to BackgroundTasks.add_task().
    It mutates _JOB_STORE[job_id] in place throughout execution so the
    GET /index/status endpoint always reflects live state.

    Progress granularity:
      - Extraction: updated after every chunk (N callbacks for N chunks).
      - Summarization: updated after every community summary (M callbacks).
      - All other stages: updated once at 100% completion.

    Failure handling:
      - If PipelineRunner.run() returns success=False, the job transitions
        to FAILED and the failing stage is marked accordingly.
      - Any unexpected exception is caught, logged, and recorded in the job.
      - _ACTIVE_JOB_ID is always cleared in the finally block.
    """
    global _ACTIVE_JOB_ID

    job = _JOB_STORE[job_id]

    try:
        job["status"]     = PipelineStatus.RUNNING
        job["started_at"] = datetime.now(tz=timezone.utc)

        log.info("Indexing pipeline starting", job_id=job_id)

        # ── Build PipelineRunner from settings + request overrides ─────────────
        settings = get_settings()

        runner = PipelineRunner(
            raw_data_dir=settings.raw_data_dir,
            artifacts_dir=settings.artifacts_dir,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            gleaning_rounds=request.gleaning_rounds,
            context_window=request.context_window_size,
            max_concurrency=20,
            community_max_levels=request.max_community_levels,
            skip_claims=request.skip_claims,
        )

        # ── Stage tracking state ───────────────────────────────────────────────
        # We track which stage is "active" so we can mark transitions cleanly.
        _current_stage: list[str] = ["chunking"]  # mutable via closure

        def _mark_stage_started(stage_name: str) -> None:
            """Mark a stage as RUNNING and record its start time."""
            _update_stage(
                job, stage_name,
                status=PipelineStatus.RUNNING,
                started_at=datetime.now(tz=timezone.utc),
                progress_pct=0.0,
            )
            _current_stage[0] = stage_name

        def _mark_stage_done(stage_name: str) -> None:
            """Mark a stage as COMPLETED."""
            _update_stage(
                job, stage_name,
                status=PipelineStatus.COMPLETED,
                progress_pct=100.0,
                completed_at=datetime.now(tz=timezone.utc),
            )

        # Mark chunking as started immediately (first stage)
        _mark_stage_started("chunking")

        # ── on_progress callback ──────────────────────────────────────────────
        # The runner calls on_progress(stage_name, pct) where pct ∈ [0.0, 1.0].
        # pct == 1.0 means the stage just finished.
        # pct < 1.0 means a sub-step completed (e.g. one chunk extracted).

        def on_progress(stage_name: str, pct: float) -> None:
            """
            Receive progress updates from PipelineRunner and write to job store.

            Called from the pipeline's async context on the same event loop,
            so direct dict mutation is safe — no locking needed.
            """
            now = datetime.now(tz=timezone.utc)

            # If we've moved to a new stage, mark the previous one done first
            if stage_name != _current_stage[0]:
                _mark_stage_done(_current_stage[0])
                _mark_stage_started(stage_name)

            if pct >= 1.0:
                # Stage completed
                _update_stage(
                    job, stage_name,
                    status=PipelineStatus.COMPLETED,
                    progress_pct=100.0,
                    completed_at=now,
                )
            else:
                # Sub-step progress within a stage
                _update_stage(
                    job, stage_name,
                    status=PipelineStatus.RUNNING,
                    progress_pct=round(pct * 100.0, 1),
                )

        # ── Run the pipeline ───────────────────────────────────────────────────
        result = await runner.run(
            force_reindex=request.force_reindex,
            max_chunks=request.max_chunks,
            on_progress=on_progress,
        )

        # ── Persist result ─────────────────────────────────────────────────────
        job["completed_at"]    = datetime.now(tz=timezone.utc)
        job["elapsed_seconds"] = result.total_elapsed_seconds

        if result.success:
            job["status"]          = PipelineStatus.COMPLETED
            job["total_chunks"]    = result.chunks_count
            job["total_nodes"]     = result.graph_nodes
            job["total_edges"]     = result.graph_edges
            job["total_summaries"] = result.summaries_count

            # Load community counts per level from the persisted community map
            try:
                gs = GraphStore(artifacts_dir=settings.artifacts_dir)
                job["total_communities"] = gs.get_community_counts()
            except Exception as exc:
                log.warning("Could not load community counts after pipeline", error=str(exc))
                job["total_communities"] = {}

            # Mark any stages still queued/running as completed
            # (handles skipped stages — resumed from checkpoint)
            for sp in job["stages"]:
                if sp.status in (PipelineStatus.QUEUED, PipelineStatus.RUNNING):
                    sp.status = PipelineStatus.COMPLETED
                    sp.completed_at = datetime.now(tz=timezone.utc)
                    sp.progress_pct = 100.0

            log.info(
                "Indexing pipeline completed",
                job_id=job_id,
                chunks=result.chunks_count,
                nodes=result.graph_nodes,
                edges=result.graph_edges,
                summaries=result.summaries_count,
                elapsed_seconds=result.total_elapsed_seconds,
            )

        else:
            # Pipeline returned gracefully but with a failure in one stage
            job["status"]        = PipelineStatus.FAILED
            job["error_message"] = result.error_message

            if result.error_stage:
                _update_stage(
                    job, result.error_stage,
                    status=PipelineStatus.FAILED,
                    error_message=result.error_message,
                    completed_at=datetime.now(tz=timezone.utc),
                )

            log.error(
                "Indexing pipeline failed",
                job_id=job_id,
                error_stage=result.error_stage,
                error_message=result.error_message,
                elapsed_seconds=result.total_elapsed_seconds,
            )

    except Exception as exc:
        # Unexpected crash (not a graceful pipeline failure)
        log.error(
            "Indexing worker crashed unexpectedly",
            job_id=job_id,
            error=str(exc),
            exc_info=True,
        )
        job["status"]        = PipelineStatus.FAILED
        job["error_message"] = f"Unexpected error: {exc}"
        job["completed_at"]  = datetime.now(tz=timezone.utc)

        # Mark the last known active stage as failed
        if _JOB_STORE.get(job_id):
            for sp in job["stages"]:
                if sp.status == PipelineStatus.RUNNING:
                    sp.status = PipelineStatus.FAILED
                    sp.error_message = str(exc)
                    sp.completed_at = datetime.now(tz=timezone.utc)
                    break

    finally:
        _ACTIVE_JOB_ID = None
        log.info(
            "Indexing worker finished",
            job_id=job_id,
            final_status=job.get("status", "unknown"),
        )