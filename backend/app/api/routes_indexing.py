"""
api/routes_indexing.py — Indexing pipeline endpoints.

Endpoints:
  POST /api/v1/index          Accept an indexing job, run it in the background.
  GET  /api/v1/index/status   Poll the status of a running or completed job.

Design:
  - The pipeline is long-running (tens of minutes for large corpora), so the
    POST endpoint returns immediately with a job_id and HTTP 202 Accepted.
  - The background task updates a shared in-memory _JOB_STORE dict.
  - In production you'd replace _JOB_STORE with Redis or a DB; for this stage
    in-memory is correct because we're single-process.
  - Only one pipeline job runs at a time; submitting a second job while one is
    running returns HTTP 409 Conflict.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from app.config import get_settings
from app.dependencies import AuthDep
from app.models.request_models import IndexRequest
from app.models.response_models import (
    IndexResponse,
    IndexStatusResponse,
    PipelineStage,
    PipelineStatus,
    StageProgress,
    TokenUsage,
)
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# ── In-memory job store (single-process; replace with Redis in prod) ───────────

_JOB_STORE: dict[str, dict[str, Any]] = {}
_ACTIVE_JOB_ID: str | None = None   # only one pipeline runs at a time

_STAGE_ORDER: list[str] = [
    "chunking",
    "extraction",
    "gleaning",
    "graph_construction",
    "community_detection",
    "summarization",
]


def _make_blank_stages() -> list[StageProgress]:
    return [
        StageProgress(
            stage=PipelineStage(s),
            status=PipelineStatus.QUEUED,
        )
        for s in _STAGE_ORDER
    ]


def _update_stage(job: dict, stage_name: str, **kwargs: Any) -> None:
    """Mutate a single StageProgress inside the job dict."""
    for sp in job["stages"]:
        if sp.stage.value == stage_name:
            for k, v in kwargs.items():
                setattr(sp, k, v)
            return


# ── Background task ────────────────────────────────────────────────────────────

async def _run_pipeline_background(job_id: str, request: IndexRequest) -> None:
    """
    Execute the full indexing pipeline as a background task.

    Updates _JOB_STORE[job_id] in-place so /index/status can poll it.
    """
    global _ACTIVE_JOB_ID
    job = _JOB_STORE[job_id]

    try:
        job["status"]     = PipelineStatus.RUNNING
        job["started_at"] = datetime.now(tz=timezone.utc)

        # ── Build runner ───────────────────────────────────────────────────────
        from app.core.pipeline.pipeline_runner import PipelineRunner
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

        # ── Progress callback — called by runner as each stage completes ───────
        def on_progress(stage_name: str, pct: float) -> None:
            _update_stage(
                job, stage_name,
                status=PipelineStatus.RUNNING if pct < 1.0 else PipelineStatus.COMPLETED,
                progress_pct=round(pct * 100, 1),
                completed_at=datetime.now(tz=timezone.utc) if pct >= 1.0 else None,
            )

        # Mark all stages as running sequentially (best we can do without
        # deeper hooks — real hooks live in the pipeline runner itself)
        for stage_name in _STAGE_ORDER:
            _update_stage(
                job, stage_name,
                status=PipelineStatus.RUNNING,
                started_at=datetime.now(tz=timezone.utc),
            )
            break  # only mark the first one as started now

        # ── Run ────────────────────────────────────────────────────────────────
        result = await runner.run(
            force_reindex=request.force_reindex,
            max_chunks=request.max_chunks,
            on_progress=on_progress,
        )

        # ── Persist result into job dict ───────────────────────────────────────
        job["completed_at"]    = datetime.now(tz=timezone.utc)
        job["elapsed_seconds"] = result.total_elapsed_seconds

        if result.success:
            job["status"]             = PipelineStatus.COMPLETED
            job["total_chunks"]       = result.chunks_count
            job["total_nodes"]        = result.graph_nodes
            job["total_edges"]        = result.graph_edges
            job["total_communities"]  = {}           # populated below
            job["total_summaries"]    = result.summaries_count

            # Populate community counts per level from the store
            try:
                from app.storage.graph_store import GraphStore
                gs = GraphStore(artifacts_dir=settings.artifacts_dir)
                job["total_communities"] = gs.get_community_counts()
            except Exception:
                pass

            # Mark any remaining stages as completed
            for sp in job["stages"]:
                if sp.status in (PipelineStatus.QUEUED, PipelineStatus.RUNNING):
                    sp.status = PipelineStatus.COMPLETED
                    sp.completed_at = datetime.now(tz=timezone.utc)

        else:
            job["status"]        = PipelineStatus.FAILED
            job["error_message"] = result.error_message

            # Mark the failed stage
            if result.error_stage:
                _update_stage(
                    job, result.error_stage,
                    status=PipelineStatus.FAILED,
                    error_message=result.error_message,
                )

    except Exception as exc:
        log.error("Pipeline background task crashed", job_id=job_id, error=str(exc))
        job["status"]        = PipelineStatus.FAILED
        job["error_message"] = str(exc)
        job["completed_at"]  = datetime.now(tz=timezone.utc)

    finally:
        _ACTIVE_JOB_ID = None
        log.info(
            "Pipeline job finished",
            job_id=job_id,
            status=job["status"].value,
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/index",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IndexResponse,
    responses={
        202: {"description": "Job accepted and queued."},
        409: {"description": "Another indexing job is already running."},
    },
    summary="Start indexing pipeline",
    description=(
        "Trigger the full offline indexing pipeline on the raw corpus in "
        "`data/raw/`. Runs asynchronously — poll `GET /api/v1/index/status` "
        "for progress. Only one pipeline job can run at a time."
    ),
)
async def start_indexing(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    _auth: AuthDep,
) -> IndexResponse:
    """
    Accept an indexing job and queue it as a background task.

    Returns HTTP 202 immediately with a job_id.
    Returns HTTP 409 if another job is already running.
    """
    global _ACTIVE_JOB_ID

    # Reject if already running
    if _ACTIVE_JOB_ID and _JOB_STORE.get(_ACTIVE_JOB_ID, {}).get("status") == PipelineStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "job_already_running",
                "message": (
                    f"Indexing job '{_ACTIVE_JOB_ID}' is already running. "
                    "Wait for it to complete or check /api/v1/index/status."
                ),
            },
        )

    job_id = f"idx_{uuid.uuid4().hex[:8]}"
    accepted_at = datetime.now(tz=timezone.utc)

    # Initialise job record
    _JOB_STORE[job_id] = {
        "job_id":          job_id,
        "status":          PipelineStatus.QUEUED,
        "current_stage":   PipelineStage.PENDING,
        "stages":          _make_blank_stages(),
        "started_at":      None,
        "completed_at":    None,
        "elapsed_seconds": None,
        "total_chunks":    None,
        "total_nodes":     None,
        "total_edges":     None,
        "total_communities": None,
        "total_summaries": None,
        "token_usage":     None,
        "error_message":   None,
    }
    _ACTIVE_JOB_ID = job_id

    background_tasks.add_task(_run_pipeline_background, job_id, request)

    log.info("Indexing job accepted", job_id=job_id, force_reindex=request.force_reindex)

    return IndexResponse(
        job_id=job_id,
        status=PipelineStatus.QUEUED,
        message=(
            f"Indexing job accepted and queued as '{job_id}'. "
            "Poll GET /api/v1/index/status?job_id={job_id} for progress."
        ),
        accepted_at=accepted_at,
        estimated_duration_minutes=60,
    )


@router.get(
    "/index/status",
    response_model=IndexStatusResponse,
    responses={
        200: {"description": "Job status."},
        404: {"description": "Job ID not found."},
    },
    summary="Poll indexing job status",
    description=(
        "Returns real-time progress of a running, completed, or failed "
        "indexing job. Pass `job_id` to query a specific job, or omit it "
        "to get the most recent job."
    ),
)
async def get_index_status(
    _auth: AuthDep,
    job_id: Annotated[str | None, Query(
        description="Job ID returned by POST /api/v1/index. Omit for most recent job.",
        examples=["idx_a1b2c3d4"],
    )] = None,
) -> IndexStatusResponse:
    """Return the current status of an indexing job."""
    # Resolve which job to report on
    if job_id:
        if job_id not in _JOB_STORE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "job_not_found",
                    "message": f"No indexing job found with id '{job_id}'.",
                },
            )
        job = _JOB_STORE[job_id]
    elif _JOB_STORE:
        # Return the most recent job
        job_id = list(_JOB_STORE.keys())[-1]
        job = _JOB_STORE[job_id]
    else:
        # No jobs ever submitted — synthesise a "not started" response
        return IndexStatusResponse(
            job_id="none",
            status=PipelineStatus.QUEUED,
            current_stage=PipelineStage.PENDING,
            stages=_make_blank_stages(),
            error_message="No indexing jobs have been submitted yet.",
        )

    # Determine current active stage
    current_stage = PipelineStage.PENDING
    for sp in job["stages"]:
        if sp.status == PipelineStatus.RUNNING:
            current_stage = sp.stage
            break
    if job["status"] == PipelineStatus.COMPLETED:
        current_stage = PipelineStage.COMPLETED
    elif job["status"] == PipelineStatus.FAILED:
        current_stage = PipelineStage.FAILED

    return IndexStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        current_stage=current_stage,
        stages=job["stages"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        elapsed_seconds=job["elapsed_seconds"],
        total_chunks=job["total_chunks"],
        total_nodes=job["total_nodes"],
        total_edges=job["total_edges"],
        total_communities=job["total_communities"],
        total_summaries=job["total_summaries"],
        token_usage=job["token_usage"],
        error_message=job["error_message"],
    )