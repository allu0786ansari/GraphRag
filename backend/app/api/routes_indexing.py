"""
api/routes_indexing.py — Indexing pipeline endpoints.

Endpoints:
  POST /api/v1/index          Accept an indexing job, run it in the background.
  GET  /api/v1/index/status   Poll the status of a running or completed job.

Design:
  - All job-store logic lives in workers/indexing_worker.py.
  - This file is a thin HTTP adapter: validate request -> submit_indexing_job()
    -> return 202.  No pipeline logic here.
  - The background task runs workers.indexing_worker.run_indexing_job().
  - Only one pipeline job runs at a time; submitting a second while one is
    running returns HTTP 409 Conflict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.dependencies import AuthDep
from app.models.request_models import IndexRequest
from app.models.response_models import (
    IndexResponse,
    IndexStatusResponse,
    PipelineStage,
    PipelineStatus,
)
from app.utils.logger import get_logger
from app.workers.indexing_worker import (
    current_stage_of,
    get_active_job_id,
    get_job,
    get_most_recent_job,
    is_job_running,
    make_blank_stages,
    run_indexing_job,
    submit_indexing_job,
)

log = get_logger(__name__)
router = APIRouter()


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
    if is_job_running():
        active_id = get_active_job_id()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "job_already_running",
                "message": (
                    f"Indexing job '{active_id}' is already running. "
                    "Wait for it to complete or check /api/v1/index/status."
                ),
            },
        )

    try:
        job_id = submit_indexing_job(request)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "job_already_running",
                "message": str(exc),
            },
        )

    accepted_at = datetime.now(tz=timezone.utc)
    background_tasks.add_task(run_indexing_job, job_id, request)

    log.info("Indexing job accepted", job_id=job_id, force_reindex=request.force_reindex)

    return IndexResponse(
        job_id=job_id,
        status=PipelineStatus.QUEUED,
        message=(
            f"Indexing job '{job_id}' accepted and queued. "
            f"Poll GET /api/v1/index/status?job_id={job_id} for progress."
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
    if job_id:
        job = get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "job_not_found",
                    "message": f"No indexing job found with id '{job_id}'.",
                },
            )
    else:
        job = get_most_recent_job()
        if job is None:
            return IndexStatusResponse(
                job_id="none",
                status=PipelineStatus.QUEUED,
                current_stage=PipelineStage.PENDING,
                stages=make_blank_stages(),
                error_message="No indexing jobs have been submitted yet.",
            )

    return IndexStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        current_stage=current_stage_of(job),
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