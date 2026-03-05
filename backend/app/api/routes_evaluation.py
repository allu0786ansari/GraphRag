"""
api/routes_evaluation.py — Evaluation endpoints.

Endpoints:
  POST /api/v1/evaluate                 LLM-as-judge evaluation (paper Section 4.1).
  GET  /api/v1/evaluation/results       List stored evaluation results.
  GET  /api/v1/evaluation/results/{id}  Retrieve a single evaluation result.

Design:
  - Evaluation is long-running (minutes for 125 questions × 4 criteria × 5 runs),
    so it runs as a background task exactly like the indexing pipeline.
  - Results are persisted to `data/evaluation/<eval_id>.json` for retrieval.
  - The in-memory _EVAL_STORE caches active/recent results; completed ones are
    also written to disk so they survive restarts.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.config import get_settings
from app.dependencies import AuthDep
from app.models.evaluation_models import EvalResponse
from app.models.request_models import EvalRequest
from app.models.response_models import PipelineStatus
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# ── In-memory store for evaluation jobs ───────────────────────────────────────

_EVAL_STORE: dict[str, dict[str, Any]] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _eval_results_dir() -> Path:
    """Return the directory where evaluation JSON files are persisted."""
    settings = get_settings()
    path = Path(settings.evaluation_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persist_result(eval_id: str, result: EvalResponse) -> None:
    """Write an EvalResponse to disk as JSON."""
    out_path = _eval_results_dir() / f"{eval_id}.json"
    try:
        out_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        log.info("Evaluation result persisted", eval_id=eval_id, path=str(out_path))
    except Exception as exc:
        log.warning("Could not persist evaluation result", eval_id=eval_id, error=str(exc))


def _load_result_from_disk(eval_id: str) -> EvalResponse | None:
    """Load an EvalResponse from disk. Returns None if not found."""
    out_path = _eval_results_dir() / f"{eval_id}.json"
    if not out_path.exists():
        return None
    try:
        return EvalResponse.model_validate_json(out_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load evaluation result", eval_id=eval_id, error=str(exc))
        return None


# ── Background task ────────────────────────────────────────────────────────────

async def _run_evaluation_background(eval_id: str, request: EvalRequest) -> None:
    """
    Run the full LLM-as-judge evaluation pipeline as a background task.

    Updates _EVAL_STORE[eval_id] in-place and persists the result to disk.
    """
    job = _EVAL_STORE[eval_id]
    try:
        job["status"] = PipelineStatus.RUNNING

        from app.core.query.evaluation_engine import EvaluationEngine
        engine = EvaluationEngine.from_settings()

        result: EvalResponse = await engine.evaluate(
            questions=request.questions,
            criteria=[c.value for c in request.criteria],
            community_level=request.community_level.value,
            eval_runs=request.eval_runs,
            randomize_answer_order=request.randomize_answer_order,
        )

        job["status"] = PipelineStatus.COMPLETED
        job["result"] = result
        _persist_result(eval_id, result)

    except Exception as exc:
        log.error("Evaluation background task failed", eval_id=eval_id, error=str(exc))
        job["status"]        = PipelineStatus.FAILED
        job["error_message"] = str(exc)
        job["completed_at"]  = datetime.now(tz=timezone.utc)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Evaluation accepted and queued."},
        503: {"description": "Corpus not indexed."},
    },
    summary="Run LLM-as-judge evaluation",
    description=(
        "Implements Experiment 1 from Edge et al. (2025): submit a list of "
        "global sensemaking questions; both GraphRAG and VectorRAG answer each "
        "one, then a judge LLM scores them across four criteria (comprehensiveness, "
        "diversity, empowerment, directness). "
        "Runs asynchronously — use `GET /api/v1/evaluation/results/{eval_id}` "
        "to retrieve results when done."
    ),
)
async def start_evaluation(
    request: EvalRequest,
    background_tasks: BackgroundTasks,
    _auth: AuthDep,
) -> dict:
    """
    Accept an evaluation job and queue it as a background task.

    Returns HTTP 202 immediately with an eval_id.
    """
    # Validate index exists
    settings = get_settings()
    from app.storage.summary_store import SummaryStore
    summary_store = SummaryStore(artifacts_dir=settings.artifacts_dir)
    if not summary_store.summaries_exist():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "not_indexed",
                "message": (
                    "The corpus has not been indexed yet. "
                    "Run POST /api/v1/index first."
                ),
            },
        )

    eval_id     = f"eval_{uuid.uuid4().hex[:8]}"
    accepted_at = datetime.now(tz=timezone.utc)

    _EVAL_STORE[eval_id] = {
        "eval_id":     eval_id,
        "status":      PipelineStatus.QUEUED,
        "accepted_at": accepted_at,
        "completed_at": None,
        "result":      None,
        "error_message": None,
        "questions_count": len(request.questions),
        "criteria":    [c.value for c in request.criteria],
        "eval_runs":   request.eval_runs,
    }

    background_tasks.add_task(_run_evaluation_background, eval_id, request)

    log.info(
        "Evaluation job accepted",
        eval_id=eval_id,
        questions=len(request.questions),
        criteria=[c.value for c in request.criteria],
        eval_runs=request.eval_runs,
    )

    return {
        "eval_id":    eval_id,
        "status":     "queued",
        "accepted_at": accepted_at.isoformat(),
        "questions_count": len(request.questions),
        "message": (
            f"Evaluation job '{eval_id}' queued. "
            f"Poll GET /api/v1/evaluation/results/{eval_id} for results."
        ),
    }


@router.get(
    "/evaluation/results",
    responses={
        200: {"description": "List of evaluation result summaries."},
    },
    summary="List evaluation results",
    description=(
        "Returns a paginated list of all evaluation runs (in-memory + on-disk). "
        "Each entry includes the eval_id, status, question count, and headline "
        "win rates. Use the eval_id to retrieve full results."
    ),
)
async def list_evaluation_results(
    _auth: AuthDep,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 20,
) -> dict:
    """Return a paginated summary of all evaluation runs."""
    results_dir = _eval_results_dir()

    # Collect from disk (persisted past results)
    on_disk: list[dict] = []
    for json_file in sorted(results_dir.glob("eval_*.json"), reverse=True):
        eval_id = json_file.stem
        if eval_id in _EVAL_STORE and _EVAL_STORE[eval_id].get("result"):
            # Already in memory — will be included below
            continue
        try:
            res = EvalResponse.model_validate_json(json_file.read_text(encoding="utf-8"))
            on_disk.append({
                "eval_id":                    res.evaluation_id,
                "status":                     "completed",
                "total_questions":            res.total_questions,
                "criteria_evaluated":         [c.value for c in res.criteria_evaluated],
                "eval_runs":                  res.eval_runs_per_question,
                "comprehensiveness_win_rate": res.comprehensiveness_win_rate,
                "diversity_win_rate":         res.diversity_win_rate,
                "empowerment_win_rate":       res.empowerment_win_rate,
                "directness_win_rate":        res.directness_win_rate,
                "started_at":                 res.started_at.isoformat() if res.started_at else None,
                "completed_at":               res.completed_at.isoformat() if res.completed_at else None,
                "duration_seconds":           res.duration_seconds,
            })
        except Exception:
            pass

    # Collect from memory (in-progress + recent)
    in_memory: list[dict] = []
    for eval_id, job in reversed(list(_EVAL_STORE.items())):
        entry: dict = {
            "eval_id":          eval_id,
            "status":           job["status"].value,
            "total_questions":  job["questions_count"],
            "criteria_evaluated": job["criteria"],
            "eval_runs":        job["eval_runs"],
            "accepted_at":      job["accepted_at"].isoformat(),
            "error_message":    job.get("error_message"),
        }
        if job.get("result"):
            r = job["result"]
            entry.update({
                "comprehensiveness_win_rate": r.comprehensiveness_win_rate,
                "diversity_win_rate":         r.diversity_win_rate,
                "empowerment_win_rate":       r.empowerment_win_rate,
                "directness_win_rate":        r.directness_win_rate,
                "started_at":                 r.started_at.isoformat() if r.started_at else None,
                "completed_at":               r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds":           r.duration_seconds,
            })
        in_memory.append(entry)

    all_results = in_memory + on_disk
    total       = len(all_results)
    start       = (page - 1) * page_size
    end         = start + page_size

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "results":   all_results[start:end],
    }


@router.get(
    "/evaluation/results/{eval_id}",
    response_model=EvalResponse,
    responses={
        200: {"description": "Full evaluation result."},
        202: {"description": "Evaluation still in progress."},
        404: {"description": "Evaluation ID not found."},
    },
    summary="Get evaluation result by ID",
    description=(
        "Retrieve the full LLM-as-judge evaluation result for a given eval_id. "
        "Returns HTTP 202 with a status message if the evaluation is still running."
    ),
)
async def get_evaluation_result(
    eval_id: str,
    _auth: AuthDep,
) -> EvalResponse:
    """Return full evaluation results for a given eval_id."""

    # Check in-memory store first
    if eval_id in _EVAL_STORE:
        job = _EVAL_STORE[eval_id]

        if job["status"] == PipelineStatus.RUNNING or job["status"] == PipelineStatus.QUEUED:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail={
                    "error":   "evaluation_in_progress",
                    "eval_id": eval_id,
                    "status":  job["status"].value,
                    "message": "Evaluation is still running. Try again in a few minutes.",
                },
            )

        if job["status"] == PipelineStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error":         "evaluation_failed",
                    "eval_id":       eval_id,
                    "error_message": job.get("error_message"),
                },
            )

        if job.get("result"):
            return job["result"]

    # Try loading from disk
    result = _load_result_from_disk(eval_id)
    if result:
        return result

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error":   "eval_not_found",
            "message": f"No evaluation result found with id '{eval_id}'.",
        },
    )