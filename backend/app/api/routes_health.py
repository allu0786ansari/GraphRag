"""
api/routes_health.py — Health check endpoints.

Provides two endpoints:
  GET /api/v1/health        — liveness probe (is the process alive?)
  GET /api/v1/health/ready  — readiness probe (can it serve traffic?)

Kubernetes, Docker, and load balancers use these to decide whether
to route traffic to this instance.

Liveness:  simple ping — returns 200 if the process is running.
Readiness: deeper check — verifies config loaded, dirs exist, artifacts
           are present (later stages will add more checks here).
"""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# Record process start time for uptime calculation
_START_TIME = time.time()


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the application process is alive.",
    tags=["health"],
)
async def liveness() -> JSONResponse:
    """
    Liveness probe — lightweight, no I/O.
    Should never fail unless the process itself is broken.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "alive",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - _START_TIME, 1),
        },
    )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 if the application is ready to serve traffic. "
        "Checks configuration, storage paths, and artifact availability."
    ),
    tags=["health"],
)
async def readiness() -> JSONResponse:
    """
    Readiness probe — checks whether the app can actually serve requests.

    Checks performed:
      - Settings loaded correctly
      - Required storage directories exist
      - Pipeline artifacts status (present / missing)

    Returns 200 with full status report.
    Returns 503 if any critical dependency is unavailable.
    """
    settings = get_settings()
    checks: dict[str, dict] = {}
    all_healthy = True

    # ── Check 1: Configuration ────────────────────────────────────────────────
    try:
        _ = settings.app_name
        _ = settings.openai_model
        checks["configuration"] = {"status": "ok", "environment": settings.app_env}
    except Exception as exc:
        checks["configuration"] = {"status": "error", "detail": str(exc)}
        all_healthy = False

    # ── Check 2: Storage directories ──────────────────────────────────────────
    dir_checks = {}
    dirs_to_check = {
        "artifacts": settings.artifacts_dir,
        "raw_data":  settings.raw_data_dir,
        "evaluation": settings.evaluation_dir,
        "logs":      settings.logs_dir,
    }
    for name, path in dirs_to_check.items():
        exists = Path(path).exists()
        dir_checks[name] = {
            "path": str(path),
            "exists": exists,
        }
        if not exists:
            all_healthy = False

    checks["storage_directories"] = {"status": "ok" if all_healthy else "error", "dirs": dir_checks}

    # ── Check 3: Pipeline artifacts ───────────────────────────────────────────
    artifact_files = {
        "chunks":               settings.artifacts_dir / "chunks.json",
        "extractions":          settings.artifacts_dir / "extractions.json",
        "graph":                settings.artifacts_dir / "graph.pkl",
        "community_map":        settings.artifacts_dir / "community_map.json",
        "community_summaries":  settings.artifacts_dir / "community_summaries.json",
        "faiss_index":          settings.faiss_index_path,
    }
    artifact_status = {}
    for name, path in artifact_files.items():
        p = Path(path)
        artifact_status[name] = {
            "present": p.exists(),
            "size_mb": round(p.stat().st_size / 1_048_576, 2) if p.exists() else None,
        }

    indexed = artifact_status["community_summaries"]["present"]
    checks["pipeline_artifacts"] = {
        "status": "ok" if indexed else "not_indexed",
        "indexed": indexed,
        "artifacts": artifact_status,
    }

    # ── Compose response ──────────────────────────────────────────────────────
    http_status = 200 if all_healthy else 503

    if not all_healthy:
        log.warning("Readiness check failed", checks=checks)
    else:
        log.debug("Readiness check passed")

    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "version": settings.app_version,
            "environment": settings.app_env,
            "python": platform.python_version(),
            "checks": checks,
        },
    )