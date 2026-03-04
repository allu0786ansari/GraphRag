"""
utils/logger.py — Structured logging for GraphRAG backend.

Clean, production-safe version using Loguru native JSON serialization.
No custom formatter to avoid KeyError issues.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger


# ─────────────────────────────────────────────────────
# Async-safe request context
# ─────────────────────────────────────────────────────

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_pipeline_stage_var: ContextVar[str] = ContextVar("pipeline_stage", default="-")


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()


def set_pipeline_stage(stage: str) -> None:
    _pipeline_stage_var.set(stage)


def get_pipeline_stage() -> str:
    return _pipeline_stage_var.get()


# ─────────────────────────────────────────────────────
# Development text format
# ─────────────────────────────────────────────────────

_TEXT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>req={extra[request_id]}</cyan> | "
    "<yellow>stage={extra[pipeline_stage]}</yellow> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


# ─────────────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────────────

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",   # "json" or "text"
    log_rotation: str = "100 MB",
    log_retention: str = "30 days",
    logs_dir: Path = Path("./logs"),
) -> None:

    logger.remove()

    # Inject default extra fields
    logger.configure(
        extra={
            "request_id": "-",
            "pipeline_stage": "-",
        }
    )

    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "app_{time:YYYY-MM-DD}.log"
    error_log_file = logs_dir / "error_{time:YYYY-MM-DD}.log"

    # ─── Console Handler ──────────────────────────
    if log_format.lower() == "json":
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,   # Native JSON
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level,
            format=_TEXT_FORMAT,
            colorize=True,
            enqueue=True,
        )

    # ─── Main File (JSON structured) ─────────────
    logger.add(
        str(log_file),
        level=log_level,
        serialize=True,
        rotation=log_rotation,
        retention=log_retention,
        compression="gz",
        enqueue=True,
        encoding="utf-8",
    )

    # ─── Error File ──────────────────────────────
    logger.add(
        str(error_log_file),
        level="ERROR",
        serialize=True,
        rotation=log_rotation,
        retention=log_retention,
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        encoding="utf-8",
    )

    logger.info(
        "Logging initialised",
        log_level=log_level,
        log_format=log_format,
        logs_dir=str(logs_dir),
    )


# ─────────────────────────────────────────────────────
# Logger accessor
# ─────────────────────────────────────────────────────

def get_logger(name: str = "graphrag"):
    return logger.bind(name=name)