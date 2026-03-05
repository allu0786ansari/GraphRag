"""
main.py — FastAPI application factory.

This module creates and configures the FastAPI application:
  - Lifespan context (startup + shutdown logic)
  - Middleware registration
  - Router registration
  - Exception handlers
  - Health check endpoint

All application-level concerns live here.
Pipeline and query logic live in core/.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.middleware import register_middleware
from app.utils.logger import get_logger, setup_logging, get_request_id

settings = get_settings()

# ── Logging must be set up before anything else ───────────────────────────────
setup_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_rotation=settings.log_rotation,
    log_retention=settings.log_retention,
    logs_dir=settings.logs_dir,
)

log = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Everything before `yield` runs on startup.
    Everything after `yield` runs on shutdown.

    Startup tasks:
      - Validate critical config (API keys present, directories exist)
      - Log startup summary
      - (Later stages) warm up LLM clients, load FAISS index, etc.

    Shutdown tasks:
      - Flush log buffers
      - (Later stages) close DB connections, clean up temp files, etc.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    log.info(
        "Application starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        log_level=settings.log_level,
        log_format=settings.log_format,
    )

    # Verify critical directories exist (config validator creates them,
    # but log here so startup is observable)
    log.info(
        "Storage paths verified",
        artifacts_dir=str(settings.artifacts_dir),
        raw_data_dir=str(settings.raw_data_dir),
        evaluation_dir=str(settings.evaluation_dir),
        logs_dir=str(settings.logs_dir),
    )

    log.info(
        "Pipeline configuration",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        gleaning_rounds=settings.gleaning_rounds,
        context_window_size=settings.context_window_size,
        community_level=settings.community_level,
        evaluation_runs=settings.evaluation_runs,
    )

    log.info("Application startup complete — ready to serve requests")

    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    log.info("Application shutting down")
    log.info("Shutdown complete")


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a fully configured app instance ready for uvicorn.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "GraphRAG vs Vector RAG — Global Question Answering System. "
            "Implements Edge et al. (2025) 'From Local to Global: A GraphRAG "
            "Approach to Query-Focused Summarization'."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters — see middleware.py) ─────────────────────────
    register_middleware(app, settings)

    # ── Exception handlers ────────────────────────────────────────────────────
    _register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """
    Register all API routers.

    Routes are added here as each stage is completed.
    The health router is always present from Stage 1.
    """
    from app.api.routes_health     import router as health_router
    from app.api.routes_indexing   import router as indexing_router
    from app.api.routes_query      import router as query_router
    from app.api.routes_evaluation import router as evaluation_router
    from app.api.routes_graph      import router as graph_router

    app.include_router(health_router,     prefix="/api/v1", tags=["health"])
    app.include_router(indexing_router,   prefix="/api/v1", tags=["indexing"])
    app.include_router(query_router,      prefix="/api/v1", tags=["query"])
    app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation"])
    app.include_router(graph_router,      prefix="/api/v1", tags=["graph"])


def _register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers for consistent error responses."""

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "not_found",
                "message": f"The requested path '{request.url.path}' was not found.",
                "request_id": get_request_id(),
            },
        )

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={
                "error": "method_not_allowed",
                "message": f"Method '{request.method}' is not allowed on this endpoint.",
                "request_id": get_request_id(),
            },
        )

    @app.exception_handler(422)
    async def validation_error_handler(request: Request, exc: Any) -> JSONResponse:
        log.warning(
            "Request validation failed",
            path=request.url.path,
            detail=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed.",
                "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
                "request_id": get_request_id(),
            },
        )


# ── Application instance ──────────────────────────────────────────────────────
app = create_app()