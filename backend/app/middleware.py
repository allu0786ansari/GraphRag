"""
middleware.py — FastAPI middleware stack.

Middleware executes in reverse registration order (last registered = outermost).
Registration order in main.py should be:
  1. ErrorHandlingMiddleware   (registered last → runs first)
  2. RequestLoggingMiddleware
  3. CORSMiddleware
  4. RateLimitMiddleware       (registered first → runs last)

Each incoming request gets:
  - A unique X-Request-ID header (generated if not provided by client)
  - Structured access logging with timing
  - CORS headers
  - Rate limit enforcement
  - Consistent JSON error responses for all unhandled exceptions
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.utils.logger import get_logger, set_request_id, get_request_id

log = get_logger(__name__)


# ── Request ID + Logging Middleware ───────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    For every request:
      1. Extract or generate a unique request ID.
      2. Store it in the async context var so all log lines carry it.
      3. Add X-Request-ID to the response headers.
      4. Log request start and completion with timing.
    """

    def __init__(self, app: ASGIApp, *, exclude_paths: list[str] | None = None) -> None:
        super().__init__(app)
        # Paths that skip verbose access logging (e.g. health checks)
        self.exclude_paths = set(exclude_paths or ["/api/v1/health", "/metrics"])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ── 1. Resolve request ID ──────────────────────────────────────────────
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        set_request_id(request_id)

        # ── 2. Log incoming request ────────────────────────────────────────────
        start_time = time.perf_counter()
        path = request.url.path
        verbose = path not in self.exclude_paths

        if verbose:
            log.info(
                "Request started",
                method=request.method,
                path=path,
                query=str(request.query_params),
                client=request.client.host if request.client else "unknown",
                request_id=request_id,
            )

        # ── 3. Process request ─────────────────────────────────────────────────
        response = await call_next(request)

        # ── 4. Add headers and log completion ─────────────────────────────────
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        if verbose:
            log.info(
                "Request completed",
                method=request.method,
                path=path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                request_id=request_id,
            )

        return response


# ── Global Error Handling Middleware ──────────────────────────────────────────
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Catch any unhandled exception that escapes route handlers and return
    a consistent JSON error response.

    This is the last line of defence. Route-level errors should be handled
    by the exception handlers registered in main.py first.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = get_request_id()
            log.exception(
                "Unhandled exception in request pipeline",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred. Please try again.",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )


# ── Rate Limit Middleware ──────────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory sliding-window rate limiter per client IP.

    For production with multiple workers, replace the in-memory store
    with Redis using the cache_manager (Stage 4).

    Excluded paths (health checks) are never rate-limited.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool = True,
        max_requests: int = 60,
        window_seconds: int = 60,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exclude_paths = set(exclude_paths or ["/api/v1/health", "/docs", "/openapi.json"])
        # client_ip → list of request timestamps
        self._store: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled or request.url.path in self.exclude_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds

        # Evict timestamps outside the current window
        timestamps = self._store.get(client_ip, [])
        timestamps = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - timestamps[0]))
            log.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                requests_in_window=len(timestamps),
                limit=self.max_requests,
                retry_after=retry_after,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Retry after {retry_after} seconds.",
                    "request_id": get_request_id(),
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                },
            )

        # Record this request
        timestamps.append(now)
        self._store[client_ip] = timestamps

        response = await call_next(request)

        # Attach rate limit headers to all responses
        remaining = max(0, self.max_requests - len(timestamps))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.window_seconds))

        return response


def register_middleware(app, settings) -> None:
    """
    Register all middleware on the FastAPI app.

    Called once from main.py during application startup.
    Middleware executes in reverse order of registration.
    """
    # ── Rate limiting (executes last) ─────────────────────────────────────────
    app.add_middleware(
        RateLimitMiddleware,
        enabled=settings.rate_limit_enabled,
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time", "X-RateLimit-Limit"],
    )

    # ── Request logging + request ID injection ────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── Global error handler (executes first) ─────────────────────────────────
    app.add_middleware(ErrorHandlingMiddleware)