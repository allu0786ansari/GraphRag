"""
dependencies.py — FastAPI dependency injection container.

All shared dependencies are defined here and injected into route handlers
via FastAPI's Depends() mechanism. This keeps route handlers thin and
makes testing easy — mock the dependency, not the entire service.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Settings dependency ────────────────────────────────────────────────────────
def get_app_settings() -> Settings:
    """Inject application settings. Uses cached singleton."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


# ── API Key authentication ─────────────────────────────────────────────────────
async def verify_api_key(
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """
    Validate the X-API-Key header on protected endpoints.

    Raises HTTP 401 if the key is missing or incorrect.
    In development mode, authentication is still enforced so
    the frontend can be built and tested against real auth.

    Usage in routes:
        @router.post("/query")
        async def query(
            request: QueryRequest,
            _: Annotated[None, Depends(verify_api_key)],
        ):
            ...
    """
    if x_api_key is None:
        log.warning("Request missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": "X-API-Key header is required.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key != settings.api_key:
        log.warning("Request has invalid X-API-Key", provided_key_prefix=x_api_key[:8] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_api_key",
                "message": "The provided API key is not valid.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )


AuthDep = Annotated[None, Depends(verify_api_key)]


# ── Pagination ─────────────────────────────────────────────────────────────────
class PaginationParams:
    """Standard pagination query parameters for list endpoints."""

    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> None:
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "invalid_pagination", "message": "page must be >= 1"},
            )
        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "invalid_pagination",
                    "message": "page_size must be between 1 and 100",
                },
            )
        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size


PaginationDep = Annotated[PaginationParams, Depends(PaginationParams)]