"""
utils/retry.py — Retry decorators for LLM and external API calls.

Uses tenacity for robust retry logic with:
- Exponential backoff with jitter
- Specific exception targeting (rate limits, timeouts, transient errors)
- Logging on each retry attempt
- Configurable max attempts and wait times
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Type

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    before_sleep_log,
    after_log,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Loguru-compatible logging bridge for tenacity ─────────────────────────────
import logging as _stdlib_logging

_tenacity_logger = _stdlib_logging.getLogger("tenacity")


# ── Exception sets ─────────────────────────────────────────────────────────────
try:
    import openai

    _OPENAI_TRANSIENT_EXCEPTIONS: tuple[Type[Exception], ...] = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
    _OPENAI_FATAL_EXCEPTIONS: tuple[Type[Exception], ...] = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.BadRequestError,
    )
except ImportError:
    # openai not yet installed — fallback to generic exceptions
    _OPENAI_TRANSIENT_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
    _OPENAI_FATAL_EXCEPTIONS = (ValueError,)


# ── Core retry decorator factory ───────────────────────────────────────────────
def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[Type[Exception], ...] | None = None,
    reraise: bool = True,
) -> Callable:
    """
    Decorator factory for retrying functions with exponential backoff + jitter.

    Args:
        max_attempts: Total attempts including the first call.
        min_wait:     Minimum seconds to wait between retries.
        max_wait:     Maximum seconds to wait between retries.
        exceptions:   Exception types that trigger a retry.
                      Defaults to transient OpenAI exceptions.
        reraise:      Re-raise the last exception after exhausting retries.

    Usage:
        @with_retry(max_attempts=3)
        async def call_openai(prompt: str) -> str:
            ...
    """
    if exceptions is None:
        exceptions = _OPENAI_TRANSIENT_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=_log_retry_attempt,
            reraise=reraise,
        )
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=_log_retry_attempt,
            reraise=reraise,
        )
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Return the correct wrapper based on whether the function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _log_retry_attempt(retry_state) -> None:
    """Log each retry attempt with context about the failure."""
    attempt = retry_state.attempt_number
    exception = retry_state.outcome.exception()
    wait = getattr(retry_state.next_action, "sleep", 0)

    log.warning(
        "Retrying after error",
        attempt=attempt,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        wait_seconds=round(wait, 2),
    )


# ── Pre-configured decorators for common cases ─────────────────────────────────

# Standard LLM call retry: 3 attempts, 1–60s backoff
llm_retry = with_retry(
    max_attempts=3,
    min_wait=1.0,
    max_wait=60.0,
    exceptions=_OPENAI_TRANSIENT_EXCEPTIONS,
)

# Aggressive retry for bulk operations: 5 attempts, longer backoff
bulk_llm_retry = with_retry(
    max_attempts=5,
    min_wait=2.0,
    max_wait=120.0,
    exceptions=_OPENAI_TRANSIENT_EXCEPTIONS,
)

# Fast retry for embedding calls: shorter wait since they're cheaper
embedding_retry = with_retry(
    max_attempts=3,
    min_wait=0.5,
    max_wait=30.0,
    exceptions=_OPENAI_TRANSIENT_EXCEPTIONS,
)


__all__ = [
    "with_retry",
    "llm_retry",
    "bulk_llm_retry",
    "embedding_retry",
    "RetryError",
]