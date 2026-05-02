"""
utils/retry.py — Retry decorators for LLM and embedding API calls.

Updated for Gemini backend: catches the same openai SDK exceptions
(since we use the openai SDK with Gemini's OpenAI-compatible endpoint,
the exception types are identical — openai.RateLimitError etc.)
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
    before_sleep_log,
)

from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Exception sets ─────────────────────────────────────────────────────────────
# We still use openai SDK (via Gemini's OpenAI-compatible endpoint),
# so the exception classes are from the openai package.

try:
    import openai

    _TRANSIENT_EXCEPTIONS: tuple[Type[Exception], ...] = (
        openai.RateLimitError,       # 429 — hit free tier RPM limit, back off
        openai.APITimeoutError,      # network timeout
        openai.APIConnectionError,   # network error
        openai.InternalServerError,  # 5xx from Gemini (includes 503 overloaded)
        openai.APIStatusError,       # catch-all for any other 5xx status codes
    )
    _FATAL_EXCEPTIONS: tuple[Type[Exception], ...] = (
        openai.AuthenticationError,  # bad API key — don't retry
        openai.PermissionDeniedError,
        openai.NotFoundError,        # wrong model name
        openai.BadRequestError,      # malformed request
    )
except ImportError:
    _TRANSIENT_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
    _FATAL_EXCEPTIONS = (ValueError,)

try:
    from google.api_core import exceptions as google_exceptions

    _TRANSIENT_EXCEPTIONS += (
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
    )
except ImportError:
    pass


# ── Core retry factory ────────────────────────────────────────────────────────

def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[Type[Exception], ...] | None = None,
) -> Callable:
    """
    Decorator factory. Retries on transient errors with exponential backoff.

    For Gemini free tier: when you hit 429 (15 RPM limit), tenacity backs off
    and retries automatically. min_wait=4s ensures we don't immediately
    hammer the rate limit again.
    """
    if exceptions is None:
        exceptions = _TRANSIENT_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        def _log_retry_attempt(retry_state: Any) -> None:
            log.warning(
                "Retrying after error",
                attempt=retry_state.attempt_number,
                error=str(retry_state.outcome.exception()),
                wait=round(retry_state.next_action.sleep, 1) if retry_state.next_action else 0,
            )

        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper

        return wrapper

    return decorator


# ── Pre-built retry configs ───────────────────────────────────────────────────

# Standard LLM calls — 3 attempts, 4–60s backoff
# min_wait=4s because Gemini free tier resets RPM every 60s;
# backing off 4s is enough to clear a short burst
llm_retry = with_retry(
    max_attempts=5,       # more retries for 503 high-demand errors
    min_wait=4.0,
    max_wait=90.0,        # longer max wait gives Gemini time to recover
    exceptions=_TRANSIENT_EXCEPTIONS,
)

# Bulk map-stage calls — 5 attempts, longer backoff
bulk_llm_retry = with_retry(
    max_attempts=5,
    min_wait=5.0,
    max_wait=120.0,
    exceptions=_TRANSIENT_EXCEPTIONS,
)

# Embedding calls — 3 attempts, shorter wait (embeddings are faster)
embedding_retry = with_retry(
    max_attempts=3,
    min_wait=2.0,
    max_wait=30.0,
    exceptions=_TRANSIENT_EXCEPTIONS,
)

__all__ = [
    "with_retry",
    "llm_retry",
    "bulk_llm_retry",
    "embedding_retry",
    "RetryError",
]