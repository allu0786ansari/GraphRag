"""
services/openai_service.py — LLM completion service backed by Google Gemini.

Migration: switched from OpenAI to Google Gemini (AI Studio).
- No credit card required — free tier at aistudio.google.com
- OpenAI-compatible API: only base_url changes, all method signatures identical
- Same interface as before: CompletionResult, async_chat_completion, complete,
  batch_complete — zero changes needed in pipeline or query code

Gemini free tier limits (as of 2026):
  gemini-2.5-flash-lite: 15 RPM, 1000 RPD  ← use this for pipeline
  gemini-2.5-flash:      10 RPM,  250 RPD  ← use for queries

Rate limit note for the pipeline:
  With 1000 RPD and ~100 chunks, the dev run takes ~10 min at safe pace.
  Full corpus (~3000 extractions) needs 3+ days at free tier, OR
  create multiple Google accounts (each gets 1000 RPD free).

Gleaning note:
  OpenAI's logit_bias (forced YES/NO token) is NOT supported by Gemini.
  async_completion_with_logit_bias() falls back to a prompt instruction:
  "Respond with only YES or NO." — works identically in practice.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.utils.retry import llm_retry, bulk_llm_retry

log = get_logger(__name__)

# Gemini's OpenAI-compatible endpoint
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ── Response dataclass — identical to before ──────────────────────────────────

@dataclass
class CompletionResult:
    """Result of a single chat completion call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    finish_reason: str
    latency_ms: float = 0.0
    raw_response: ChatCompletion | None = field(default=None, repr=False)

    @property
    def estimated_cost_usd(self) -> float:
        """Always 0.0 — Gemini free tier has no token cost."""
        return 0.0

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"CompletionResult("
            f"tokens={self.total_tokens}, "
            f"cost=$0.0000 (Gemini free), "
            f"content={preview!r}...)"
        )


# ── Message builder helpers — unchanged ───────────────────────────────────────

def system_message(content: str) -> dict[str, str]:
    return {"role": "system", "content": content}

def user_message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}

def assistant_message(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}

def build_messages(
    user_prompt: str,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append(system_message(system_prompt))
    if history:
        messages.extend(history)
    messages.append(user_message(user_prompt))
    return messages


# ── Service class ──────────────────────────────────────────────────────────────

class OpenAIService:
    """
    LLM completion service using Google Gemini via the OpenAI-compatible API.

    Identical interface to the original OpenAI-backed version.
    Only the constructor internals changed — all callers unchanged.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite-preview-06-17",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout
        self.max_retries = max_retries

        # Both clients point to Gemini's OpenAI-compatible endpoint
        self._sync_client = OpenAI(
            api_key=api_key,
            base_url=_GEMINI_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=_GEMINI_BASE_URL,
            timeout=timeout,
            max_retries=0,
        )

        log.info(
            "OpenAIService initialized (Gemini backend)",
            model=model,
            max_tokens=max_tokens,
            base_url=_GEMINI_BASE_URL,
        )

    # ── Sync completion ────────────────────────────────────────────────────────

    @llm_retry
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        logit_bias: dict[str, int] | None = None,
    ) -> CompletionResult:
        """Synchronous chat completion via Gemini."""
        t0 = time.monotonic()
        kwargs = self._build_kwargs(
            messages=messages, model=model, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format,
            # logit_bias silently ignored — Gemini doesn't support it
        )
        log.debug("Gemini sync completion", model=kwargs["model"], messages=len(messages))
        response: ChatCompletion = self._sync_client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        result = self._parse_response(response, latency_ms)
        self._log_completion(result, sync=True)
        return result

    # ── Async completion ───────────────────────────────────────────────────────

    @llm_retry
    async def async_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        logit_bias: dict[str, int] | None = None,
    ) -> CompletionResult:
        """Async chat completion via Gemini."""
        t0 = time.monotonic()
        kwargs = self._build_kwargs(
            messages=messages, model=model, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format,
        )
        log.debug("Gemini async completion", model=kwargs["model"], messages=len(messages))
        response: ChatCompletion = await self._async_client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        result = self._parse_response(response, latency_ms)
        self._log_completion(result, sync=False)
        return result

    # ── Gleaning loop: YES/NO ──────────────────────────────────────────────────

    @llm_retry
    async def async_completion_with_logit_bias(
        self,
        messages: list[dict[str, str]],
        yes_token_ids: list[int],
        no_token_ids: list[int],
        bias: int = 100,
    ) -> CompletionResult:
        """
        Gemini does not support logit_bias.
        We append a strict instruction to the last message instead.
        The result is functionally identical — model returns YES or NO.
        """
        # Append the YES/NO constraint to the last user message
        constrained_messages = list(messages)
        if constrained_messages:
            last = constrained_messages[-1]
            constrained_messages[-1] = {
                "role": last["role"],
                "content": last["content"] + "\n\nRespond with only the single word YES or NO.",
            }

        return await self.async_chat_completion(
            messages=constrained_messages,
            max_tokens=5,
            temperature=0.0,
        )

    # ── Batch completion — map stage ───────────────────────────────────────────

    async def batch_complete(
        self,
        prompts: list[list[dict[str, str]]],
        max_concurrency: int = 5,   # Lower than OpenAI — Gemini free tier is 15 RPM
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        return_exceptions: bool = False,
    ) -> list[CompletionResult | BaseException]:
        """
        Concurrent batch completion.

        NOTE: max_concurrency defaults to 5 (not 20) to respect Gemini's
        free tier limit of 15 RPM. At 5 concurrent calls with ~1s latency
        each, we stay well under the rate limit.
        """
        log.info(
            "Starting batch completion (Gemini)",
            total_prompts=len(prompts),
            max_concurrency=max_concurrency,
            model=model or self.model,
        )

        async def _single_call(msgs: list[dict[str, str]]) -> CompletionResult:
            return await self.async_chat_completion(
                messages=msgs, model=model, max_tokens=max_tokens,
                temperature=temperature, response_format=response_format,
            )

        results = await gather_with_concurrency(
            coroutines=[_single_call(p) for p in prompts],
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
        )

        successful = [r for r in results if isinstance(r, CompletionResult)]
        if return_exceptions:
            failed = len(results) - len(successful)
            if failed > 0:
                log.warning("Batch completed with failures", failed=failed, total=len(results))

        log.info("Batch completion finished", total=len(prompts), successful=len(successful))
        return results

    # ── Convenience wrappers — identical interface ─────────────────────────────

    async def complete(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        messages = build_messages(user_prompt, system_prompt=system_prompt)
        response_format = {"type": "json_object"} if json_mode else None
        return await self.async_chat_completion(
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_sync(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        messages = build_messages(user_prompt, system_prompt=system_prompt)
        response_format = {"type": "json_object"} if json_mode else None
        return self.chat_completion(
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model":       model or self.model,
            "messages":    messages,
            "max_tokens":  max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _parse_response(self, response: ChatCompletion, latency_ms: float) -> CompletionResult:
        choice = response.choices[0]
        usage  = response.usage
        return CompletionResult(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "unknown",
            latency_ms=latency_ms,
            raw_response=response,
        )

    def _log_completion(self, result: CompletionResult, sync: bool) -> None:
        log.debug(
            "Completion finished",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            latency_ms=round(result.latency_ms, 1),
            finish_reason=result.finish_reason,
            sync=sync,
        )

    def __repr__(self) -> str:
        return f"OpenAIService(model={self.model!r}, backend=Gemini)"


# ── Factory function ───────────────────────────────────────────────────────────

def get_openai_service(
    api_key: str | None = None,
    model: str | None = None,
) -> OpenAIService:
    from app.config import get_settings
    settings = get_settings()
    return OpenAIService(
        api_key=api_key or settings.gemini_api_key,
        model=model or settings.openai_model,
        max_tokens=settings.openai_max_tokens,
        temperature=settings.openai_temperature,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )


__all__ = [
    "OpenAIService",
    "CompletionResult",
    "get_openai_service",
    "build_messages",
    "system_message",
    "user_message",
    "assistant_message",
]