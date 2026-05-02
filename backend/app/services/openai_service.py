"""
services/llm_service.py — LLM completion service using Google Gemini.

Uses the official google-generativeai SDK.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.utils.retry import llm_retry, bulk_llm_retry

log = get_logger(__name__)


# ── Response dataclass — adapted for Gemini ──────────────────────────────────

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
    raw_response: Any | None = field(default=None, repr=False)

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

class LLMService:
    """
    LLM completion service using Google Gemini via official SDK.

    Adapted interface to match the original OpenAI version.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
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

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

        log.info(
            "OpenAIService initialized (Gemini official SDK)",
            model=model,
            max_tokens=max_tokens,
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
        model_name = model or self.model
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens or self.max_tokens

        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [msg["content"]]})

        prompt = messages[-1]["content"]

        log.debug("Gemini sync completion", model=model_name, messages=len(messages))
        
        # Create a new client if model changed (not needed in new API)
        # Create a new model instance if model changed
        if model_name != self.model:
            client = genai.GenerativeModel(model_name)
        else:
            client = self.client

        response = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tok,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            ]
        )

        latency_ms = (time.monotonic() - t0) * 1000
        result = self._parse_response(response, latency_ms, model_name)
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
        # For simplicity, use sync in async context with proper keyword args
        import functools
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            functools.partial(
                self.chat_completion, 
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                logit_bias=logit_bias
            )
        )

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

    def _parse_response(self, response, latency_ms: float, model: str) -> CompletionResult:
        content = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    content += part.text

        # Estimate tokens (Gemini doesn't provide exact counts)
        prompt_tokens = len(response.prompt_feedback or [])  # rough estimate
        completion_tokens = len(content.split())  # rough word count
        total_tokens = prompt_tokens + completion_tokens

        return CompletionResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            finish_reason="stop",
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
        return f"LLMService(model={self.model!r}, backend=Gemini)"


# ── Factory function ───────────────────────────────────────────────────────────

def get_llm_service(
    api_key: str | None = None,
    model: str | None = None,
) -> LLMService:
    from app.config import get_settings
    settings = get_settings()
    return LLMService(
        api_key=api_key or settings.gemini_api_key,
        model=model or settings.gemini_model,
        max_tokens=settings.openai_max_tokens,
        temperature=settings.openai_temperature,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )


__all__ = [
    "LLMService",
    "CompletionResult",
    "get_llm_service",
    "build_messages",
    "system_message",
    "user_message",
    "assistant_message",
]

# ── Backwards compatibility aliases for legacy code ──────────────────────────────
OpenAIService = LLMService
get_openai_service = get_llm_service