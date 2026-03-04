"""
services/openai_service.py — OpenAI chat completion service.

Every LLM call in the entire GraphRAG pipeline goes through this service.
Never import openai directly in pipeline or query code.

Responsibilities:
  - chat_completion()        : sync completion (used in tests and CLI)
  - async_chat_completion()  : async completion (used everywhere in pipeline)
  - async_completion_with_logit_bias() : gleaning yes/no forced responses
  - batch_complete()         : fire N prompts concurrently (map stage)
  - build_messages()         : helpers for constructing message arrays

All calls:
  - Go through llm_retry / embedding_retry from utils/retry.py
  - Log token usage to structured logger
  - Return both content and token usage together
  - Raise OpenAI exceptions directly — let retry handle transient ones

Paper-relevant methods:
  - async_completion_with_logit_bias() implements the gleaning loop's
    forced yes/no response (logit_bias = {YES_TOKEN: 100})
    from Section 3.1.2 of the paper.
  - batch_complete() implements the parallel map stage from Section 3.2.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.utils.retry import llm_retry, bulk_llm_retry

log = get_logger(__name__)


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class CompletionResult:
    """
    Result of a single chat completion call.

    Bundles the text content with token usage so callers never have to
    unpack the raw OpenAI response object.
    """
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    finish_reason: str
    latency_ms: float = 0.0
    # Raw response for advanced use (e.g. logprobs)
    raw_response: ChatCompletion | None = field(default=None, repr=False)

    @property
    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate at GPT-4o pricing (as of 2024-11).
        Input: $2.50/1M tokens. Output: $10.00/1M tokens.
        Use for logging/budgeting only — not billing.
        """
        input_cost  = (self.prompt_tokens     / 1_000_000) * 2.50
        output_cost = (self.completion_tokens / 1_000_000) * 10.00
        return round(input_cost + output_cost, 6)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"CompletionResult("
            f"tokens={self.total_tokens}, "
            f"cost=${self.estimated_cost_usd:.4f}, "
            f"content={preview!r}...)"
        )


# ── Message builder helpers ────────────────────────────────────────────────────

def system_message(content: str) -> dict[str, str]:
    """Build a system message dict."""
    return {"role": "system", "content": content}


def user_message(content: str) -> dict[str, str]:
    """Build a user message dict."""
    return {"role": "user", "content": content}


def assistant_message(content: str) -> dict[str, str]:
    """Build an assistant message dict (for few-shot / continuation)."""
    return {"role": "assistant", "content": content}


def build_messages(
    user_prompt: str,
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Build a well-formed messages array for the chat completions API.

    Args:
        user_prompt:   The user's message / task prompt.
        system_prompt: Optional system instruction. Added first if provided.
        history:       Optional list of prior messages for multi-turn context.
                       Inserted between system and user messages.

    Returns:
        List of message dicts ready for the OpenAI API.
    """
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
    Centralized OpenAI chat completion service.

    All LLM calls in the GraphRAG pipeline go through this class.
    Wraps both the sync and async OpenAI clients with:
      - Automatic retry on transient errors (rate limits, timeouts)
      - Structured logging of every call and its token usage
      - Consistent return type (CompletionResult)
      - Concurrency-controlled batch completion for the map stage

    Usage:
        service = OpenAIService(api_key="sk-...", model="gpt-4o")

        # Single async call
        result = await service.async_chat_completion(
            user_prompt="What are the main themes?",
            system_prompt="You are an analyst.",
        )
        print(result.content)

        # Batch (map stage)
        results = await service.batch_complete(prompts, max_concurrency=20)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            api_key:     OpenAI API key.
            model:       Model to use. Default: gpt-4o.
            max_tokens:  Max completion tokens. Default: 4096.
            temperature: Sampling temperature. Default: 0.0 (deterministic).
            timeout:     Request timeout in seconds.
            max_retries: Max retry attempts for transient errors.
        """
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout
        self.max_retries = max_retries

        # Sync client — for tests and CLI usage
        self._sync_client = OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves via tenacity
        )

        # Async client — for all pipeline code
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

        log.info(
            "OpenAIService initialized",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
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
        """
        Synchronous chat completion.

        Use this in tests, CLI scripts, and anywhere async is unavailable.
        For all pipeline code, prefer async_chat_completion().

        Args:
            messages:       List of message dicts (use build_messages()).
            model:          Override the default model.
            max_tokens:     Override max completion tokens.
            temperature:    Override sampling temperature.
            response_format: e.g. {"type": "json_object"} for JSON mode.
            logit_bias:     Token ID → bias mapping. Used in gleaning loop.

        Returns:
            CompletionResult with content and token usage.

        Raises:
            openai.RateLimitError:      → retried automatically
            openai.APITimeoutError:     → retried automatically
            openai.APIConnectionError:  → retried automatically
            openai.AuthenticationError: → NOT retried, raises immediately
        """
        t0 = time.monotonic()
        kwargs = self._build_kwargs(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            logit_bias=logit_bias,
        )

        log.debug("OpenAI sync completion", model=kwargs["model"], messages=len(messages))

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
        """
        Async chat completion — use this in all pipeline and query code.

        Identical interface to chat_completion() but uses the async client
        so it does not block the event loop during the HTTP request.

        Args:
            Same as chat_completion().

        Returns:
            CompletionResult with content and token usage.
        """
        t0 = time.monotonic()
        kwargs = self._build_kwargs(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            logit_bias=logit_bias,
        )

        log.debug("OpenAI async completion", model=kwargs["model"], messages=len(messages))

        response: ChatCompletion = await self._async_client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        result = self._parse_response(response, latency_ms)
        self._log_completion(result, sync=False)
        return result

    # ── Gleaning loop: forced yes/no response ──────────────────────────────────

    @llm_retry
    async def async_completion_with_logit_bias(
        self,
        messages: list[dict[str, str]],
        yes_token_ids: list[int],
        no_token_ids: list[int],
        bias: int = 100,
    ) -> CompletionResult:
        """
        Async completion with logit bias to force a yes/no response.

        Implements the gleaning loop's forced response mechanism from the paper
        (Section 3.1.2):

          "We prompt the LLM: 'Were all entities extracted?' with a logit_bias
           of 100 on the YES token to force a yes/no answer."

        This ensures the gleaning check produces exactly one token (YES or NO)
        rather than a free-form explanation, making it fast and parseable.

        Args:
            messages:      Conversation so far, ending with the gleaning question.
            yes_token_ids: tiktoken IDs for "yes" variants.
                           Typically: [9642] for "Yes", [4470] for "YES"
            no_token_ids:  tiktoken IDs for "no" variants.
                           Typically: [2201] for "No", [3458] for "NO"
            bias:          Logit bias magnitude. 100 = force these tokens exclusively.

        Returns:
            CompletionResult where content is "YES" or "NO".

        Example:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            yes_ids = enc.encode("YES")  # → [31794]
            no_ids  = enc.encode("NO")   # → [9173]

            result = await service.async_completion_with_logit_bias(
                messages=gleaning_messages,
                yes_token_ids=yes_ids,
                no_token_ids=no_ids,
            )
            missed_entities = result.content.strip().upper().startswith("Y")
        """
        logit_bias: dict[str, int] = {}
        for tid in yes_token_ids:
            logit_bias[str(tid)] = bias
        for tid in no_token_ids:
            logit_bias[str(tid)] = bias

        return await self.async_chat_completion(
            messages=messages,
            max_tokens=1,   # Only need 1 token for yes/no
            temperature=0.0,
            logit_bias=logit_bias,
        )

    # ── Batch completion — map stage ───────────────────────────────────────────

    async def batch_complete(
        self,
        prompts: list[list[dict[str, str]]],
        max_concurrency: int = 20,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        return_exceptions: bool = False,
    ) -> list[CompletionResult | BaseException]:
        """
        Fire N completion requests concurrently with a concurrency cap.

        Implements the map stage from the paper (Section 3.2):
          "Generate intermediate partial answers in parallel from each
           community summary context chunk."

        Using gather_with_concurrency() ensures we never exceed OpenAI's
        rate limits by capping the number of in-flight requests.

        Args:
            prompts:          List of message arrays (one per community chunk).
            max_concurrency:  Max concurrent API calls. Default: 20.
                              At 2M TPM, ~20 concurrent GPT-4o calls is safe.
            model:            Override model for this batch.
            max_tokens:       Override max tokens for this batch.
            temperature:      Override temperature.
            response_format:  Override response format.
            return_exceptions: If True, return exceptions instead of raising.
                               Use for fault-tolerant map stage.

        Returns:
            List of CompletionResult in the same order as prompts.
            If return_exceptions=True, failed items are BaseException instances.

        Example:
            # Map stage: generate partial answers for all community chunks
            results = await service.batch_complete(
                prompts=community_prompts,
                max_concurrency=20,
                return_exceptions=True,  # don't let one failure kill the batch
            )
            valid = [r for r in results if isinstance(r, CompletionResult)]
        """
        log.info(
            "Starting batch completion",
            total_prompts=len(prompts),
            max_concurrency=max_concurrency,
            model=model or self.model,
        )

        async def _single_call(messages: list[dict[str, str]]) -> CompletionResult:
            return await self.async_chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )

        results = await gather_with_concurrency(
            coroutines=[_single_call(p) for p in prompts],
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
        )

        # Log batch summary
        if not return_exceptions:
            successful = [r for r in results if isinstance(r, CompletionResult)]
        else:
            successful = [r for r in results if not isinstance(r, BaseException)]
            failed = len(results) - len(successful)
            if failed > 0:
                log.warning("Batch completed with failures", failed=failed, total=len(results))

        total_tokens = sum(r.total_tokens for r in successful)
        log.info(
            "Batch completion finished",
            total_prompts=len(prompts),
            successful=len(successful),
            total_tokens=total_tokens,
        )

        return results

    # ── Simple convenience wrappers ────────────────────────────────────────────

    async def complete(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """
        Convenience wrapper: build messages and call async_chat_completion.

        The most commonly used method in pipeline code.

        Args:
            user_prompt:   The task/question for the LLM.
            system_prompt: Optional system instruction.
            json_mode:     If True, set response_format={"type":"json_object"}.
                           The prompt must explicitly ask for JSON output.
            max_tokens:    Override max tokens.
            temperature:   Override temperature.

        Returns:
            CompletionResult.

        Example:
            result = await service.complete(
                user_prompt=extraction_prompt,
                system_prompt="You are a knowledge graph expert.",
                json_mode=False,
            )
            entities = parse_extraction_output(result.content)
        """
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
        """
        Sync convenience wrapper. Use only in tests or CLI scripts.
        """
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
        response_format: dict | None,
        logit_bias: dict[str, int] | None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for an OpenAI API call."""
        kwargs: dict[str, Any] = {
            "model":       model or self.model,
            "messages":    messages,
            "max_tokens":  max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if logit_bias is not None:
            kwargs["logit_bias"] = logit_bias
        return kwargs

    def _parse_response(
        self, response: ChatCompletion, latency_ms: float
    ) -> CompletionResult:
        """Extract content and usage from an OpenAI response object."""
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
        """Log a completion result at DEBUG level."""
        log.debug(
            "Completion finished",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.estimated_cost_usd,
            latency_ms=round(result.latency_ms, 1),
            finish_reason=result.finish_reason,
            sync=sync,
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIService(model={self.model!r}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature})"
        )


# ── Factory function ───────────────────────────────────────────────────────────

def get_openai_service(
    api_key: str | None = None,
    model: str | None = None,
) -> OpenAIService:
    """
    Build an OpenAIService from application settings.

    Args:
        api_key: Override API key (useful in tests).
        model:   Override model name.

    Returns:
        Configured OpenAIService instance.
    """
    from app.config import get_settings
    settings = get_settings()

    return OpenAIService(
        api_key=api_key or settings.openai_api_key,
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