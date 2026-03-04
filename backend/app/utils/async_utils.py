"""
utils/async_utils.py — Async concurrency helpers for pipeline and query stages.

The GraphRAG map stage and the extraction pipeline both need to fire many
LLM calls in parallel while staying within OpenAI rate limits.  These
utilities provide:

- Rate-limited batch execution with a semaphore
- Progress-tracked gather for long-running operations
- Chunked async iteration to avoid overwhelming the event loop
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Coroutine, Iterable, TypeVar

from app.utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")


async def run_with_semaphore(
    coro: Coroutine[Any, Any, T],
    semaphore: asyncio.Semaphore,
) -> T:
    """
    Run a coroutine while holding a semaphore slot.

    Used to cap concurrent OpenAI calls across the entire pipeline.
    """
    async with semaphore:
        return await coro


async def gather_with_concurrency(
    coroutines: Iterable[Coroutine[Any, Any, T]],
    max_concurrency: int = 20,
    return_exceptions: bool = False,
) -> list[T]:
    """
    Gather coroutines with a hard concurrency cap.

    Args:
        coroutines:       Iterable of coroutines to execute.
        max_concurrency:  Maximum number of coroutines running at once.
                          20 is a safe default for GPT-4o at standard rate limits.
        return_exceptions: If True, exceptions are returned as results instead
                           of being raised. Use for fault-tolerant bulk ops.

    Returns:
        List of results in the same order as the input coroutines.

    Example:
        results = await gather_with_concurrency(
            [call_openai(chunk) for chunk in chunks],
            max_concurrency=20,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(run_with_semaphore(coro, semaphore))
        for coro in coroutines
    ]

    log.debug(
        "Starting concurrent batch",
        total_tasks=len(tasks),
        max_concurrency=max_concurrency,
    )

    results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    # Log any exceptions if return_exceptions=True
    if return_exceptions:
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            log.warning(
                "Batch completed with errors",
                total=len(results),
                errors=len(errors),
                first_error=str(errors[0]),
            )

    return list(results)


async def batch_process(
    items: list[Any],
    processor: Callable[[Any], Coroutine[Any, Any, T]],
    batch_size: int = 50,
    max_concurrency: int = 20,
    log_progress: bool = True,
) -> list[T]:
    """
    Process a large list of items in batches, with concurrency control.

    Splits items into batches to avoid creating thousands of tasks at once,
    which would consume excessive memory.

    Args:
        items:            List of items to process.
        processor:        Async function to apply to each item.
        batch_size:       Number of items per batch (limits task queue size).
        max_concurrency:  Max concurrent LLM calls within each batch.
        log_progress:     Whether to log batch completion progress.

    Returns:
        Flat list of results in input order.

    Example:
        answers = await batch_process(
            chunks,
            processor=lambda c: extract_entities(c),
            batch_size=50,
            max_concurrency=20,
        )
    """
    all_results: list[T] = []
    total = len(items)
    batches = [items[i:i + batch_size] for i in range(0, total, batch_size)]

    log.info(
        "Starting batch processing",
        total_items=total,
        batch_size=batch_size,
        total_batches=len(batches),
        max_concurrency=max_concurrency,
    )

    for batch_idx, batch in enumerate(batches):
        batch_results = await gather_with_concurrency(
            [processor(item) for item in batch],
            max_concurrency=max_concurrency,
            return_exceptions=False,
        )
        all_results.extend(batch_results)

        if log_progress:
            processed = min((batch_idx + 1) * batch_size, total)
            log.info(
                "Batch progress",
                processed=processed,
                total=total,
                percent=round(processed / total * 100, 1),
            )

    log.info("Batch processing complete", total_items=total)
    return all_results


async def run_in_executor(func: Callable[..., T], *args: Any) -> T:
    """
    Run a blocking (synchronous) function in a thread pool executor
    without blocking the event loop.

    Used for: pickle serialisation, FAISS operations, file I/O.

    Example:
        graph = await run_in_executor(pickle.load, file_handle)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


async def async_chunks(
    items: list[Any],
    chunk_size: int,
) -> AsyncIterator[list[Any]]:
    """
    Async generator that yields chunks of a list, yielding control to the
    event loop between chunks so other tasks can run.

    Used in the map stage to avoid blocking when iterating large summary lists.
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]
        await asyncio.sleep(0)          # yield to event loop


__all__ = [
    "gather_with_concurrency",
    "batch_process",
    "run_in_executor",
    "async_chunks",
    "run_with_semaphore",
]