"""
workers/extraction_worker.py — Parallel chunk extraction worker pool.

Purpose:
  The extraction stage (Stage 2) is the most expensive part of the indexing
  pipeline: it calls the LLM once per chunk (plus gleaning rounds), meaning a
  1,000-chunk corpus requires 2,000–4,000 OpenAI calls.  This module wraps the
  ExtractionPipeline with worker-pool semantics so the pipeline_runner can:

    1. Run up to `max_concurrency` LLM calls at once (OpenAI rate-limit aware).
    2. Save each result to disk immediately after extraction (crash recovery).
    3. Stream live progress back to the indexing worker via a callback.
    4. Support graceful cancellation: stop accepting new chunks while in-flight
       calls complete.

Design:
  - ExtractionWorkerPool is an async context manager.  Use `async with` to
    ensure clean shutdown even on exceptions.
  - Internally it uses asyncio.Semaphore (via gather_with_concurrency) to cap
    concurrency, matching OpenAI's TPM/RPM rate limits.
  - The on_chunk_complete callback is the same contract as ExtractionPipeline
    uses internally — the pipeline_runner passes the same callback through.
  - Results are yielded in completion order, not submission order, because LLM
    latency per chunk varies.  The pipeline_runner sorts by chunk_id if order
    matters downstream.

Usage:
    pool = ExtractionWorkerPool(
        openai_service=openai_svc,
        tokenizer=tokenizer,
        gleaning_rounds=2,
        max_concurrency=20,
    )
    async with pool:
        results = await pool.extract_batch(
            chunks=chunks,
            on_chunk_complete=save_to_disk,
        )

    # Or use the convenience function:
    results = await run_extraction_workers(
        chunks=chunks,
        openai_service=openai_svc,
        tokenizer=tokenizer,
        gleaning_rounds=2,
        max_concurrency=20,
        on_chunk_complete=save_to_disk,
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger

# Imported at module level so unittest.mock.patch() can intercept them.
# patch("app.workers.extraction_worker.ExtractionPipeline") works correctly.
from app.core.pipeline.extraction import ExtractionPipeline
from app.core.pipeline.gleaning import GleaningLoop
from app.models.graph_models import ChunkExtraction

log = get_logger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ExtractionBatchResult:
    """
    Summary of a completed extraction batch.

    Aggregates counts across all chunks so the caller (indexing_worker) can
    update the job store with a single dict write.
    """
    total_chunks:    int = 0
    successful:      int = 0
    failed:          int = 0
    total_entities:  int = 0
    total_relations: int = 0
    elapsed_seconds: float = 0.0
    chunks_per_second: float = 0.0

    # Individual extractions — same ordering as the input chunks list
    extractions: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of chunks that extracted successfully (0.0–1.0)."""
        if self.total_chunks == 0:
            return 0.0
        return self.successful / self.total_chunks

    def __repr__(self) -> str:
        return (
            f"ExtractionBatchResult("
            f"total={self.total_chunks}, "
            f"ok={self.successful}, "
            f"failed={self.failed}, "
            f"entities={self.total_entities}, "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )


# ── Worker pool ────────────────────────────────────────────────────────────────

class ExtractionWorkerPool:
    """
    Async worker pool for parallel chunk extraction.

    Manages a bounded pool of concurrent LLM calls, progress tracking, and
    incremental persistence.  Use as an async context manager.

    Args:
        openai_service:  OpenAIService instance (authenticated, rate-limited).
        tokenizer:       TokenizerService for prompt budgeting.
        gleaning_rounds: Number of self-reflection gleaning iterations (paper: 2).
        max_concurrency: Maximum simultaneous LLM calls (default: 20).
        skip_claims:     If True, skip claim extraction during entity extraction.
    """

    def __init__(
        self,
        openai_service: Any,
        tokenizer: Any,
        gleaning_rounds: int = 2,
        max_concurrency: int = 20,
        skip_claims: bool = False,
    ) -> None:
        self._openai_service  = openai_service
        self._tokenizer       = tokenizer
        self._gleaning_rounds = gleaning_rounds
        self._max_concurrency = max_concurrency
        self._skip_claims     = skip_claims

        self._pipeline        = None   # built lazily in __aenter__
        self._cancelled        = False
        self._active_tasks: list[asyncio.Task] = []

    # ── Context manager ────────────────────────────────────────────────────────

    async def __aenter__(self) -> "ExtractionWorkerPool":
        """Build the ExtractionPipeline and GleaningLoop on entry."""
        gleaning_loop = (
            GleaningLoop(self._openai_service)
            if self._gleaning_rounds > 0
            else None
        )
        self._pipeline = ExtractionPipeline(
            openai_service=self._openai_service,
            tokenizer=self._tokenizer,
            gleaning_loop=gleaning_loop,
            skip_claims=self._skip_claims,
        )
        log.debug(
            "ExtractionWorkerPool entered",
            max_concurrency=self._max_concurrency,
            gleaning_rounds=self._gleaning_rounds,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Cancel any remaining in-flight tasks on exit.

        Returns False so exceptions propagate normally.
        """
        if self._active_tasks:
            pending = [t for t in self._active_tasks if not t.done()]
            if pending:
                log.info(
                    "ExtractionWorkerPool: cancelling in-flight tasks",
                    pending=len(pending),
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        self._active_tasks.clear()
        return False

    # ── Core extraction method ─────────────────────────────────────────────────

    async def extract_batch(
        self,
        chunks: list,
        on_chunk_complete: Callable | None = None,
    ) -> ExtractionBatchResult:
        """
        Extract all chunks concurrently and return an ExtractionBatchResult.

        Args:
            chunks:            List of ChunkSchema objects to extract.
            on_chunk_complete: Optional async or sync callback called after each
                               chunk completes.  Signature: callback(extraction).
                               Called even for failed extractions.

        Returns:
            ExtractionBatchResult with aggregate counts and individual extractions.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "ExtractionWorkerPool must be used as an async context manager. "
                "Use `async with ExtractionWorkerPool(...) as pool:`"
            )

        if not chunks:
            log.info("ExtractionWorkerPool.extract_batch: no chunks to process")
            return ExtractionBatchResult()

        t0 = time.monotonic()
        completed_count = [0]
        extractions_out: list = []

        log.info(
            "Extraction batch starting",
            total_chunks=len(chunks),
            gleaning_rounds=self._gleaning_rounds,
            max_concurrency=self._max_concurrency,
        )

        async def _extract_one(chunk) -> Any:
            """
            Extract one chunk, invoke the progress callback, and return.

            Does not raise — failed extractions are returned as ChunkExtraction
            objects with extraction_completed=False so the caller can decide
            whether to retry or continue.
            """
            if self._cancelled:
                return None

            try:
                result = await self._pipeline.extract_chunk(
                    chunk,
                    gleaning_rounds=self._gleaning_rounds,
                )
            except Exception as exc:
                log.warning(
                    "Chunk extraction failed",
                    chunk_id=getattr(chunk, "chunk_id", "?"),
                    error=str(exc),
                )
                # Build a failed extraction so the callback still fires
                result = ChunkExtraction(
                    chunk_id=getattr(chunk, "chunk_id", "unknown"),
                    extraction_completed=False,
                    error_message=str(exc),
                )

            # Invoke the persistence / progress callback
            if on_chunk_complete:
                try:
                    if asyncio.iscoroutinefunction(on_chunk_complete):
                        await on_chunk_complete(result)
                    else:
                        on_chunk_complete(result)
                except Exception as cb_exc:
                    log.warning(
                        "on_chunk_complete callback raised",
                        error=str(cb_exc),
                    )

            completed_count[0] += 1
            return result

        # Run all coroutines through the semaphore-bounded gather
        raw_results = await gather_with_concurrency(
            coroutines=[_extract_one(chunk) for chunk in chunks],
            max_concurrency=self._max_concurrency,
            return_exceptions=False,
        )

        elapsed = time.monotonic() - t0

        # Filter out None (cancelled) results
        extractions = [r for r in raw_results if r is not None]

        successful = [e for e in extractions if getattr(e, "extraction_completed", False)]
        failed     = [e for e in extractions if not getattr(e, "extraction_completed", True)]

        total_entities  = sum(len(getattr(e, "entities", [])) for e in successful)
        total_relations = sum(len(getattr(e, "relationships", [])) for e in successful)

        result = ExtractionBatchResult(
            total_chunks=len(chunks),
            successful=len(successful),
            failed=len(failed),
            total_entities=total_entities,
            total_relations=total_relations,
            elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(len(chunks) / elapsed, 2) if elapsed > 0 else 0.0,
            extractions=extractions,
        )

        log.info(
            "Extraction batch complete",
            total=result.total_chunks,
            successful=result.successful,
            failed=result.failed,
            total_entities=result.total_entities,
            total_relationships=result.total_relations,
            elapsed_seconds=result.elapsed_seconds,
            chunks_per_second=result.chunks_per_second,
        )

        return result

    # ── Cancellation ───────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """
        Signal the pool to stop accepting new work.

        In-flight extractions will complete normally; no new _extract_one
        coroutines will start after this is called.  Used by the indexing
        worker when the job is cancelled externally.
        """
        self._cancelled = True
        log.info("ExtractionWorkerPool: cancellation requested")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_cancelled(self) -> bool:
        """True if cancel() has been called."""
        return self._cancelled

    @property
    def max_concurrency(self) -> int:
        """The concurrency cap this pool was initialised with."""
        return self._max_concurrency


# ── Convenience function ───────────────────────────────────────────────────────

async def run_extraction_workers(
    chunks: list,
    openai_service: Any,
    tokenizer: Any,
    gleaning_rounds: int = 2,
    max_concurrency: int = 20,
    skip_claims: bool = False,
    on_chunk_complete: Callable | None = None,
) -> ExtractionBatchResult:
    """
    Run parallel chunk extraction without manually managing the pool lifecycle.

    This is the simplest way to call the extraction worker pool from the
    pipeline runner or from tests.

    Args:
        chunks:            ChunkSchema list to extract.
        openai_service:    Authenticated OpenAIService.
        tokenizer:         TokenizerService.
        gleaning_rounds:   LLM gleaning iterations (paper: 2).
        max_concurrency:   Max simultaneous LLM calls (default: 20).
        skip_claims:       If True, skip claim extraction.
        on_chunk_complete: Callback(extraction) invoked after each chunk.

    Returns:
        ExtractionBatchResult with aggregate stats and all extractions.

    Example:
        result = await run_extraction_workers(
            chunks=pending_chunks,
            openai_service=openai_svc,
            tokenizer=tokenizer_svc,
            gleaning_rounds=2,
            on_chunk_complete=lambda e: artifact_store.append_extraction(e),
        )
        print(f"Extracted {result.successful}/{result.total_chunks} chunks")
    """
    async with ExtractionWorkerPool(
        openai_service=openai_service,
        tokenizer=tokenizer,
        gleaning_rounds=gleaning_rounds,
        max_concurrency=max_concurrency,
        skip_claims=skip_claims,
    ) as pool:
        return await pool.extract_batch(
            chunks=chunks,
            on_chunk_complete=on_chunk_complete,
        )


# ── Batch splitter helper ──────────────────────────────────────────────────────

def split_into_batches(items: list, batch_size: int) -> list[list]:
    """
    Split a flat list into fixed-size sub-lists.

    Used when you want to process chunks in discrete batches rather than
    one continuous stream (e.g. for checkpoint saving between batches).

    Args:
        items:      List to split.
        batch_size: Target size of each sub-list.  The last batch may be smaller.

    Returns:
        List of sub-lists.

    Example:
        batches = split_into_batches(chunks, batch_size=50)
        for batch in batches:
            await pool.extract_batch(batch)
            artifact_store.checkpoint()
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]