"""
services/embedding_service.py — Text embedding service using Google Gemini.

Uses the official google-generativeai SDK.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import google.generativeai as genai
import numpy as np

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.utils.retry import embedding_retry

log = get_logger(__name__)

# Gemini embedding dimensions
EMBEDDING_DIM_DEFAULT = 768
EMBEDDING_DIM_FULL    = 3072


class EmbeddingService:
    """
    Text embedding service using Google Gemini embeddings.

    Identical interface to the original OpenAI-backed version.
    All vectors are L2-normalized, ready for FAISS cosine similarity.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        timeout: int = 60,
        dimensions: int | None = 768,
    ) -> None:
        self.model      = model
        self.timeout    = timeout
        self.dimensions = dimensions or EMBEDDING_DIM_DEFAULT

        # Configure the official Gemini SDK
        genai.configure(api_key=api_key)
        self.client = genai

        log.info(
            "EmbeddingService initialized (Gemini)",
            model=model,
            dimensions=self.dimensions,
        )

    @property
    def embedding_dim(self) -> int:
        return self.dimensions

    # ── Single text embedding ──────────────────────────────────────────────────

    @embedding_retry
    def embed_text_sync(self, text: str) -> np.ndarray:
        """Synchronous single-text embedding."""
        text = _clean_text(text)
        t0 = time.monotonic()
        result = self.client.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        latency_ms = (time.monotonic() - t0) * 1000
        vector = np.array(result['embedding'], dtype=np.float32)
        # Truncate to target dimensions if needed
        if len(vector) > self.dimensions:
            vector = vector[:self.dimensions]
        log.debug("Sync embedding complete", model=self.model, latency_ms=round(latency_ms, 1))
        return vector

    @embedding_retry
    async def embed_text(self, text: str) -> np.ndarray:
        """Async single-text embedding. Used at query time."""
        text = _clean_text(text)
        t0 = time.monotonic()
        # Gemini SDK doesn't have async embed_content, so run in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.client.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
            )
        latency_ms = (time.monotonic() - t0) * 1000
        vector = np.array(result['embedding'], dtype=np.float32)
        # Truncate to target dimensions if needed
        if len(vector) > self.dimensions:
            vector = vector[:self.dimensions]
        log.debug("Async embedding complete", model=self.model, latency_ms=round(latency_ms, 1))
        return vector

    # ── Batch embedding ────────────────────────────────────────────────────────

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 50,        # Gemini max is 100; use 50 for safety
        max_concurrency: int = 3,    # Conservative for free tier (1000 RPD)
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts in batches. Returns shape (N, embedding_dim)."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cleaned = [_clean_text(t) for t in texts]
        batches = [
            cleaned[i:i + batch_size]
            for i in range(0, len(cleaned), batch_size)
        ]

        log.info(
            "Starting batch embedding (Gemini)",
            total_texts=len(texts),
            batch_size=batch_size,
            total_batches=len(batches),
        )

        t0 = time.monotonic()

        batch_results = await gather_with_concurrency(
            coroutines=[self._embed_batch_call(b) for b in batches],
            max_concurrency=max_concurrency,
            return_exceptions=False,
        )

        all_vectors = np.vstack(batch_results).astype(np.float32)
        log.info(
            "Batch embedding complete",
            total_texts=len(texts),
            shape=str(all_vectors.shape),
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return all_vectors

    @embedding_retry
    async def _embed_batch_call(self, texts: list[str]) -> np.ndarray:
        """Single API call to embed a batch."""
        t0 = time.monotonic()
        # Gemini SDK doesn't have async embed_content, so run in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.client.embed_content(
                    model=self.model,
                    content=texts,
                    task_type="retrieval_document"
                )
            )
        latency_ms = (time.monotonic() - t0) * 1000

        vectors = np.array(result['embedding'], dtype=np.float32)
        # If single text, reshape to (1, dim)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # Truncate to target dimensions if needed
        if vectors.shape[1] > self.dimensions:
            vectors = vectors[:, :self.dimensions]

        vectors = _l2_normalize(vectors)
        log.debug("Batch embedding call done", batch_size=len(texts), latency_ms=round(latency_ms, 1))
        return vectors

    def __repr__(self) -> str:
        return f"EmbeddingService(model={self.model!r}, embedding_dim={self.embedding_dim})"


# ── Module-level helpers ───────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    if not text or not text.strip():
        return " "
    return text.replace("\n", " ").strip()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


# ── Factory function ───────────────────────────────────────────────────────────

def get_embedding_service(api_key: str | None = None) -> EmbeddingService:
    from app.config import get_settings
    settings = get_settings()
    return EmbeddingService(
        api_key=api_key or settings.gemini_api_key,
        model=settings.embedding_model,
        timeout=settings.openai_timeout,
        dimensions=settings.embedding_dimension,
    )


# ── Backwards-compat aliases (old names used in services/__init__.py) ─────────
EMBEDDING_DIM_SMALL = EMBEDDING_DIM_DEFAULT   # 768
EMBEDDING_DIM_LARGE = EMBEDDING_DIM_FULL      # 3072

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "EMBEDDING_DIM_DEFAULT",
    "EMBEDDING_DIM_FULL",
    "EMBEDDING_DIM_SMALL",
    "EMBEDDING_DIM_LARGE",
]