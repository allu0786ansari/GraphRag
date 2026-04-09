"""
services/embedding_service.py — Text embedding service using Google Gemini.

Migration: switched from OpenAI text-embedding-3-small to Gemini embeddings.
- gemini-embedding-exp-03-07: free, 1000 RPD, outputs up to 3072 dims
- We use 768 dimensions (configurable) to keep FAISS index small and fast
- Still uses the OpenAI-compatible endpoint so no SDK change needed

Gemini embedding API notes:
  - Max 100 texts per batch request
  - 1000 requests/day free (no credit card)
  - embedding_dimension config controls output size (768 default)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.utils.retry import embedding_retry

log = get_logger(__name__)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini embedding-exp-03-07 supports up to 3072 dims, we use 768 by default
EMBEDDING_DIM_DEFAULT = 768
EMBEDDING_DIM_FULL    = 3072

# Gemini max texts per embedding request
_GEMINI_EMBED_BATCH_SIZE = 100


class EmbeddingService:
    """
    Text embedding service using Google Gemini embeddings.

    Identical interface to the original OpenAI-backed version.
    All vectors are L2-normalized, ready for FAISS cosine similarity.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-exp-03-07",
        timeout: int = 60,
        dimensions: int | None = 768,
    ) -> None:
        self.model      = model
        self.timeout    = timeout
        self.dimensions = dimensions or EMBEDDING_DIM_DEFAULT

        # Use openai SDK with Gemini base_url — OpenAI-compatible endpoint
        from openai import AsyncOpenAI, OpenAI
        self._sync_client  = OpenAI(
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
        kwargs = self._build_kwargs([text])
        response = self._sync_client.embeddings.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        vector = _extract_vector(response, 0, self.dimensions)
        log.debug("Sync embedding complete", model=self.model, latency_ms=round(latency_ms, 1))
        return vector

    @embedding_retry
    async def embed_text(self, text: str) -> np.ndarray:
        """Async single-text embedding. Used at query time."""
        text = _clean_text(text)
        t0 = time.monotonic()
        kwargs = self._build_kwargs([text])
        response = await self._async_client.embeddings.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        vector = _extract_vector(response, 0, self.dimensions)
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
        kwargs = self._build_kwargs(texts)
        response = await self._async_client.embeddings.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for item in response.data:
            raw = np.array(item.embedding, dtype=np.float32)
            # Truncate or pad to target dimensions
            if len(raw) >= self.dimensions:
                vectors[item.index] = raw[:self.dimensions]
            else:
                vectors[item.index, :len(raw)] = raw

        vectors = _l2_normalize(vectors)
        log.debug("Batch embedding call done", batch_size=len(texts), latency_ms=round(latency_ms, 1))
        return vectors

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_kwargs(self, texts: list[str]) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

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


def _extract_vector(response: Any, index: int, dimensions: int) -> np.ndarray:
    raw = np.array(response.data[index].embedding, dtype=np.float32)
    if len(raw) >= dimensions:
        vector = raw[:dimensions].reshape(1, -1)
    else:
        vector = np.zeros((1, dimensions), dtype=np.float32)
        vector[0, :len(raw)] = raw
    return _l2_normalize(vector).reshape(-1)


# ── Factory function ───────────────────────────────────────────────────────────

def get_embedding_service(api_key: str | None = None) -> EmbeddingService:
    from app.config import get_settings
    settings = get_settings()
    return EmbeddingService(
        api_key=api_key or settings.gemini_api_key,
        model=settings.openai_embedding_model,
        timeout=settings.openai_timeout,
        dimensions=settings.embedding_dimension,
    )


__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "EMBEDDING_DIM_DEFAULT",
    "EMBEDDING_DIM_FULL",
]