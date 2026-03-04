"""
services/embedding_service.py — OpenAI text embedding service.

Wraps the OpenAI embeddings API for all vector operations in the pipeline.
Used for:
  - Embedding text chunks for the VectorRAG FAISS index (Stage 5 indexing)
  - Embedding queries at query time for FAISS similarity search (Stage 6)

The paper's baseline system ("SS" — semantic search) uses embeddings to
fill the context window: embed query → top-k similar chunks → fill 8k window.

Model: text-embedding-3-small (1536 dimensions)
  - Same model as the paper's SS baseline
  - OpenAI recommends normalizing embeddings before cosine similarity search
  - FAISS IndexFlatIP (inner product) on L2-normalized vectors = cosine similarity

All calls go through embedding_retry from utils/retry.py.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse

from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.utils.retry import embedding_retry

log = get_logger(__name__)

# ── Embedding dimension constants ──────────────────────────────────────────────
# text-embedding-3-small: 1536 dimensions (default)
# text-embedding-3-large: 3072 dimensions
# text-embedding-ada-002: 1536 dimensions (legacy)
EMBEDDING_DIM_SMALL = 1536
EMBEDDING_DIM_LARGE = 3072

# OpenAI's max batch size for embeddings API
_OPENAI_EMBED_BATCH_SIZE = 2048


class EmbeddingService:
    """
    OpenAI text embedding service.

    Provides both sync and async embedding for single texts and batches.
    All vectors are L2-normalized before returning so they are ready
    for cosine similarity via FAISS IndexFlatIP.

    Usage:
        service = EmbeddingService(api_key="sk-...", model="text-embedding-3-small")

        # Single embedding
        vector = await service.embed_text("OpenAI was founded in 2015")

        # Batch — for indexing all chunks
        vectors = await service.embed_batch(chunk_texts)
        # → numpy array shape (N, 1536), dtype float32, L2-normalized
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout: int = 60,
        dimensions: int | None = None,
    ) -> None:
        """
        Args:
            api_key:    OpenAI API key.
            model:      Embedding model. Default: text-embedding-3-small.
            timeout:    Request timeout in seconds.
            dimensions: Optional output dimension reduction (text-embedding-3 only).
                        None uses the model's default (1536 for small).
        """
        self.model      = model
        self.timeout    = timeout
        self.dimensions = dimensions

        self._sync_client  = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self._async_client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

        log.info(
            "EmbeddingService initialized",
            model=model,
            dimensions=dimensions or "default",
        )

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        if self.dimensions:
            return self.dimensions
        if "large" in self.model:
            return EMBEDDING_DIM_LARGE
        return EMBEDDING_DIM_SMALL

    # ── Single text embedding ──────────────────────────────────────────────────

    @embedding_retry
    def embed_text_sync(self, text: str) -> np.ndarray:
        """
        Synchronous single-text embedding.

        Returns L2-normalized float32 vector of shape (embedding_dim,).
        Use in tests and CLI. For pipeline code use embed_text().

        Args:
            text: Text to embed. Will be cleaned before embedding.

        Returns:
            numpy ndarray of shape (embedding_dim,), dtype float32, L2-normalized.
        """
        text = _clean_text(text)
        t0 = time.monotonic()

        kwargs = self._build_kwargs([text])
        response: CreateEmbeddingResponse = self._sync_client.embeddings.create(**kwargs)

        latency_ms = (time.monotonic() - t0) * 1000
        vector = _extract_vector(response, 0)

        log.debug(
            "Sync embedding complete",
            model=self.model,
            tokens=response.usage.total_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return vector

    @embedding_retry
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Async single-text embedding.

        Returns L2-normalized float32 vector of shape (embedding_dim,).
        Used at query time to embed the user's question before FAISS search.

        Args:
            text: Text to embed.

        Returns:
            numpy ndarray of shape (embedding_dim,), dtype float32, L2-normalized.

        Example:
            query_vector = await service.embed_text("What are the main themes?")
            # → array of shape (1536,), ready for faiss.search()
        """
        text = _clean_text(text)
        t0 = time.monotonic()

        kwargs = self._build_kwargs([text])
        response: CreateEmbeddingResponse = await self._async_client.embeddings.create(**kwargs)

        latency_ms = (time.monotonic() - t0) * 1000
        vector = _extract_vector(response, 0)

        log.debug(
            "Async embedding complete",
            model=self.model,
            tokens=response.usage.total_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return vector

    # ── Batch embedding ────────────────────────────────────────────────────────

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 512,
        max_concurrency: int = 5,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts in batches with concurrency control.

        Used during indexing to embed all document chunks. OpenAI's embedding
        API accepts up to 2048 texts per request, but we use smaller batches
        for reliability and rate limit safety.

        Args:
            texts:           List of text strings to embed.
            batch_size:      Texts per API call. Default: 512.
                             Larger batches are more efficient but may hit
                             token limits for long texts.
            max_concurrency: Concurrent API calls. Default: 5.
                             Embedding calls are faster than chat completions —
                             5 concurrent calls is safe at standard rate limits.
            show_progress:   Log progress every batch.

        Returns:
            numpy ndarray of shape (len(texts), embedding_dim),
            dtype float32, every row is L2-normalized.

        Example:
            chunk_texts = [c["text"] for c in chunks]
            embeddings = await service.embed_batch(chunk_texts)
            # → shape (N, 1536), ready to pass to faiss_service.build_index()
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Clean all texts first
        cleaned = [_clean_text(t) for t in texts]
        total = len(cleaned)

        # Split into batches
        batches = [
            cleaned[i:i + batch_size]
            for i in range(0, total, batch_size)
        ]

        log.info(
            "Starting batch embedding",
            total_texts=total,
            batch_size=batch_size,
            total_batches=len(batches),
            max_concurrency=max_concurrency,
        )

        t0 = time.monotonic()

        async def _embed_one_batch(batch: list[str]) -> np.ndarray:
            return await self._embed_batch_call(batch)

        # Run batches concurrently
        batch_results = await gather_with_concurrency(
            coroutines=[_embed_one_batch(b) for b in batches],
            max_concurrency=max_concurrency,
            return_exceptions=False,
        )

        # Stack all batch results into one array
        all_vectors = np.vstack(batch_results).astype(np.float32)

        latency_ms = (time.monotonic() - t0) * 1000
        log.info(
            "Batch embedding complete",
            total_texts=total,
            shape=str(all_vectors.shape),
            latency_ms=round(latency_ms, 1),
        )

        return all_vectors

    @embedding_retry
    async def _embed_batch_call(self, texts: list[str]) -> np.ndarray:
        """
        Single API call to embed a batch of texts.

        Returns ndarray of shape (len(texts), embedding_dim).
        All vectors are L2-normalized.
        """
        t0 = time.monotonic()
        kwargs = self._build_kwargs(texts)
        response: CreateEmbeddingResponse = await self._async_client.embeddings.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        # Extract all vectors in sorted order (API guarantees order but we verify)
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for item in response.data:
            vectors[item.index] = np.array(item.embedding, dtype=np.float32)

        # L2-normalize each vector for cosine similarity via FAISS inner product
        vectors = _l2_normalize(vectors)

        log.debug(
            "Batch embedding call complete",
            batch_size=len(texts),
            tokens=response.usage.total_tokens,
            latency_ms=round(latency_ms, 1),
        )
        return vectors

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_kwargs(self, texts: list[str]) -> dict[str, Any]:
        """Build kwargs for an embeddings API call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    def __repr__(self) -> str:
        return (
            f"EmbeddingService(model={self.model!r}, "
            f"embedding_dim={self.embedding_dim})"
        )


# ── Module-level helpers ───────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Clean text before embedding.

    OpenAI recommends replacing newlines with spaces for embedding quality.
    Empty strings cause API errors — replace with a single space.
    """
    if not text or not text.strip():
        return " "
    return text.replace("\n", " ").strip()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2D array of embedding vectors in-place.

    After normalization, inner product = cosine similarity.
    FAISS IndexFlatIP + L2-normalized vectors = cosine similarity search.

    Zero vectors (from empty texts) remain zero after normalization.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


def _extract_vector(response: CreateEmbeddingResponse, index: int) -> np.ndarray:
    """Extract and normalize a single embedding vector from an API response."""
    vector = np.array(response.data[index].embedding, dtype=np.float32)
    # Normalize single vector: reshape to (1, D), normalize, reshape back
    normalized = _l2_normalize(vector.reshape(1, -1)).reshape(-1)
    return normalized


# ── Factory function ───────────────────────────────────────────────────────────

def get_embedding_service(api_key: str | None = None) -> EmbeddingService:
    """
    Build an EmbeddingService from application settings.

    Args:
        api_key: Override API key (useful in tests).

    Returns:
        Configured EmbeddingService instance.
    """
    from app.config import get_settings
    settings = get_settings()

    return EmbeddingService(
        api_key=api_key or settings.openai_api_key,
        model=settings.openai_embedding_model,
        timeout=settings.openai_timeout,
    )


__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "EMBEDDING_DIM_SMALL",
    "EMBEDDING_DIM_LARGE",
]