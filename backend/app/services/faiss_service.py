"""
services/faiss_service.py — FAISS vector index for VectorRAG semantic search.

This service manages the FAISS index that powers the VectorRAG baseline system
(called "SS" — semantic search — in the paper, Section 3.2).

The VectorRAG pipeline:
  Indexing time:  embed all 600-token chunks → build FAISS index → save to disk
  Query time:     embed query → search FAISS for top-k → fill 8k context → answer

Index type: IndexFlatIP (Inner Product)
  - Combined with L2-normalized vectors from EmbeddingService
  - Inner product of L2-normalized vectors = cosine similarity
  - IndexFlatIP is exact (no approximation) — correct for our corpus sizes
  - For 1M+ token corpora chunked at 600 tokens: ~1700 chunks
    → exact search over 1700 vectors is fast (<5ms) and needs no ANN approximation

Metadata storage:
  - FAISS only stores float vectors — it has no concept of text or metadata
  - We store chunk metadata (text, source, chunk_id) in parallel numpy/JSON files
  - Index ID i corresponds to metadata[i]
  - This parallel structure is kept in sync by build_index() and save()/load()

Thread safety:
  - FAISS IndexFlatIP is thread-safe for concurrent reads (search)
  - Writes (build, add) must be done before serving queries
  - This service uses a threading.RLock for defensive safety
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    A single result from a FAISS similarity search.

    FAISS returns (distances, indices) — we enrich this with the
    stored chunk metadata so callers get everything in one object.
    """
    rank: int                          # 1-based rank (1 = most similar)
    index_id: int                      # FAISS internal index ID
    score: float                       # Cosine similarity score (0.0–1.0)
    chunk_id: str                      # e.g. "news_article_001_0000"
    text: str                          # The chunk text
    source_document: str               # Source filename
    token_count: int                   # Tokens in this chunk
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"chunk_id={self.chunk_id!r}, text={preview!r}...)"
        )


# ── Service class ──────────────────────────────────────────────────────────────

class FAISSService:
    """
    FAISS vector index service for VectorRAG semantic search.

    Manages the full lifecycle of the FAISS index:
      build_index()  → construct from embeddings + metadata
      search()       → query top-k similar chunks
      save()         → persist to disk (index.bin + metadata.json)
      load()         → restore from disk
      add_vectors()  → incremental additions (post-indexing)

    Usage:
        service = FAISSService(embedding_dim=1536)

        # Build from scratch during indexing pipeline
        service.build_index(embeddings, chunk_metadata)
        service.save(index_path, metadata_path)

        # At query time
        service.load(index_path, metadata_path)
        results = service.search(query_vector, top_k=10)
    """

    def __init__(self, embedding_dim: int = 1536) -> None:
        """
        Args:
            embedding_dim: Dimension of embedding vectors.
                           1536 for text-embedding-3-small (default).
                           3072 for text-embedding-3-large.
        """
        self.embedding_dim = embedding_dim
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []   # parallel to FAISS index IDs
        self._lock = threading.RLock()
        self._is_built = False

        log.debug("FAISSService initialized", embedding_dim=embedding_dim)

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        chunk_metadata: list[dict[str, Any]],
    ) -> None:
        """
        Build a fresh FAISS IndexFlatIP from embeddings and metadata.

        This is called once during the indexing pipeline after all chunks
        have been embedded. Replaces any existing index.

        Args:
            embeddings:     Float32 ndarray of shape (N, embedding_dim).
                            Must be L2-normalized (EmbeddingService does this).
            chunk_metadata: List of N metadata dicts, one per embedding.
                            Each must contain at minimum:
                              - chunk_id:        str
                              - text:            str
                              - source_document: str
                              - token_count:     int
                            Additional keys are stored and returned in SearchResult.

        Raises:
            ValueError: If embeddings shape doesn't match embedding_dim.
            ValueError: If len(embeddings) != len(chunk_metadata).

        Example:
            embeddings = await embedding_service.embed_batch(chunk_texts)
            service.build_index(embeddings, chunk_metadata_list)
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D array, got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        if len(embeddings) != len(chunk_metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and "
                f"chunk_metadata ({len(chunk_metadata)}) must have same length"
            )

        t0 = time.monotonic()

        with self._lock:
            # Build fresh IndexFlatIP
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._index.add(embeddings)
            self._metadata = list(chunk_metadata)
            self._is_built = True

        latency_ms = (time.monotonic() - t0) * 1000

        log.info(
            "FAISS index built",
            vectors=len(embeddings),
            embedding_dim=self.embedding_dim,
            index_size=self._index.ntotal,
            latency_ms=round(latency_ms, 1),
        )

    def add_vectors(
        self,
        embeddings: np.ndarray,
        chunk_metadata: list[dict[str, Any]],
    ) -> None:
        """
        Add new vectors to an existing index (incremental update).

        Useful for adding new documents to an existing index without
        rebuilding from scratch.

        Args:
            embeddings:     Float32 ndarray of shape (M, embedding_dim).
            chunk_metadata: List of M metadata dicts.
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        if not self._is_built:
            # If no index exists, just build one
            self.build_index(embeddings, chunk_metadata)
            return

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        with self._lock:
            self._index.add(embeddings)
            self._metadata.extend(chunk_metadata)

        log.info(
            "Vectors added to FAISS index",
            added=len(embeddings),
            total=self._index.ntotal,
        )

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search the FAISS index for the top-k most similar chunks.

        This is the core of the VectorRAG query pipeline:
          embed(query) → search(query_vector, top_k=10) → fill 8k context

        Args:
            query_vector:    Float32 ndarray of shape (embedding_dim,) or (1, embedding_dim).
                             Must be L2-normalized (EmbeddingService does this).
            top_k:           Number of results to return. Default: 10.
                             The top-k chunks are used to fill the 8k context window.
            score_threshold: Minimum cosine similarity score (0.0–1.0).
                             Results below this threshold are filtered out.
                             Default: 0.0 (return all top-k results).

        Returns:
            List of SearchResult ordered by score descending (most similar first).
            May be shorter than top_k if fewer results exist or score_threshold filters some.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.

        Example:
            query_vec = await embedding_service.embed_text("What are the main themes?")
            results = service.search(query_vec, top_k=10)
            context = "\n\n".join(r.text for r in results)
        """
        if not self._is_built or self._index is None:
            raise RuntimeError(
                "FAISS index is not built. Call build_index() or load() first."
            )

        # Ensure correct shape: (1, embedding_dim) for FAISS
        query = np.array(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query vector dimension {query.shape[1]} != "
                f"index dimension {self.embedding_dim}"
            )

        actual_k = min(top_k, self._index.ntotal)
        if actual_k == 0:
            return []

        t0 = time.monotonic()

        with self._lock:
            scores, indices = self._index.search(query, actual_k)

        latency_ms = (time.monotonic() - t0) * 1000

        # scores shape: (1, k), indices shape: (1, k)
        scores_flat  = scores[0].tolist()
        indices_flat = indices[0].tolist()

        results = []
        for rank_0, (idx, score) in enumerate(zip(indices_flat, scores_flat)):
            if idx < 0:
                # FAISS returns -1 for padding when fewer results than top_k
                continue
            if score < score_threshold:
                continue
            if idx >= len(self._metadata):
                log.warning("FAISS index ID out of range", index_id=idx)
                continue

            meta = self._metadata[idx]
            results.append(SearchResult(
                rank=rank_0 + 1,
                index_id=idx,
                score=float(score),
                chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                text=meta.get("text", ""),
                source_document=meta.get("source_document", ""),
                token_count=meta.get("token_count", 0),
                metadata={k: v for k, v in meta.items()
                          if k not in ("chunk_id", "text", "source_document", "token_count")},
            ))

        log.debug(
            "FAISS search complete",
            top_k=top_k,
            returned=len(results),
            latency_ms=round(latency_ms, 2),
            top_score=round(results[0].score, 4) if results else 0.0,
        )

        return results

    def search_and_fill_context(
        self,
        query_vector: np.ndarray,
        max_tokens: int = 8000,
        top_k: int = 50,
        token_counter=None,
    ) -> tuple[list[SearchResult], int]:
        """
        Search and greedily fill a context window up to max_tokens.

        This is the complete VectorRAG context-filling logic matching the
        paper's baseline "SS" system (Section 3.2):
          1. Retrieve top_k candidates
          2. Add chunks greedily until the context window is full

        Args:
            query_vector:  L2-normalized query embedding.
            max_tokens:    Context window token budget. Default: 8000 (paper).
            top_k:         Initial candidate retrieval count.
                           Retrieve more candidates than needed so we can
                           fill the window precisely.
            token_counter: Optional callable(text) → int for token counting.
                           If None, estimates by character count / 4.

        Returns:
            Tuple of:
              - results:       SearchResult list that fills the context window.
              - total_tokens:  Actual tokens used.
        """
        candidates = self.search(query_vector, top_k=top_k)
        if not candidates:
            return [], 0

        if token_counter is None:
            # Rough estimate: 1 token ≈ 4 characters
            token_counter = lambda t: len(t) // 4

        selected = []
        total_tokens = 0

        for result in candidates:
            chunk_tokens = result.token_count or token_counter(result.text)
            if total_tokens + chunk_tokens > max_tokens:
                break
            selected.append(result)
            total_tokens += chunk_tokens

        log.debug(
            "Context window filled",
            candidates=len(candidates),
            selected=len(selected),
            total_tokens=total_tokens,
            max_tokens=max_tokens,
        )

        return selected, total_tokens

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """
        Save the FAISS index and metadata to disk.

        Two files are written:
          {index_path}    → binary FAISS index (faiss.write_index)
          {metadata_path} → JSON array of chunk metadata dicts

        Args:
            index_path:    Path for the FAISS binary file (.bin).
            metadata_path: Path for the metadata JSON file (.json).

        Raises:
            RuntimeError: If index has not been built.
        """
        if not self._is_built or self._index is None:
            raise RuntimeError("Cannot save: index has not been built.")

        index_path    = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()

        with self._lock:
            faiss.write_index(self._index, str(index_path))
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False)

        latency_ms = (time.monotonic() - t0) * 1000

        log.info(
            "FAISS index saved",
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            vectors=self._index.ntotal,
            latency_ms=round(latency_ms, 1),
        )

    def load(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """
        Load a FAISS index and metadata from disk.

        Args:
            index_path:    Path to the FAISS binary file (.bin).
            metadata_path: Path to the metadata JSON file (.json).

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If loaded index dimension doesn't match embedding_dim.
        """
        index_path    = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        t0 = time.monotonic()

        with self._lock:
            loaded_index = faiss.read_index(str(index_path))

            if loaded_index.d != self.embedding_dim:
                raise ValueError(
                    f"Loaded index dimension {loaded_index.d} != "
                    f"expected {self.embedding_dim}"
                )

            self._index = loaded_index

            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self._is_built = True

        latency_ms = (time.monotonic() - t0) * 1000

        log.info(
            "FAISS index loaded",
            index_path=str(index_path),
            vectors=self._index.ntotal,
            metadata_count=len(self._metadata),
            latency_ms=round(latency_ms, 1),
        )

    # ── Properties and diagnostics ─────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """True if the index has been built or loaded."""
        return self._is_built

    @property
    def total_vectors(self) -> int:
        """Number of vectors in the index. 0 if not built."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def metadata_count(self) -> int:
        """Number of metadata records. Should equal total_vectors."""
        return len(self._metadata)

    def get_metadata(self, index_id: int) -> dict[str, Any] | None:
        """Retrieve metadata for a specific FAISS index ID."""
        if index_id < 0 or index_id >= len(self._metadata):
            return None
        return self._metadata[index_id]

    def reset(self) -> None:
        """Clear the index and metadata. Useful in tests."""
        with self._lock:
            self._index = None
            self._metadata = []
            self._is_built = False
        log.debug("FAISSService reset")

    def __repr__(self) -> str:
        return (
            f"FAISSService("
            f"embedding_dim={self.embedding_dim}, "
            f"vectors={self.total_vectors}, "
            f"is_built={self._is_built})"
        )


# ── Factory function ───────────────────────────────────────────────────────────

def get_faiss_service() -> FAISSService:
    """
    Build a FAISSService from application settings.

    Returns:
        Configured FAISSService instance (not yet loaded — call load() separately).
    """
    from app.config import get_settings
    settings = get_settings()

    return FAISSService(embedding_dim=settings.embedding_dimension)


__all__ = [
    "FAISSService",
    "SearchResult",
    "get_faiss_service",
]