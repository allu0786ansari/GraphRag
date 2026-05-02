"""
core/query/vectorrag_engine.py — VectorRAG (Semantic Search) query engine.

Implements the 'SS' (semantic search) baseline from the paper (Section 3.2).

Pipeline at query time:
  1. Embed the query using the same embedding model used during indexing
     (text-embedding-3-small, L2-normalized, 1536 dimensions).
  2. Search the FAISS index for the top-k most similar document chunks.
  3. Fill an 8k-token context window by concatenating chunks in rank order
     until the token limit is reached.
  4. Send: system_prompt + context_chunks + query → LLM → answer.

Design decisions:
  - Context is strictly sorted by similarity score (best first).
  - The 8k window is a hard limit matching the paper's setting.
    Chunks that would overflow the limit are excluded entirely
    (no partial truncation) to preserve coherent text boundaries.
  - A single LLM call produces the final answer (no map-reduce).
    This is the key structural difference from GraphRAG.
  - FAISS index is loaded lazily on first query and cached in memory.
  - All LLM calls go through LLMService.complete() for consistent
    retry, logging, and token tracking.

Paper reference: Section 3.2 paragraph "Semantic Search baseline":
  "We embed the query and retrieve the top-k most similar text chunks,
   filling the 8,192-token context window, then generate a single answer."

Usage:
    engine = VectorRAGEngine.from_settings()
    answer = await engine.query("What are the main themes in this corpus?")
    print(answer.answer)        # the LLM answer
    print(answer.chunks_retrieved)  # how many chunks were used
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.models.response_models import RetrievedChunk, TokenUsage, VectorRAGAnswer
from app.services.embedding_service import EmbeddingService
from app.services.faiss_service import FAISSService
from app.services.llm_service import LLMService
from app.services.tokenizer_service import TokenizerService
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a document corpus. "
    "Use ONLY the provided document excerpts to construct your answer. "
    "Be comprehensive and specific. "
    "If the excerpts do not contain enough information to answer fully, "
    "say so clearly."
)

_ANSWER_PROMPT = """\
The following document excerpts were retrieved as most relevant to the question below.
Use them to provide a thorough, well-structured answer.

---
{context}
---

Question: {query}

Answer:"""


class VectorRAGEngine:
    """
    VectorRAG (Semantic Search) query engine.

    Embeds the query → FAISS top-k → fill 8k context → single LLM call.
    This is the paper's 'SS' baseline (Section 3.2).

    The engine is stateful: the FAISS index and embedding service are
    loaded lazily on first use and kept in memory for subsequent queries.

    Usage:
        engine = VectorRAGEngine.from_settings()

        # Simple query
        answer = await engine.query("What are the main themes?")

        # Full control
        answer = await engine.query(
            query="What are the main themes?",
            top_k=15,
            include_context=True,
        )
    """

    def __init__(
        self,
        openai_service: LLMService,
        embedding_service: EmbeddingService,
        faiss_service: FAISSService,
        tokenizer: TokenizerService,
        faiss_index_path: Path,
        embeddings_metadata_path: Path,
        context_window: int = 8000,
    ) -> None:
        """
        Args:
            openai_service:          For generating the final answer.
            embedding_service:       For embedding the query vector.
            faiss_service:           The FAISS index for top-k retrieval.
            tokenizer:               For counting tokens in context chunks.
            faiss_index_path:        Path to faiss_index.bin.
            embeddings_metadata_path: Path to embeddings.json (chunk metadata).
            context_window:          Hard token limit for context. Paper: 8000.
        """
        self.openai_service           = openai_service
        self.embedding_service        = embedding_service
        self.faiss_service            = faiss_service
        self.tokenizer                = tokenizer
        self.faiss_index_path         = Path(faiss_index_path)
        self.embeddings_metadata_path = Path(embeddings_metadata_path)
        self.context_window           = context_window

        self._index_loaded = False

    # ── Public query interface ─────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        top_k: int = 10,
        include_context: bool = True,
        include_token_usage: bool = True,
        score_threshold: float = 0.0,
    ) -> VectorRAGAnswer:
        """
        Answer a question using VectorRAG (semantic search + LLM).

        Args:
            query:               The question to answer.
            top_k:               Number of chunks to retrieve. Default: 10.
                                 Retrieved chunks are trimmed to fit the 8k window.
            include_context:     If True, include the retrieved chunks in the
                                 response (for transparency / debugging).
            include_token_usage: If True, include token usage in the response.
            score_threshold:     Minimum FAISS similarity score to include a chunk.
                                 Default: 0.0 (include all top-k results).

        Returns:
            VectorRAGAnswer with answer text, chunk count, token usage, and
            optionally the retrieved chunk objects.
        """
        t0 = time.monotonic()
        query = query.strip()

        log.info("VectorRAG query started", query_preview=query[:80], top_k=top_k)

        # 1. Ensure index is loaded
        self._ensure_index_loaded()

        # 2. Embed the query
        query_vector = await self.embedding_service.embed_text(query)

        # 3. FAISS top-k retrieval
        search_results = self.faiss_service.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        if not search_results:
            log.warning("VectorRAG: FAISS returned no results", query=query)
            return self._empty_answer(query, latency_ms=(time.monotonic() - t0) * 1000)

        # 4. Fill 8k context window (chunks in rank order, stop before overflow)
        context_chunks, context_text, context_tokens = self._fill_context_window(
            search_results, query
        )

        # 5. Build prompt and call LLM
        prompt = _ANSWER_PROMPT.format(context=context_text, query=query)
        completion = await self.openai_service.complete(
            user_prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
        )

        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        # 6. Assemble response
        retrieved_chunks = None
        if include_context:
            retrieved_chunks = [
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    source_document=r.source_document,
                    text=r.text,
                    similarity_score=round(r.score, 4),
                    token_count=r.token_count,
                )
                for r in context_chunks
            ]

        token_usage = None
        if include_token_usage:
            token_usage = TokenUsage(
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.total_tokens,
                estimated_cost_usd=completion.estimated_cost_usd,
            )

        log.info(
            "VectorRAG query complete",
            chunks_retrieved=len(context_chunks),
            context_tokens=context_tokens,
            answer_tokens=completion.completion_tokens,
            latency_ms=latency_ms,
        )

        return VectorRAGAnswer(
            answer=completion.content,
            query=query,
            chunks_retrieved=len(context_chunks),
            context_tokens_used=context_tokens,
            context=retrieved_chunks,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    # ── Context window management ──────────────────────────────────────────────

    def _fill_context_window(
        self,
        search_results: list,
        query: str,
    ) -> tuple[list, str, int]:
        """
        Fill the 8k context window with chunks in rank order.

        Iterates through FAISS results (best similarity first) and adds each
        chunk to the context until adding the next chunk would exceed the
        context_window token limit. The query itself consumes some tokens,
        so we reserve space for it plus the prompt overhead.

        Returns:
            Tuple of:
              - list of SearchResult objects actually included
              - context text (newline-separated chunk texts)
              - total token count of the context text
        """
        # Reserve tokens for the prompt template and query itself
        overhead = self.tokenizer.count_tokens(
            _ANSWER_PROMPT.format(context="", query=query)
        )
        available_tokens = self.context_window - overhead

        included = []
        total_tokens = 0
        chunk_texts: list[str] = []

        for result in search_results:
            chunk_tokens = self.tokenizer.count_tokens(result.text)
            if total_tokens + chunk_tokens > available_tokens:
                # Adding this chunk would overflow the window — stop here
                # (never truncate mid-chunk; whole-chunk boundaries only)
                break
            included.append(result)
            chunk_texts.append(
                f"[Source: {result.source_document}, Score: {result.score:.3f}]\n"
                f"{result.text}"
            )
            total_tokens += chunk_tokens

        context_text = "\n\n---\n\n".join(chunk_texts)

        log.debug(
            "Context window filled",
            total_results=len(search_results),
            included=len(included),
            context_tokens=total_tokens,
            available_tokens=available_tokens,
        )

        return included, context_text, total_tokens

    # ── Index lifecycle ────────────────────────────────────────────────────────

    def _ensure_index_loaded(self) -> None:
        """Load the FAISS index from disk if not already loaded."""
        if self._index_loaded:
            return

        if not self.faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.faiss_index_path}. "
                "Run the indexing pipeline (Stage 5) first."
            )
        if not self.embeddings_metadata_path.exists():
            raise FileNotFoundError(
                f"Embeddings metadata not found at {self.embeddings_metadata_path}. "
                "Run the indexing pipeline (Stage 5) first."
            )

        log.info(
            "Loading FAISS index",
            index_path=str(self.faiss_index_path),
            metadata_path=str(self.embeddings_metadata_path),
        )
        self.faiss_service.load(
            index_path=self.faiss_index_path,
            metadata_path=self.embeddings_metadata_path,
        )
        self._index_loaded = True

        log.info(
            "FAISS index loaded",
            total_vectors=self.faiss_service.total_vectors,
        )

    def reload_index(self) -> None:
        """Force reload the FAISS index from disk (e.g. after re-indexing)."""
        self._index_loaded = False
        self._ensure_index_loaded()

    # ── Utility ────────────────────────────────────────────────────────────────

    def _empty_answer(self, query: str, latency_ms: float) -> VectorRAGAnswer:
        """Return a graceful empty answer when no chunks are retrieved."""
        return VectorRAGAnswer(
            answer=(
                "I could not find any relevant document excerpts to answer this question. "
                "The corpus may not contain information about this topic."
            ),
            query=query,
            chunks_retrieved=0,
            context_tokens_used=0,
            context=[],
            token_usage=None,
            latency_ms=latency_ms,
        )

    def get_index_stats(self) -> dict:
        """Return stats about the current FAISS index state."""
        if not self._index_loaded:
            return {"loaded": False, "total_vectors": 0}
        return {
            "loaded": True,
            "total_vectors": self.faiss_service.total_vectors,
            "embedding_dim": self.faiss_service.embedding_dim,
            "index_path": str(self.faiss_index_path),
        }

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> "VectorRAGEngine":
        """Build a VectorRAGEngine from application settings."""
        from app.config import get_settings
        settings = get_settings()

        llm_svc = LLMService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            max_tokens=settings.openai_max_tokens,
            temperature=0.0,
        )
        embedding_svc = EmbeddingService(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
        )
        faiss_svc = FAISSService(
            embedding_dim=settings.embedding_dimension,
        )
        tokenizer = TokenizerService(model=settings.gemini_model)

        return cls(
            openai_service=llm_svc,
            embedding_service=embedding_svc,
            faiss_service=faiss_svc,
            tokenizer=tokenizer,
            faiss_index_path=settings.faiss_index_path,
            embeddings_metadata_path=settings.embeddings_path.with_suffix(".json"),
            context_window=settings.context_window_size,
        )


def get_vectorrag_engine() -> VectorRAGEngine:
    """FastAPI dependency: return a VectorRAGEngine built from settings."""
    return VectorRAGEngine.from_settings()


__all__ = [
    "VectorRAGEngine",
    "get_vectorrag_engine",
]