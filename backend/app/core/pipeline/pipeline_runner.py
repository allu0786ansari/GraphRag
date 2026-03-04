"""
core/pipeline/pipeline_runner.py — Orchestrate all indexing pipeline stages.

Wires together all pipeline stages in dependency order:
  1. chunking.py          → chunks.json
  2. extraction.py        → extractions.json   (with gleaning.py)
  3. graph_builder.py     → graph.pkl
  4. community_detection  → community_map.json
  5. summarization.py     → community_summaries.json
  6. Embedding + FAISS    → faiss_index.bin + embeddings.npy

Key features:
  - Resumable: reads CacheManager state before each stage to skip completed work.
  - Checkpointing: saves each stage's artifact before starting the next.
  - Incremental extraction: per-chunk save via ArtifactStore.append_extraction()
    so a crash mid-extraction loses at most one chunk's work.
  - Progress callbacks: optional on_progress(stage, pct) for the API endpoint.
  - force_reindex: when True, deletes all artifacts and starts from scratch.
  - max_chunks: limits total chunks processed (for development / cost control).

Usage:
    runner = PipelineRunner.from_settings()
    result = await runner.run()
    # → PipelineResult with artifact counts and timing
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.storage.artifact_store import ArtifactStore
from app.storage.cache_manager import CacheManager, ALL_STAGES
from app.storage.graph_store import GraphStore
from app.storage.summary_store import SummaryStore
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Result of a complete pipeline run.

    All timing values are in seconds. Artifact counts reflect what was
    produced in this run (not cumulative).
    """
    success: bool
    run_id: str
    total_elapsed_seconds: float

    # Stage timing
    stage_elapsed: dict[str, float] = field(default_factory=dict)

    # Artifact counts
    chunks_count: int = 0
    extractions_count: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    communities_count: int = 0
    summaries_count: int = 0
    embeddings_count: int = 0

    # Stage completion flags
    stages_completed: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)

    # Error info (if success=False)
    error_stage: str | None = None
    error_message: str | None = None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAILED at {self.error_stage}"
        return (
            f"PipelineResult({status}, "
            f"chunks={self.chunks_count}, "
            f"nodes={self.graph_nodes}, "
            f"communities={self.communities_count}, "
            f"elapsed={self.total_elapsed_seconds:.1f}s)"
        )


# ── Runner ─────────────────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Full indexing pipeline orchestrator.

    Runs all stages in dependency order with checkpointing and resume.
    Each stage checks CacheManager to skip already-completed work.

    Usage:
        runner = PipelineRunner.from_settings()
        result = await runner.run(
            force_reindex=False,   # resume from where we left off
            max_chunks=50,         # limit for development
        )
    """

    def __init__(
        self,
        raw_data_dir: Path,
        artifacts_dir: Path,
        openai_api_key: str,
        openai_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        gleaning_rounds: int = 2,
        context_window: int = 8000,
        max_concurrency: int = 20,
        min_entity_mentions: int = 1,
        community_max_levels: int = 3,
        skip_embedding: bool = False,
        skip_claims: bool = False,
    ) -> None:
        self.raw_data_dir       = Path(raw_data_dir)
        self.artifacts_dir      = Path(artifacts_dir)
        self.openai_api_key     = openai_api_key
        self.openai_model       = openai_model
        self.embedding_model    = embedding_model
        self.chunk_size         = chunk_size
        self.chunk_overlap      = chunk_overlap
        self.gleaning_rounds    = gleaning_rounds
        self.context_window     = context_window
        self.max_concurrency    = max_concurrency
        self.min_entity_mentions = min_entity_mentions
        self.community_max_levels = community_max_levels
        self.skip_embedding     = skip_embedding
        self.skip_claims        = skip_claims

        # Storage layer
        self.artifact_store = ArtifactStore(artifacts_dir=self.artifacts_dir)
        self.graph_store    = GraphStore(artifacts_dir=self.artifacts_dir)
        self.summary_store  = SummaryStore(artifacts_dir=self.artifacts_dir)
        self.cache          = CacheManager(artifacts_dir=self.artifacts_dir)

        # Services (built lazily to avoid requiring API key at init time)
        self._tokenizer     = None
        self._openai_svc    = None
        self._embedding_svc = None

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(
        self,
        force_reindex: bool = False,
        max_chunks: int | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> PipelineResult:
        """
        Run the complete indexing pipeline.

        Args:
            force_reindex: If True, delete all existing artifacts and
                           reindex from scratch.
            max_chunks:    Maximum chunks to process. None = process all.
                           Use small values (50–100) for development.
            on_progress:   Optional callback(stage_name, pct_complete)
                           called as each stage progresses.

        Returns:
            PipelineResult with counts, timing, and success status.
        """
        t0 = time.monotonic()
        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0.0)

        try:
            # ── Pre-run setup ──────────────────────────────────────────────────
            if force_reindex:
                log.info("Force reindex requested — deleting all artifacts")
                self._delete_all_artifacts()

            self._build_services()

            # ── Stage 1: Chunking ──────────────────────────────────────────────
            chunks = await self._run_chunking(result, max_chunks, on_progress)
            if chunks is None:
                return result  # fatal error

            # Initialize cache now we know the chunk count
            run_id = self.cache.initialize_run(
                total_chunks=len(chunks),
                force_reset=force_reindex,
            )
            result.run_id = run_id
            result.chunks_count = len(chunks)

            if on_progress:
                on_progress("chunking", 1.0)

            # ── Stage 2: Extraction ────────────────────────────────────────────
            extractions = await self._run_extraction(result, chunks, on_progress)
            if extractions is None:
                return result

            result.extractions_count = len(extractions)
            if on_progress:
                on_progress("extraction", 1.0)

            # ── Stage 3: Graph construction ────────────────────────────────────
            graph = await self._run_graph_construction(result, extractions, on_progress)
            if graph is None:
                return result

            result.graph_nodes = graph.number_of_nodes()
            result.graph_edges = graph.number_of_edges()
            if on_progress:
                on_progress("graph_construction", 1.0)

            # ── Stage 4: Community detection ───────────────────────────────────
            communities = await self._run_community_detection(result, graph, on_progress)
            if communities is None:
                return result

            result.communities_count = len(communities)
            if on_progress:
                on_progress("community_detection", 1.0)

            # ── Stage 5: Summarization ─────────────────────────────────────────
            summaries = await self._run_summarization(result, communities, graph, on_progress)
            if summaries is None:
                return result

            result.summaries_count = len(summaries)
            if on_progress:
                on_progress("summarization", 1.0)

            # ── Stage 6: Embedding + FAISS (optional) ──────────────────────────
            if not self.skip_embedding:
                n_embedded = await self._run_embedding(result, chunks, on_progress)
                result.embeddings_count = n_embedded or 0
                if on_progress:
                    on_progress("embedding", 1.0)
            else:
                log.info("Embedding stage skipped (skip_embedding=True)")
                result.stages_skipped.append("embedding")

            # ── Done ───────────────────────────────────────────────────────────
            result.success = True
            result.total_elapsed_seconds = round(time.monotonic() - t0, 2)

            log.info(
                "Pipeline completed successfully",
                run_id=result.run_id,
                chunks=result.chunks_count,
                extractions=result.extractions_count,
                nodes=result.graph_nodes,
                edges=result.graph_edges,
                communities=result.communities_count,
                summaries=result.summaries_count,
                elapsed_seconds=result.total_elapsed_seconds,
            )

        except Exception as e:
            result.total_elapsed_seconds = round(time.monotonic() - t0, 2)
            result.error_message = str(e)
            log.error(
                "Pipeline failed with unexpected error",
                error=str(e),
                elapsed_seconds=result.total_elapsed_seconds,
            )

        return result

    # ── Stage implementations ──────────────────────────────────────────────────

    async def _run_chunking(
        self,
        result: PipelineResult,
        max_chunks: int | None,
        on_progress: Callable | None,
    ):
        """Stage 1: chunk all raw documents."""
        stage = "chunking"

        if self.cache.is_stage_complete(stage) and self.artifact_store.chunks_exist():
            log.info("Chunking already complete — loading from disk")
            chunks = self.artifact_store.load_chunks()
            result.stages_skipped.append(stage)
            if max_chunks and len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
            return chunks

        log.info("Stage 1: Chunking")
        t0 = time.monotonic()

        try:
            from app.core.pipeline.chunking import ChunkingPipeline
            pipeline = ChunkingPipeline(tokenizer=self._tokenizer)
            chunks = pipeline.run(
                raw_data_dir=self.raw_data_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_chunks=max_chunks,
            )

            if not chunks:
                result.error_stage = stage
                result.error_message = "No chunks produced — check raw_data_dir"
                log.error("Chunking produced no chunks", raw_data_dir=str(self.raw_data_dir))
                return None

            self.artifact_store.save_chunks(chunks)
            self.cache.mark_stage_complete(stage)
            result.stages_completed.append(stage)
            result.stage_elapsed[stage] = round(time.monotonic() - t0, 2)

            log.info("Chunking complete", chunks=len(chunks))
            return chunks

        except Exception as e:
            result.error_stage = stage
            result.error_message = str(e)
            log.error("Chunking stage failed", error=str(e))
            return None

    async def _run_extraction(
        self,
        result: PipelineResult,
        chunks: list,
        on_progress: Callable | None,
    ):
        """Stage 2: extract entities + relationships from each chunk."""
        stage = "extraction"

        if self.cache.is_stage_complete(stage) and self.artifact_store.extractions_exist():
            log.info("Extraction already complete — loading from disk")
            extractions = self.artifact_store.load_extractions()
            result.stages_skipped.append(stage)
            return extractions

        log.info("Stage 2: Extraction + Gleaning")
        t0 = time.monotonic()

        try:
            from app.core.pipeline.extraction import ExtractionPipeline
            from app.core.pipeline.gleaning import GleaningLoop

            gleaning_loop = GleaningLoop(self._openai_svc) if self.gleaning_rounds > 0 else None
            pipeline = ExtractionPipeline(
                openai_service=self._openai_svc,
                tokenizer=self._tokenizer,
                gleaning_loop=gleaning_loop,
                skip_claims=self.skip_claims,
            )

            # Only process chunks not yet extracted (resume support)
            pending_chunks = self.cache.filter_pending_chunks(chunks)
            log.info(
                "Extraction pending chunks",
                total=len(chunks),
                pending=len(pending_chunks),
                already_done=len(chunks) - len(pending_chunks),
            )

            # Per-chunk callback: save + mark immediately after each chunk
            completed_count = [0]

            async def _on_chunk_done(extraction) -> None:
                self.artifact_store.append_extraction(extraction)
                if extraction.extraction_completed:
                    self.cache.mark_extracted(extraction.chunk_id)
                else:
                    self.cache.mark_failed(
                        extraction.chunk_id,
                        extraction.error_message or "unknown error",
                    )
                completed_count[0] += 1
                if on_progress and len(pending_chunks) > 0:
                    pct = completed_count[0] / len(pending_chunks)
                    on_progress(stage, pct)

            await pipeline.extract_chunks_batch(
                chunks=pending_chunks,
                gleaning_rounds=self.gleaning_rounds,
                max_concurrency=self.max_concurrency,
                on_chunk_complete=_on_chunk_done,
            )

            self.cache.mark_stage_complete(stage)
            result.stages_completed.append(stage)
            result.stage_elapsed[stage] = round(time.monotonic() - t0, 2)

            # Load full extractions (pending + previously saved)
            extractions = self.artifact_store.load_extractions()
            log.info("Extraction complete", extractions=len(extractions))
            return extractions

        except Exception as e:
            result.error_stage = stage
            result.error_message = str(e)
            log.error("Extraction stage failed", error=str(e))
            return None

    async def _run_graph_construction(
        self,
        result: PipelineResult,
        extractions: list,
        on_progress: Callable | None,
    ):
        """Stage 3: build NetworkX graph from extractions."""
        stage = "graph_construction"

        if self.cache.is_stage_complete(stage) and self.graph_store.graph_exists():
            log.info("Graph construction already complete — loading from disk")
            graph = self.graph_store.load_graph()
            result.stages_skipped.append(stage)
            return graph

        log.info("Stage 3: Graph construction")
        t0 = time.monotonic()

        try:
            from app.core.pipeline.graph_builder import GraphBuilder
            builder = GraphBuilder()
            graph = builder.build(
                extractions=extractions,
                min_entity_mentions=self.min_entity_mentions,
            )

            if graph.number_of_nodes() == 0:
                result.error_stage = stage
                result.error_message = "Graph has zero nodes — extraction may have failed"
                log.error("Graph construction produced empty graph")
                return None

            self.graph_store.save_graph(graph)
            self.cache.mark_stage_complete(stage)
            result.stages_completed.append(stage)
            result.stage_elapsed[stage] = round(time.monotonic() - t0, 2)

            log.info(
                "Graph construction complete",
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
            )
            return graph

        except Exception as e:
            result.error_stage = stage
            result.error_message = str(e)
            log.error("Graph construction failed", error=str(e))
            return None

    async def _run_community_detection(
        self,
        result: PipelineResult,
        graph: Any,
        on_progress: Callable | None,
    ):
        """Stage 4: Leiden hierarchical community detection."""
        stage = "community_detection"

        if self.cache.is_stage_complete(stage) and self.graph_store.community_map_exists():
            log.info("Community detection already complete — loading from disk")
            communities = self.graph_store.load_community_map()
            result.stages_skipped.append(stage)
            return communities

        log.info("Stage 4: Community detection")
        t0 = time.monotonic()

        try:
            from app.core.pipeline.community_detection import CommunityDetection
            detector = CommunityDetection(max_cluster_size=10, random_seed=42)
            communities = detector.detect(graph, max_levels=self.community_max_levels)

            if not communities:
                result.error_stage = stage
                result.error_message = "Community detection produced no communities"
                return None

            self.graph_store.save_community_map(communities)
            # Re-save graph with community_ids annotated on nodes
            self.graph_store.save_graph(graph)
            self.cache.mark_stage_complete(stage)
            result.stages_completed.append(stage)
            result.stage_elapsed[stage] = round(time.monotonic() - t0, 2)

            log.info("Community detection complete", communities=len(communities))
            return communities

        except Exception as e:
            result.error_stage = stage
            result.error_message = str(e)
            log.error("Community detection failed", error=str(e))
            return None

    async def _run_summarization(
        self,
        result: PipelineResult,
        communities: list,
        graph: Any,
        on_progress: Callable | None,
    ):
        """Stage 5: generate LLM community summaries."""
        stage = "summarization"

        if self.cache.is_stage_complete(stage) and self.summary_store.summaries_exist():
            log.info("Summarization already complete — loading from disk")
            summaries = self.summary_store.load_summaries()
            result.stages_skipped.append(stage)
            return summaries

        log.info("Stage 5: Summarization")
        t0 = time.monotonic()

        try:
            from app.core.pipeline.summarization import SummarizationPipeline
            pipeline = SummarizationPipeline(
                openai_service=self._openai_svc,
                tokenizer=self._tokenizer,
                context_window=self.context_window,
            )

            completed_count = [0]

            async def _on_summary_done(summary) -> None:
                completed_count[0] += 1
                if on_progress and len(communities) > 0:
                    on_progress(stage, completed_count[0] / len(communities))

            summaries = await pipeline.summarize_all(
                communities=communities,
                graph=graph,
                max_concurrency=self.max_concurrency,
                on_summary_complete=_on_summary_done,
            )

            self.summary_store.save_summaries(summaries)
            self.cache.mark_stage_complete(stage)
            result.stages_completed.append(stage)
            result.stage_elapsed[stage] = round(time.monotonic() - t0, 2)

            log.info("Summarization complete", summaries=len(summaries))
            return summaries

        except Exception as e:
            result.error_stage = stage
            result.error_message = str(e)
            log.error("Summarization failed", error=str(e))
            return None

    async def _run_embedding(
        self,
        result: PipelineResult,
        chunks: list,
        on_progress: Callable | None,
    ) -> int | None:
        """Stage 6: embed all chunks and build FAISS index."""
        stage = "embedding"

        if self.cache.is_stage_complete(stage) and self._faiss_index_exists():
            log.info("Embedding already complete — skipping")
            result.stages_skipped.append(stage)
            n = self.artifact_store.chunks_count()
            return n

        log.info("Stage 6: Embedding + FAISS index")
        t0 = time.monotonic()

        try:
            from app.services.embedding_service import EmbeddingService
            from app.services.faiss_service import FAISSService
            from app.config import get_settings

            settings = get_settings()
            emb_svc = EmbeddingService(
                api_key=self.openai_api_key,
                model=self.embedding_model,
            )
            faiss_svc = FAISSService(embedding_dim=settings.embedding_dimension)

            chunk_texts = [c.text for c in chunks]
            chunk_meta  = [
                {
                    "chunk_id":        c.chunk_id,
                    "text":            c.text,
                    "source_document": c.source_document,
                    "token_count":     c.token_count,
                }
                for c in chunks
            ]

            log.info("Embedding chunks", count=len(chunk_texts))
            embeddings = await emb_svc.embed_batch(
                texts=chunk_texts,
                max_concurrency=min(5, self.max_concurrency),
            )

            faiss_svc.build_index(embeddings, chunk_meta)
            faiss_svc.save(
                index_path=settings.faiss_index_path,
                metadata_path=settings.embeddings_path.with_suffix(".json"),
            )

            # Save raw embeddings as numpy for potential reuse
            import numpy as np
            np.save(str(settings.embeddings_path), embeddings)

            self.cache.mark_stage_complete(stage)
            result.stages_completed.append(stage)
            result.stage_elapsed[stage] = round(time.monotonic() - t0, 2)

            log.info("Embedding complete", chunks_embedded=len(chunk_texts))
            return len(chunk_texts)

        except Exception as e:
            result.error_stage = stage
            result.error_message = str(e)
            log.error("Embedding stage failed", error=str(e))
            return None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_services(self) -> None:
        """Lazily initialize service objects."""
        from app.services.tokenizer_service import TokenizerService
        from app.services.openai_service import OpenAIService
        from app.config import get_settings

        settings = get_settings()

        if self._tokenizer is None:
            self._tokenizer = TokenizerService(model=self.openai_model)

        if self._openai_svc is None:
            self._openai_svc = OpenAIService(
                api_key=self.openai_api_key,
                model=self.openai_model,
                max_tokens=settings.openai_max_tokens,
                temperature=0.0,
                timeout=settings.openai_timeout,
                max_retries=settings.openai_max_retries,
            )

    def _delete_all_artifacts(self) -> None:
        """Wipe all artifacts for force_reindex."""
        self.artifact_store.delete_all()
        self.graph_store.delete_all()
        self.summary_store.delete_all()
        self.cache.reset_all()
        # Delete FAISS + embeddings if they exist
        try:
            from app.config import get_settings
            settings = get_settings()
            for path in [settings.faiss_index_path, settings.embeddings_path]:
                if Path(path).exists():
                    Path(path).unlink()
        except Exception:
            pass

    def _faiss_index_exists(self) -> bool:
        """Check if the FAISS index file exists."""
        try:
            from app.config import get_settings
            settings = get_settings()
            return Path(settings.faiss_index_path).exists()
        except Exception:
            return False

    def get_pipeline_status(self) -> dict:
        """Return current pipeline progress and artifact status."""
        return {
            "progress":   self.cache.get_progress(),
            "artifacts": {
                "chunks":      self.artifact_store.get_stats()["chunks"],
                "extractions": self.artifact_store.get_stats()["extractions"],
                "graph":       self.graph_store.get_graph_stats(),
                "community_map":  {
                    "exists": self.graph_store.community_map_exists(),
                    "counts": self.graph_store.get_community_counts(),
                },
                "summaries": {
                    "exists": self.summary_store.summaries_exist(),
                    "counts": self.summary_store.get_summary_counts(),
                },
            },
        }

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> "PipelineRunner":
        """Build a PipelineRunner from application settings."""
        from app.config import get_settings
        settings = get_settings()
        return cls(
            raw_data_dir=settings.raw_data_dir,
            artifacts_dir=settings.artifacts_dir,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            gleaning_rounds=settings.gleaning_rounds,
            context_window=settings.context_window_size,
            max_concurrency=20,
        )


__all__ = [
    "PipelineRunner",
    "PipelineResult",
]