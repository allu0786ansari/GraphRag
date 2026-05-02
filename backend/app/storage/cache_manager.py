"""
storage/cache_manager.py — Per-chunk extraction cache for resumable pipeline.

Tracks which pipeline stages have been completed for each chunk so that
a crashed or interrupted pipeline can resume exactly where it left off
without re-processing already-completed work.

Persistent state file:
  {artifacts_dir}/pipeline_state.json

  {
    "pipeline_run_id":   "run_20241115_143022",
    "started_at":        "2024-11-15T14:30:22",
    "last_updated_at":   "2024-11-15T16:45:01",
    "total_chunks":      1702,
    "extracted_chunks":  set → stored as list in JSON,
    "failed_chunks":     {"chunk_id": "error message"},
    "stage_completed":   {"chunking": true, "extraction": false, ...}
  }

Resume logic:
  When a pipeline run starts:
    1. CacheManager.load_state() reads the state file.
    2. filter_pending_chunks(all_chunks) returns only chunks not yet extracted.
    3. The extraction loop processes only pending chunks.
    4. After each chunk: mark_extracted(chunk_id).
    5. After all chunks: mark_stage_complete("extraction").

  On crash and restart:
    1. State file is read — extracted_chunks contains all completed chunks.
    2. filter_pending_chunks() returns only the ones not in extracted_chunks.
    3. Pipeline continues from where it stopped.

The state file is updated atomically (same .tmp → rename pattern) to prevent
corruption if the process is killed during a write.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.models.graph_models import ChunkSchema
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── File name constant ─────────────────────────────────────────────────────────
STATE_FILENAME = "pipeline_state.json"

# ── Pipeline stage names ───────────────────────────────────────────────────────
PipelineStageName = Literal[
    "chunking",
    "extraction",
    "graph_construction",
    "community_detection",
    "summarization",
    "embedding",
]

ALL_STAGES: list[PipelineStageName] = [
    "chunking",
    "extraction",
    "graph_construction",
    "community_detection",
    "summarization",
    "embedding",
]


class CacheManager:
    """
    Per-chunk extraction cache for resumable pipeline execution.

    Maintains a persistent JSON state file tracking:
      - Which chunks have been successfully extracted
      - Which chunks failed and why
      - Which high-level pipeline stages are complete
      - Pipeline run metadata (id, timestamps, total chunks)

    This enables the pipeline to resume from exactly the right point
    after a crash, rate limit, or manual interruption.

    Usage:
        cache = CacheManager(artifacts_dir=settings.artifacts_dir)

        # At pipeline start
        cache.initialize_run(total_chunks=len(all_chunks))

        # Get only the chunks that still need processing
        pending = cache.filter_pending_chunks(all_chunks)

        # After each chunk is processed
        cache.mark_extracted(chunk.chunk_id)

        # After each chunk fails
        cache.mark_failed(chunk.chunk_id, error_message="Rate limit")

        # After all extraction is done
        cache.mark_stage_complete("extraction")

        # Check if a stage is already done (for full reindex logic)
        if cache.is_stage_complete("chunking"):
            skip_chunking()
    """

    def __init__(self, artifacts_dir: Path | str) -> None:
        """
        Args:
            artifacts_dir: Directory containing pipeline_state.json.
                           Created if it does not exist.
                           Always use settings.artifacts_dir — never hardcode.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.artifacts_dir / STATE_FILENAME

        # In-memory state — synced to disk after every mutation
        self._state: dict = self._empty_state()

        # Load existing state if available
        if self._state_path.exists():
            self._load_state_from_disk()

        log.debug(
            "CacheManager initialized",
            artifacts_dir=str(self.artifacts_dir),
            state_exists=self._state_path.exists(),
        )

    # ── State initialization ───────────────────────────────────────────────────

    def initialize_run(
        self,
        total_chunks: int,
        run_id: str | None = None,
        force_reset: bool = False,
    ) -> str:
        """
        Initialize or resume a pipeline run.

        If a state file exists and force_reset=False: loads the existing
        state (resume mode — previously extracted chunks are preserved).

        If force_reset=True or no state file exists: creates a fresh
        state (full reindex mode).

        Args:
            total_chunks: Total number of chunks to be processed.
                          Stored for progress tracking.
            run_id:       Optional run identifier. If None, generates
                          one from the current timestamp.
            force_reset:  If True, wipe the existing state and start fresh.

        Returns:
            The pipeline run_id string.
        """
        if force_reset or not self._state_path.exists():
            run_id = run_id or _generate_run_id()
            self._state = self._empty_state()
            self._state["pipeline_run_id"] = run_id
            self._state["started_at"] = _now_iso()
            self._state["total_chunks"] = total_chunks
            self._persist()

            log.info(
                "Pipeline run initialized",
                run_id=run_id,
                total_chunks=total_chunks,
                mode="fresh",
            )
        else:
            # Resume: update total_chunks in case corpus changed
            self._state["total_chunks"] = total_chunks
            self._state["last_updated_at"] = _now_iso()
            self._persist()

            extracted = len(self._state.get("extracted_chunks", []))
            log.info(
                "Pipeline run resumed",
                run_id=self._state.get("pipeline_run_id", "unknown"),
                total_chunks=total_chunks,
                already_extracted=extracted,
                remaining=total_chunks - extracted,
            )

        return self._state["pipeline_run_id"]

    # ── Per-chunk tracking ─────────────────────────────────────────────────────

    def mark_extracted(self, chunk_id: str) -> None:
        """
        Mark a chunk as successfully extracted.

        Called immediately after each chunk's extraction completes.
        Persists atomically to disk so a crash right after this call
        does not re-process the chunk.

        Args:
            chunk_id: The chunk_id of the successfully processed chunk.
        """
        extracted: list = self._state.setdefault("extracted_chunks", [])
        if chunk_id not in extracted:
            extracted.append(chunk_id)

        # Remove from failed if it was previously failed and now succeeded
        failed: dict = self._state.setdefault("failed_chunks", {})
        failed.pop(chunk_id, None)

        self._state["last_updated_at"] = _now_iso()
        self._persist()

    def mark_failed(self, chunk_id: str, error_message: str) -> None:
        """
        Mark a chunk as failed with an error message.

        Failed chunks are NOT retried automatically — they are skipped
        on resume. Use this for unrecoverable errors (e.g. malformed chunk
        text that the LLM cannot parse). For transient errors (rate limits,
        timeouts), the retry logic in OpenAIService handles those before
        this method is called.

        Args:
            chunk_id:      The chunk_id that failed.
            error_message: Human-readable error description for debugging.
        """
        failed: dict = self._state.setdefault("failed_chunks", {})
        failed[chunk_id] = {
            "error": error_message,
            "failed_at": _now_iso(),
        }
        self._state["last_updated_at"] = _now_iso()
        self._persist()

        log.warning(
            "Chunk marked as failed",
            chunk_id=chunk_id,
            error=error_message[:200],
        )

    def is_extracted(self, chunk_id: str) -> bool:
        """Return True if this chunk has already been successfully extracted."""
        return chunk_id in self._state.get("extracted_chunks", [])

    def is_failed(self, chunk_id: str) -> bool:
        """Return True if this chunk previously failed and should be skipped."""
        return chunk_id in self._state.get("failed_chunks", {})

    def filter_pending_chunks(self, all_chunks: list[ChunkSchema]) -> list[ChunkSchema]:
        """
        Return only the chunks that have not yet been extracted or failed.

        This is the main resume method. Call at the start of the extraction
        loop to get only the chunks that still need processing.

        Args:
            all_chunks: The full list of ChunkSchema objects.

        Returns:
            Subset of all_chunks where chunk_id is not in extracted_chunks
            and not in failed_chunks.

        Example:
            all_chunks = store.load_chunks()
            pending    = cache.filter_pending_chunks(all_chunks)
            log.info(f"Resuming: {len(pending)}/{len(all_chunks)} chunks remaining")
            for chunk in pending:
                result = await extract(chunk)
                cache.mark_extracted(chunk.chunk_id)
        """
        extracted: set[str] = set(self._state.get("extracted_chunks", []))
        failed:    set[str] = set(self._state.get("failed_chunks", {}).keys())
        done = extracted | failed

        pending = [c for c in all_chunks if c.chunk_id not in done]

        log.info(
            "Pending chunks computed",
            total=len(all_chunks),
            extracted=len(extracted),
            failed=len(failed),
            pending=len(pending),
        )
        return pending

    def filter_pending_chunk_ids(self, all_chunk_ids: list[str]) -> list[str]:
        """
        Same as filter_pending_chunks() but works with IDs directly.

        Useful when you have chunk IDs without full ChunkSchema objects.
        """
        extracted: set[str] = set(self._state.get("extracted_chunks", []))
        failed:    set[str] = set(self._state.get("failed_chunks", {}).keys())
        done = extracted | failed
        return [cid for cid in all_chunk_ids if cid not in done]

    # ── Stage-level tracking ───────────────────────────────────────────────────

    def mark_stage_complete(self, stage: PipelineStageName) -> None:
        """
        Mark an entire pipeline stage as complete.

        Called once after all chunks in a stage have been processed.
        Allows the pipeline runner to skip completed stages entirely
        on a resume (not just individual chunks).

        Args:
            stage: One of the stage name strings defined in ALL_STAGES.
        """
        completed: dict = self._state.setdefault("stage_completed", {})
        completed[stage] = True
        completed[f"{stage}_completed_at"] = _now_iso()
        self._state["last_updated_at"] = _now_iso()
        self._persist()

        log.info("Pipeline stage marked complete", stage=stage)

    def is_stage_complete(self, stage: PipelineStageName) -> bool:
        """
        Return True if the specified pipeline stage has been completed.

        Used by the pipeline runner to skip stages that are already done:
          if cache.is_stage_complete("chunking"):
              chunks = store.load_chunks()  # just load from disk
          else:
              chunks = run_chunking_stage()
              store.save_chunks(chunks)
              cache.mark_stage_complete("chunking")
        """
        return bool(self._state.get("stage_completed", {}).get(stage, False))

    def get_completed_stages(self) -> list[str]:
        """Return list of stage names that are marked complete."""
        return [
            stage for stage in ALL_STAGES
            if self.is_stage_complete(stage)
        ]

    def get_next_pending_stage(self) -> PipelineStageName | None:
        """
        Return the name of the next stage that needs to run.

        Returns None if all stages are complete (fully indexed).

        Example:
            next_stage = cache.get_next_pending_stage()
            # → "extraction" (if chunking is done but extraction is not)
        """
        for stage in ALL_STAGES:
            if not self.is_stage_complete(stage):
                return stage
        return None

    def reset_stage(self, stage: PipelineStageName) -> None:
        """
        Mark a stage as NOT complete (force it to re-run).

        Also clears per-chunk extraction state for stages that use it.
        Used when force_reindex=True for specific stages.
        """
        completed: dict = self._state.get("stage_completed", {})
        completed.pop(stage, None)
        completed.pop(f"{stage}_completed_at", None)

        # For extraction stage, also clear per-chunk state
        if stage == "extraction":
            self._state["extracted_chunks"] = []
            self._state["failed_chunks"] = {}

        self._persist()
        log.info("Pipeline stage reset", stage=stage)

    # ── Progress reporting ─────────────────────────────────────────────────────

    def get_progress(self) -> dict:
        """
        Return a progress summary for the current pipeline run.

        Used by the pipeline status endpoint and logging.

        Returns:
            Dict with counts, percentages, and stage status.
        """
        total    = self._state.get("total_chunks", 0)
        extracted = len(self._state.get("extracted_chunks", []))
        failed   = len(self._state.get("failed_chunks", {}))
        pending  = max(0, total - extracted - failed)

        pct_done = round(extracted / total * 100, 1) if total > 0 else 0.0

        return {
            "run_id":          self._state.get("pipeline_run_id", "unknown"),
            "started_at":      self._state.get("started_at"),
            "last_updated_at": self._state.get("last_updated_at"),
            "total_chunks":    total,
            "extracted":       extracted,
            "failed":          failed,
            "pending":         pending,
            "pct_complete":    pct_done,
            "stages": {
                stage: self.is_stage_complete(stage)
                for stage in ALL_STAGES
            },
            "next_stage":      self.get_next_pending_stage(),
        }

    def get_failed_chunks(self) -> dict[str, dict]:
        """
        Return all failed chunks with their error messages.

        Returns:
            Dict mapping chunk_id → {"error": str, "failed_at": str}.
        """
        return dict(self._state.get("failed_chunks", {}))

    def extraction_completion_rate(self) -> float:
        """
        Return the fraction of chunks successfully extracted (0.0–1.0).

        Returns 0.0 if no chunks have been processed yet.
        """
        total = self._state.get("total_chunks", 0)
        if total == 0:
            return 0.0
        extracted = len(self._state.get("extracted_chunks", []))
        return extracted / total

    # ── Full reset ─────────────────────────────────────────────────────────────

    def reset_all(self) -> None:
        """
        Completely reset the pipeline state.

        Called when force_reindex=True — wipes all extraction state so
        the pipeline runs from scratch.

        Does NOT delete artifact files — that is handled by ArtifactStore,
        GraphStore, and SummaryStore's delete_all() methods.
        """
        self._state = self._empty_state()
        if self._state_path.exists():
            self._state_path.unlink()
        log.info("Pipeline state fully reset")

    def delete_state_file(self) -> bool:
        """Delete the state file from disk. Returns True if deleted."""
        if self._state_path.exists():
            self._state_path.unlink()
            self._state = self._empty_state()
            log.info("Deleted pipeline_state.json")
            return True
        return False

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """
        Atomically write current state to pipeline_state.json.

        Uses .tmp in the same directory (not system temp) for atomic writes.
        Retries with delays to handle file locks (especially OneDrive).
        """
        import time
        
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in the same directory to avoid cross-device issues on Windows
        tmp_path = self._state_path.parent / f".{STATE_FILENAME}.tmp.{os.getpid()}"
        
        try:
            # Write to temp file
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            
            # Retry renaming with exponential backoff to handle OneDrive locks
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.replace(str(tmp_path), str(self._state_path))
                    return
                except OSError as e:
                    if attempt < max_retries - 1:
                        delay = 0.1 * (2 ** attempt)  # exponential backoff
                        time.sleep(delay)
                    else:
                        raise
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    def _load_state_from_disk(self) -> None:
        """Load state from the existing state file."""
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self._state = loaded
                log.debug(
                    "Pipeline state loaded from disk",
                    run_id=loaded.get("pipeline_run_id", "unknown"),
                    extracted=len(loaded.get("extracted_chunks", [])),
                )
            else:
                log.warning(
                    "pipeline_state.json has unexpected format, starting fresh",
                    found_type=type(loaded).__name__,
                )
                self._state = self._empty_state()
        except Exception as e:
            log.warning(
                "Could not load pipeline state, starting fresh",
                error=str(e),
            )
            self._state = self._empty_state()

    @staticmethod
    def _empty_state() -> dict:
        """Return a fresh, empty pipeline state dict."""
        return {
            "pipeline_run_id":  "",
            "started_at":       None,
            "last_updated_at":  None,
            "total_chunks":     0,
            "extracted_chunks": [],
            "failed_chunks":    {},
            "stage_completed":  {stage: False for stage in ALL_STAGES},
        }

    # ── State file info ────────────────────────────────────────────────────────

    def state_exists(self) -> bool:
        """True if pipeline_state.json exists."""
        return self._state_path.exists()

    @property
    def run_id(self) -> str:
        """Current pipeline run ID."""
        return self._state.get("pipeline_run_id", "")

    def __repr__(self) -> str:
        progress = self.get_progress()
        return (
            f"CacheManager("
            f"run_id={progress['run_id']!r}, "
            f"extracted={progress['extracted']}/{progress['total_chunks']}, "
            f"pct={progress['pct_complete']}%)"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _generate_run_id() -> str:
    """Generate a unique pipeline run ID from current timestamp."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


# ── Factory function ───────────────────────────────────────────────────────────

def get_cache_manager() -> CacheManager:
    """Build a CacheManager from application settings."""
    from app.config import get_settings
    settings = get_settings()
    return CacheManager(artifacts_dir=settings.artifacts_dir)


__all__ = [
    "CacheManager",
    "get_cache_manager",
    "ALL_STAGES",
    "STATE_FILENAME",
]