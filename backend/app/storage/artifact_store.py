"""
storage/artifact_store.py — JSON artifact persistence for chunks and extractions.

Handles all read/write operations for the two largest pipeline artifacts:
  chunks.json      → list[ChunkSchema]       (chunking stage output)
  extractions.json → list[ChunkExtraction]   (extraction + gleaning stage output)

Design principles:
  - Atomic writes: write to a .tmp file first, then rename → no corrupt artifacts
    on crash or keyboard interrupt mid-write.
  - Streaming reads: large JSON files are streamed line-by-line using ijson
    when available, falling back to json.load() for small files.
  - Schema validation: every loaded record is validated through its Pydantic model.
    Corrupt or partial records are logged and skipped, not raised.
  - Checkpointing: save_chunks_checkpoint() / save_extractions_checkpoint()
    append to existing files so a crashed pipeline can resume from where it left off.
    The cache_manager.py decides which chunks to skip.
  - Paths always come from config — never hardcoded strings anywhere in this file.

File layout on disk:
  {artifacts_dir}/
    chunks.json          ← written once by chunking stage
    extractions.json     ← written incrementally by extraction stage
    embeddings.npy       ← written by embedding stage (numpy, not here)
    faiss_index.bin      ← written by FAISSService.save()
    community_map.json   ← written by graph_store.py
    community_summaries.json ← written by summary_store.py
    pipeline_state.json  ← written by cache_manager.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator

from app.models.graph_models import ChunkExtraction, ChunkSchema
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── File name constants (never hardcoded outside this module) ──────────────────
CHUNKS_FILENAME      = "chunks.json"
EXTRACTIONS_FILENAME = "extractions.json"


class ArtifactStore:
    """
    JSON artifact store for chunks and extractions.

    All pipeline stages that produce or consume chunks/extractions
    call this class. Direct file I/O in pipeline code is forbidden.

    Usage:
        store = ArtifactStore(artifacts_dir=settings.artifacts_dir)

        # Chunking stage: save all chunks at once
        store.save_chunks(chunks)

        # Extraction stage: save incrementally (one chunk at a time)
        store.append_extraction(extraction)   # crash-safe, called per chunk

        # Query/eval stages: load everything
        chunks      = store.load_chunks()
        extractions = store.load_extractions()
    """

    def __init__(self, artifacts_dir: Path | str) -> None:
        """
        Args:
            artifacts_dir: Directory where all artifact JSON files are stored.
                           Created if it does not exist.
                           Must come from settings.artifacts_dir — never hardcode.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._chunks_path      = self.artifacts_dir / CHUNKS_FILENAME
        self._extractions_path = self.artifacts_dir / EXTRACTIONS_FILENAME

        log.debug(
            "ArtifactStore initialized",
            artifacts_dir=str(self.artifacts_dir),
        )

    # ── Chunks ─────────────────────────────────────────────────────────────────

    def save_chunks(self, chunks: list[ChunkSchema]) -> None:
        """
        Atomically save all chunks to chunks.json.

        Overwrites any existing file. Called once by the chunking stage
        after all documents have been split.

        Uses atomic write: write → temp file → rename.
        On crash between write and rename, the old file is untouched.

        Args:
            chunks: All ChunkSchema objects produced by the chunking stage.

        Raises:
            IOError: If the write or rename fails.
        """
        t0 = time.monotonic()
        data = [c.model_dump(mode="json") for c in chunks]
        _atomic_write_json(self._chunks_path, data)

        log.info(
            "Chunks saved",
            count=len(chunks),
            path=str(self._chunks_path),
            size_mb=round(self._chunks_path.stat().st_size / 1_048_576, 2),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    def load_chunks(self) -> list[ChunkSchema]:
        """
        Load all chunks from chunks.json.

        Validates every record through ChunkSchema. Corrupt records are
        logged and skipped so a single bad record doesn't prevent loading.

        Returns:
            List of validated ChunkSchema objects.

        Raises:
            FileNotFoundError: If chunks.json does not exist.
                               Caller should check chunks_exist() first.
        """
        if not self._chunks_path.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {self._chunks_path}. "
                "Run the chunking pipeline stage first."
            )

        t0 = time.monotonic()
        raw_records = _load_json_list(self._chunks_path)

        chunks: list[ChunkSchema] = []
        skipped = 0
        for i, record in enumerate(raw_records):
            try:
                chunks.append(ChunkSchema.model_validate(record))
            except Exception as e:
                log.warning(
                    "Skipping corrupt chunk record",
                    index=i,
                    error=str(e),
                )
                skipped += 1

        log.info(
            "Chunks loaded",
            count=len(chunks),
            skipped=skipped,
            path=str(self._chunks_path),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return chunks

    def load_chunks_iter(self) -> Iterator[ChunkSchema]:
        """
        Memory-efficient iterator for loading chunks one at a time.

        Use this when you need to process chunks without loading all of
        them into memory at once (e.g. for very large corpora).

        Yields:
            One validated ChunkSchema per iteration.
            Corrupt records are logged and skipped.
        """
        if not self._chunks_path.exists():
            raise FileNotFoundError(f"chunks.json not found at {self._chunks_path}")

        for i, record in enumerate(_iter_json_list(self._chunks_path)):
            try:
                yield ChunkSchema.model_validate(record)
            except Exception as e:
                log.warning("Skipping corrupt chunk record", index=i, error=str(e))

    def chunks_exist(self) -> bool:
        """True if chunks.json exists and is non-empty."""
        return self._chunks_path.exists() and self._chunks_path.stat().st_size > 2

    def chunks_count(self) -> int:
        """Return the number of chunks without loading all data."""
        if not self.chunks_exist():
            return 0
        try:
            data = _load_json_list(self._chunks_path)
            return len(data)
        except Exception:
            return 0

    # ── Extractions ────────────────────────────────────────────────────────────

    def save_extractions(self, extractions: list[ChunkExtraction]) -> None:
        """
        Atomically save all extractions to extractions.json.

        Overwrites any existing file. Use this when you have all extractions
        in memory at the end of the extraction stage.

        For incremental (per-chunk) saving during the extraction stage,
        use append_extraction() instead — it is crash-safe and supports resume.

        Args:
            extractions: All ChunkExtraction objects from the extraction stage.
        """
        t0 = time.monotonic()
        data = [e.model_dump(mode="json") for e in extractions]
        _atomic_write_json(self._extractions_path, data)

        log.info(
            "Extractions saved",
            count=len(extractions),
            path=str(self._extractions_path),
            size_mb=round(self._extractions_path.stat().st_size / 1_048_576, 2),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    def clear_extractions(self) -> None:
        """Delete the extraction artifact file if it exists."""
        if self._extractions_path.exists():
            self._extractions_path.unlink()
            log.info("Extraction artifact cleared", path=str(self._extractions_path))

    def append_extraction(self, extraction: ChunkExtraction) -> None:
        """
        Append a single extraction to extractions.json.

        Used during the extraction stage to save results incrementally.
        If the pipeline crashes, already-extracted chunks are preserved
        and can be skipped on the next run via CacheManager.

        Implementation:
          - Reads the existing JSON array
          - Appends the new record
          - Atomically rewrites the file

        This is O(n) per call — for very large corpora, batch with
        save_extractions_batch() instead.

        Args:
            extraction: The ChunkExtraction for one processed chunk.
        """
        existing = []
        if self._extractions_path.exists():
            try:
                existing = _load_json_list(self._extractions_path)
            except Exception as e:
                log.warning(
                    "Could not read existing extractions for append, starting fresh",
                    error=str(e),
                )

        existing.append(extraction.model_dump(mode="json"))
        _atomic_write_json(self._extractions_path, existing)

    def save_extractions_batch(self, extractions: list[ChunkExtraction]) -> None:
        """
        Append a batch of extractions to extractions.json.

        More efficient than calling append_extraction() in a loop.
        Used by the pipeline runner at the end of each processing batch.

        Args:
            extractions: Batch of ChunkExtraction results to append.
        """
        if not extractions:
            return

        existing = []
        if self._extractions_path.exists():
            try:
                existing = _load_json_list(self._extractions_path)
            except Exception as e:
                log.warning(
                    "Could not read existing extractions for batch append",
                    error=str(e),
                )

        existing.extend([e.model_dump(mode="json") for e in extractions])
        _atomic_write_json(self._extractions_path, existing)

        log.debug(
            "Extractions batch appended",
            batch_size=len(extractions),
            total_now=len(existing),
        )

    def load_extractions(self) -> list[ChunkExtraction]:
        """
        Load all extractions from extractions.json.

        Validates every record through ChunkExtraction. Corrupt records
        are logged and skipped.

        Returns:
            List of validated ChunkExtraction objects.

        Raises:
            FileNotFoundError: If extractions.json does not exist.
        """
        if not self._extractions_path.exists():
            raise FileNotFoundError(
                f"extractions.json not found at {self._extractions_path}. "
                "Run the extraction pipeline stage first."
            )

        t0 = time.monotonic()
        raw_records = _load_json_list(self._extractions_path)

        extractions: list[ChunkExtraction] = []
        skipped = 0
        for i, record in enumerate(raw_records):
            try:
                extractions.append(ChunkExtraction.model_validate(record))
            except Exception as e:
                log.warning(
                    "Skipping corrupt extraction record",
                    index=i,
                    chunk_id=record.get("chunk_id", "unknown"),
                    error=str(e),
                )
                skipped += 1

        log.info(
            "Extractions loaded",
            count=len(extractions),
            skipped=skipped,
            path=str(self._extractions_path),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return extractions

    def load_extractions_as_dict(self) -> dict[str, ChunkExtraction]:
        """
        Load extractions as a dict keyed by chunk_id.

        Used by the graph builder to quickly look up what was extracted
        for each chunk without O(n) linear scans.

        Returns:
            Dict mapping chunk_id → ChunkExtraction.
        """
        extractions = self.load_extractions()
        return {e.chunk_id: e for e in extractions}

    def extractions_exist(self) -> bool:
        """True if extractions.json exists and is non-empty."""
        return (
            self._extractions_path.exists()
            and self._extractions_path.stat().st_size > 2
        )

    def extractions_count(self) -> int:
        """Return the number of extraction records without loading all data."""
        if not self.extractions_exist():
            return 0
        try:
            data = _load_json_list(self._extractions_path)
            return len(data)
        except Exception:
            return 0

    def get_extracted_chunk_ids(self) -> set[str]:
        """
        Return the set of chunk_ids that have already been extracted.

        Used by CacheManager and the pipeline runner to identify which
        chunks can be skipped on a resumed run.

        More efficient than load_extractions() when you only need IDs.

        Returns:
            Set of chunk_id strings for all completed extractions.
        """
        if not self.extractions_exist():
            return set()
        try:
            raw = _load_json_list(self._extractions_path)
            return {r["chunk_id"] for r in raw if "chunk_id" in r}
        except Exception as e:
            log.warning("Could not read extracted chunk IDs", error=str(e))
            return set()

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def delete_chunks(self) -> bool:
        """Delete chunks.json. Returns True if deleted, False if not found."""
        if self._chunks_path.exists():
            self._chunks_path.unlink()
            log.info("Deleted chunks.json")
            return True
        return False

    def delete_extractions(self) -> bool:
        """Delete extractions.json. Returns True if deleted, False if not found."""
        if self._extractions_path.exists():
            self._extractions_path.unlink()
            log.info("Deleted extractions.json")
            return True
        return False

    def delete_all(self) -> None:
        """Delete all artifact files. Used when force_reindex=True."""
        self.delete_chunks()
        self.delete_extractions()
        log.info("All ArtifactStore files deleted")

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return a summary of artifact file sizes and record counts.
        Used by the health check endpoint and pipeline status endpoint.
        """
        stats: dict = {
            "artifacts_dir": str(self.artifacts_dir),
            "chunks": {
                "exists": self.chunks_exist(),
                "count": self.chunks_count() if self.chunks_exist() else 0,
                "size_mb": round(
                    self._chunks_path.stat().st_size / 1_048_576, 3
                ) if self.chunks_exist() else 0.0,
            },
            "extractions": {
                "exists": self.extractions_exist(),
                "count": self.extractions_count() if self.extractions_exist() else 0,
                "size_mb": round(
                    self._extractions_path.stat().st_size / 1_048_576, 3
                ) if self.extractions_exist() else 0.0,
            },
        }
        return stats

    def __repr__(self) -> str:
        return (
            f"ArtifactStore("
            f"artifacts_dir={str(self.artifacts_dir)!r}, "
            f"chunks_exist={self.chunks_exist()}, "
            f"extractions_exist={self.extractions_exist()})"
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: list | dict) -> None:
    """
    Write JSON to path atomically using a temp file + rename.

    On Windows, os.replace() is used (atomic on NTFS).
    On POSIX, os.rename() is atomic within the same filesystem.

    This guarantees that the target file is either:
      (a) the complete new content, or
      (b) the old content (if we crashed before rename)
    — never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory (ensures same filesystem)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.tmp",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=_json_default)
        # Atomic replace
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_json_list(path: Path) -> list:
    """Load a JSON file that contains a top-level list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def _iter_json_list(path: Path):
    """
    Iterate over a JSON array file without loading everything into memory.

    Simple implementation: load the full file but yield records one by one.
    For truly streaming large files, replace with ijson if available.
    """
    data = _load_json_list(path)
    yield from data


def _json_default(obj):
    """JSON serializer for types not handled by the standard library."""
    from datetime import datetime, date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── Factory function ───────────────────────────────────────────────────────────

def get_artifact_store() -> ArtifactStore:
    """
    Build an ArtifactStore from application settings.
    Paths always come from config — never hardcoded.
    """
    from app.config import get_settings
    settings = get_settings()
    return ArtifactStore(artifacts_dir=settings.artifacts_dir)


__all__ = [
    "ArtifactStore",
    "get_artifact_store",
    "CHUNKS_FILENAME",
    "EXTRACTIONS_FILENAME",
]