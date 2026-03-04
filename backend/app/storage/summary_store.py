"""
storage/summary_store.py — Community summary persistence.

Handles read/write for the final and most query-critical artifact:
  community_summaries.json → list[CommunitySummary]

This file is the heart of the GraphRAG query engine. Every query reads it
to get the community summaries that feed the map-reduce pipeline.

The summarization stage writes this file once at the end of indexing.
The query engine reads it at every query (with an optional in-memory cache
so repeated queries don't re-read from disk).

Querying patterns:
  - GraphRAG map stage: load all summaries at a given level (c0/c1/c2/c3)
  - Community API:      load summaries paginated by level
  - Status endpoint:    count summaries without loading full text

Index for fast lookup:
  An in-memory index by community_id and by level is built on first load.
  This avoids O(n) scans when looking up a single community.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from app.models.graph_models import CommunitySummary
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── File name constants ────────────────────────────────────────────────────────
SUMMARIES_FILENAME = "community_summaries.json"


class SummaryStore:
    """
    Persistence layer for community summaries.

    Supports both full-file loading and level-filtered loading.
    An in-memory cache is maintained after the first load to avoid
    repeated disk reads during a query session.

    Usage:
        store = SummaryStore(artifacts_dir=settings.artifacts_dir)

        # Summarization stage: save all summaries once
        store.save_summaries(all_summaries)

        # Query engine: load summaries for the selected community level
        c1_summaries = store.load_summaries_by_level("c1")

        # Status endpoint: count without loading full content
        counts = store.get_summary_counts()
    """

    def __init__(
        self,
        artifacts_dir: Path | str,
        use_cache: bool = True,
    ) -> None:
        """
        Args:
            artifacts_dir: Directory containing community_summaries.json.
                           Always use settings.artifacts_dir — never hardcode.
            use_cache:     If True, cache loaded summaries in memory after
                           the first load. Subsequent load_summaries() calls
                           return the cached copy without reading disk.
                           Set False in tests or when summaries change.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        self._summaries_path = self.artifacts_dir / SUMMARIES_FILENAME

        # In-memory cache: populated on first load
        self._cache: list[CommunitySummary] | None = None
        self._cache_by_id: dict[str, CommunitySummary] = {}
        self._cache_by_level: dict[str, list[CommunitySummary]] = {}

        log.debug(
            "SummaryStore initialized",
            artifacts_dir=str(self.artifacts_dir),
            use_cache=use_cache,
        )

    # ── Save ───────────────────────────────────────────────────────────────────

    def save_summaries(self, summaries: list[CommunitySummary]) -> None:
        """
        Atomically save all community summaries to community_summaries.json.

        Called once by the summarization pipeline stage after all community
        summaries have been generated.

        Args:
            summaries: All CommunitySummary objects from all hierarchy levels.
                       Typically ~4,500 summaries for a 1M-token corpus.
        """
        t0 = time.monotonic()
        data = [s.model_dump(mode="json") for s in summaries]

        self._summaries_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._summaries_path.parent),
            prefix=f".{SUMMARIES_FILENAME}.tmp",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=_json_default)
            os.replace(tmp_path, str(self._summaries_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Invalidate cache after save
        self._invalidate_cache()

        # Count by level for logging
        by_level: dict[str, int] = {}
        for s in summaries:
            lv = s.level.value if hasattr(s.level, "value") else str(s.level)
            by_level[lv] = by_level.get(lv, 0) + 1

        log.info(
            "Community summaries saved",
            total=len(summaries),
            by_level=by_level,
            path=str(self._summaries_path),
            size_mb=round(self._summaries_path.stat().st_size / 1_048_576, 2),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    def append_summaries(self, summaries: list[CommunitySummary]) -> None:
        """
        Append new summaries to the existing file.

        Used when summarization is done level-by-level and we want to
        persist each level's results as they complete (crash safety).

        Args:
            summaries: New CommunitySummary objects to append.
        """
        if not summaries:
            return

        existing: list = []
        if self._summaries_path.exists():
            try:
                with open(self._summaries_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception as e:
                log.warning(
                    "Could not read existing summaries for append",
                    error=str(e),
                )

        existing.extend([s.model_dump(mode="json") for s in summaries])

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._summaries_path.parent),
            prefix=f".{SUMMARIES_FILENAME}.tmp",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, default=_json_default)
            os.replace(tmp_path, str(self._summaries_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._invalidate_cache()

        log.debug(
            "Summaries appended",
            appended=len(summaries),
            total_now=len(existing),
        )

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_summaries(self) -> list[CommunitySummary]:
        """
        Load all community summaries from disk (or cache).

        On first call: reads community_summaries.json, validates every record,
        populates the in-memory cache (id → summary, level → summaries).

        On subsequent calls: returns the cached list immediately.

        Returns:
            List of all CommunitySummary objects across all levels.

        Raises:
            FileNotFoundError: If community_summaries.json does not exist.
        """
        if self.use_cache and self._cache is not None:
            return self._cache

        if not self._summaries_path.exists():
            raise FileNotFoundError(
                f"community_summaries.json not found at {self._summaries_path}. "
                "Run the summarization pipeline stage first."
            )

        t0 = time.monotonic()
        with open(self._summaries_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        if not isinstance(raw_records, list):
            raise ValueError(
                f"community_summaries.json must contain a JSON array, "
                f"got {type(raw_records).__name__}"
            )

        summaries: list[CommunitySummary] = []
        skipped = 0
        for i, record in enumerate(raw_records):
            try:
                summaries.append(CommunitySummary.model_validate(record))
            except Exception as e:
                log.warning(
                    "Skipping corrupt summary record",
                    index=i,
                    community_id=record.get("community_id", "unknown"),
                    error=str(e),
                )
                skipped += 1

        log.info(
            "Community summaries loaded",
            count=len(summaries),
            skipped=skipped,
            path=str(self._summaries_path),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

        if self.use_cache:
            self._populate_cache(summaries)

        return summaries

    def load_summaries_by_level(self, level: str) -> list[CommunitySummary]:
        """
        Load all summaries at a specific hierarchy level.

        This is the primary method called by the GraphRAG map stage:
          summaries = store.load_summaries_by_level("c1")
          # → ~555 CommunitySummary objects at C1 level

        Args:
            level: Community level string: "c0", "c1", "c2", or "c3".

        Returns:
            List of CommunitySummary objects at the specified level.
            Empty list if no summaries exist at that level.
        """
        if self.use_cache and self._cache is not None:
            return self._cache_by_level.get(level, [])

        all_summaries = self.load_summaries()
        return self._cache_by_level.get(level, [])

    def load_summary_by_id(self, community_id: str) -> CommunitySummary | None:
        """
        Look up a single summary by community_id.

        O(1) after first load (uses the in-memory id index).

        Args:
            community_id: e.g. "comm_c1_0045"

        Returns:
            CommunitySummary if found, None otherwise.
        """
        if self.use_cache and self._cache is not None:
            return self._cache_by_id.get(community_id)

        self.load_summaries()
        return self._cache_by_id.get(community_id)

    def load_summaries_paginated(
        self,
        level: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[CommunitySummary], int]:
        """
        Load summaries for a level with pagination.

        Used by the GET /api/v1/communities/{level} endpoint.

        Args:
            level:     Community level to filter.
            page:      1-based page number.
            page_size: Items per page.

        Returns:
            Tuple of (page_summaries, total_count).
        """
        level_summaries = self.load_summaries_by_level(level)
        total = len(level_summaries)
        start = (page - 1) * page_size
        end   = start + page_size
        return level_summaries[start:end], total

    # ── Queries ────────────────────────────────────────────────────────────────

    def summaries_exist(self) -> bool:
        """True if community_summaries.json exists and is non-empty."""
        return (
            self._summaries_path.exists()
            and self._summaries_path.stat().st_size > 2
        )

    def get_summary_counts(self) -> dict[str, int]:
        """
        Return summary counts per level without loading full text.

        Returns:
            Dict like {"c0": 55, "c1": 555, "c2": 1797, "c3": 2142}.
            Empty dict if file does not exist.
        """
        if self.use_cache and self._cache is not None:
            return {lv: len(summaries) for lv, summaries in self._cache_by_level.items()}

        if not self.summaries_exist():
            return {}
        try:
            with open(self._summaries_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            counts: dict[str, int] = {}
            for r in raw:
                lv = r.get("level", "unknown")
                if isinstance(lv, dict):
                    lv = lv.get("value", "unknown")
                counts[str(lv)] = counts.get(str(lv), 0) + 1
            return counts
        except Exception as e:
            log.warning("Could not get summary counts", error=str(e))
            return {}

    def total_summaries(self) -> int:
        """Return total number of summaries across all levels."""
        return sum(self.get_summary_counts().values())

    # ── Cache management ───────────────────────────────────────────────────────

    def warm_cache(self) -> None:
        """
        Pre-load summaries into the in-memory cache.

        Call this at application startup so the first query doesn't
        incur the disk read latency.
        """
        if not self.summaries_exist():
            log.warning("Cannot warm cache: community_summaries.json does not exist")
            return
        self.load_summaries()
        log.info(
            "Summary cache warmed",
            total=len(self._cache) if self._cache else 0,
            levels=list(self._cache_by_level.keys()),
        )

    def invalidate_cache(self) -> None:
        """
        Clear the in-memory cache.

        Call after save_summaries() or if summaries are updated externally.
        """
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache = None
        self._cache_by_id = {}
        self._cache_by_level = {}

    def _populate_cache(self, summaries: list[CommunitySummary]) -> None:
        """Build the id → summary and level → summaries indexes."""
        self._cache = summaries
        self._cache_by_id = {s.community_id: s for s in summaries}
        self._cache_by_level = {}
        for s in summaries:
            lv = s.level.value if hasattr(s.level, "value") else str(s.level)
            self._cache_by_level.setdefault(lv, []).append(s)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def delete_summaries(self) -> bool:
        """Delete community_summaries.json. Returns True if deleted."""
        if self._summaries_path.exists():
            self._summaries_path.unlink()
            self._invalidate_cache()
            log.info("Deleted community_summaries.json")
            return True
        return False

    def delete_all(self) -> None:
        """Alias for delete_summaries(). Used by force_reindex."""
        self.delete_summaries()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a summary of the summaries file."""
        return {
            "artifacts_dir": str(self.artifacts_dir),
            "summaries": {
                "exists": self.summaries_exist(),
                "counts_by_level": self.get_summary_counts(),
                "total": self.total_summaries(),
                "size_mb": round(
                    self._summaries_path.stat().st_size / 1_048_576, 3
                ) if self.summaries_exist() else 0.0,
                "cache_loaded": self._cache is not None,
            },
        }

    def __repr__(self) -> str:
        return (
            f"SummaryStore("
            f"artifacts_dir={str(self.artifacts_dir)!r}, "
            f"summaries_exist={self.summaries_exist()}, "
            f"cache_loaded={self._cache is not None})"
        )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _json_default(obj):
    from datetime import datetime, date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── Factory function ───────────────────────────────────────────────────────────

def get_summary_store() -> SummaryStore:
    """Build a SummaryStore from application settings."""
    from app.config import get_settings
    settings = get_settings()
    return SummaryStore(artifacts_dir=settings.artifacts_dir)


__all__ = [
    "SummaryStore",
    "get_summary_store",
    "SUMMARIES_FILENAME",
]