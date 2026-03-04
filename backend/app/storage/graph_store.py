"""
storage/graph_store.py — NetworkX graph and community map persistence.

Handles all read/write operations for graph artifacts:
  graph.pkl            → NetworkX DiGraph (nodes=entities, edges=relationships)
  community_map.json   → list[CommunitySchema] (Leiden community assignments)

Why pickle for the graph?
  NetworkX graphs contain Python objects (dicts, lists, custom attributes)
  that don't serialize cleanly to JSON without lossy conversion. Pickle
  preserves the full graph structure with all node/edge attributes intact.
  The graph is only written once (after graph construction) and read a few
  times (community detection, summarization, graph API endpoint) — so the
  pickle overhead is acceptable.

  Security note: only load pickle files you produced yourself. The graph
  is an internal artifact, never user-uploaded, so this is safe.

Why JSON for the community map?
  The community map is a list of CommunitySchema objects — pure data with
  no Python-specific types. JSON is human-readable and can be inspected
  for debugging, which is valuable during development.

Atomic writes are used for both files (same .tmp → rename pattern as
artifact_store.py).

Graph node/edge attributes stored:
  Nodes: name, entity_type, description, degree, source_chunk_ids,
         claims, mention_count, community_ids
  Edges: description, weight, source_chunk_ids, combined_degree
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

from app.models.graph_models import CommunitySchema
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── File name constants ────────────────────────────────────────────────────────
GRAPH_FILENAME         = "graph.pkl"
COMMUNITY_MAP_FILENAME = "community_map.json"


class GraphStore:
    """
    Persistence layer for the NetworkX knowledge graph and community map.

    Usage:
        store = GraphStore(artifacts_dir=settings.artifacts_dir)

        # Graph construction stage: save the built graph
        store.save_graph(nx_graph)

        # Community detection stage: load graph, save community map
        graph = store.load_graph()
        store.save_community_map(communities)

        # Query/summarization stages: load what you need
        graph       = store.load_graph()
        communities = store.load_community_map()
    """

    def __init__(self, artifacts_dir: Path | str) -> None:
        """
        Args:
            artifacts_dir: Directory for graph artifact files.
                           Created if it does not exist.
                           Always use settings.artifacts_dir — never hardcode.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._graph_path         = self.artifacts_dir / GRAPH_FILENAME
        self._community_map_path = self.artifacts_dir / COMMUNITY_MAP_FILENAME

        log.debug(
            "GraphStore initialized",
            artifacts_dir=str(self.artifacts_dir),
        )

    # ── NetworkX Graph ─────────────────────────────────────────────────────────

    def save_graph(self, graph: Any) -> None:
        """
        Atomically save a NetworkX graph to graph.pkl.

        Preserves all node attributes (entity descriptions, community_ids,
        mention counts, etc.) and edge attributes (weight, combined_degree,
        source_chunk_ids).

        Args:
            graph: A NetworkX Graph or DiGraph object with full node/edge attributes.
                   Must be picklable (standard NetworkX graphs always are).

        Raises:
            IOError: If the write fails.
        """
        t0 = time.monotonic()

        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._graph_path.parent),
            prefix=f".{GRAPH_FILENAME}.tmp",
            suffix=".pkl",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, str(self._graph_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        size_mb = self._graph_path.stat().st_size / 1_048_576
        log.info(
            "Graph saved",
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            path=str(self._graph_path),
            size_mb=round(size_mb, 2),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    def load_graph(self) -> Any:
        """
        Load the NetworkX graph from graph.pkl.

        Returns:
            The NetworkX Graph or DiGraph object with all node/edge attributes.

        Raises:
            FileNotFoundError: If graph.pkl does not exist.
            pickle.UnpicklingError: If the file is corrupt.
        """
        if not self._graph_path.exists():
            raise FileNotFoundError(
                f"graph.pkl not found at {self._graph_path}. "
                "Run the graph construction pipeline stage first."
            )

        t0 = time.monotonic()
        with open(self._graph_path, "rb") as f:
            graph = pickle.load(f)

        log.info(
            "Graph loaded",
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            path=str(self._graph_path),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return graph

    def graph_exists(self) -> bool:
        """True if graph.pkl exists and is non-empty."""
        return self._graph_path.exists() and self._graph_path.stat().st_size > 0

    def get_graph_stats(self) -> dict:
        """
        Return node/edge counts without fully loading the graph.

        Loads the graph but discards everything except the counts.
        Use for the status endpoint.
        """
        if not self.graph_exists():
            return {"nodes": 0, "edges": 0, "exists": False}
        try:
            g = self.load_graph()
            return {
                "exists": True,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "size_mb": round(self._graph_path.stat().st_size / 1_048_576, 3),
            }
        except Exception as e:
            log.warning("Could not get graph stats", error=str(e))
            return {"exists": True, "nodes": 0, "edges": 0, "error": str(e)}

    # ── Community Map ──────────────────────────────────────────────────────────

    def save_community_map(self, communities: list[CommunitySchema]) -> None:
        """
        Atomically save the community map to community_map.json.

        Serializes each CommunitySchema to JSON using Pydantic's model_dump.
        The file is a flat JSON array — all communities from all levels.

        Args:
            communities: All CommunitySchema objects from all hierarchy levels
                         (C0 through C3), as produced by community detection.
        """
        t0 = time.monotonic()

        data = [c.model_dump(mode="json") for c in communities]

        self._community_map_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._community_map_path.parent),
            prefix=f".{COMMUNITY_MAP_FILENAME}.tmp",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=_json_default)
            os.replace(tmp_path, str(self._community_map_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Count by level for logging
        by_level: dict[str, int] = {}
        for c in communities:
            lv = str(c.level.value) if hasattr(c.level, "value") else str(c.level)
            by_level[lv] = by_level.get(lv, 0) + 1

        log.info(
            "Community map saved",
            total=len(communities),
            by_level=by_level,
            path=str(self._community_map_path),
            size_mb=round(self._community_map_path.stat().st_size / 1_048_576, 2),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    def load_community_map(self) -> list[CommunitySchema]:
        """
        Load all communities from community_map.json.

        Validates every record through CommunitySchema. Corrupt records
        are logged and skipped.

        Returns:
            List of CommunitySchema objects for all levels.

        Raises:
            FileNotFoundError: If community_map.json does not exist.
        """
        if not self._community_map_path.exists():
            raise FileNotFoundError(
                f"community_map.json not found at {self._community_map_path}. "
                "Run the community detection pipeline stage first."
            )

        t0 = time.monotonic()
        with open(self._community_map_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        if not isinstance(raw_records, list):
            raise ValueError(
                f"community_map.json must contain a JSON array, "
                f"got {type(raw_records).__name__}"
            )

        communities: list[CommunitySchema] = []
        skipped = 0
        for i, record in enumerate(raw_records):
            try:
                communities.append(CommunitySchema.model_validate(record))
            except Exception as e:
                log.warning(
                    "Skipping corrupt community record",
                    index=i,
                    community_id=record.get("community_id", "unknown"),
                    error=str(e),
                )
                skipped += 1

        log.info(
            "Community map loaded",
            count=len(communities),
            skipped=skipped,
            path=str(self._community_map_path),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return communities

    def load_community_map_by_level(self) -> dict[str, list[CommunitySchema]]:
        """
        Load the community map grouped by hierarchy level.

        Returns:
            Dict mapping level string ("c0", "c1", "c2", "c3")
            → list of CommunitySchema at that level.

        Example:
            by_level = store.load_community_map_by_level()
            c1_communities = by_level["c1"]   # ~555 communities
        """
        communities = self.load_community_map()
        by_level: dict[str, list[CommunitySchema]] = {}
        for comm in communities:
            lv = comm.level.value if hasattr(comm.level, "value") else str(comm.level)
            by_level.setdefault(lv, []).append(comm)
        return by_level

    def community_map_exists(self) -> bool:
        """True if community_map.json exists and is non-empty."""
        return (
            self._community_map_path.exists()
            and self._community_map_path.stat().st_size > 2
        )

    def get_community_counts(self) -> dict[str, int]:
        """
        Return community counts per level without loading full records.

        Returns:
            Dict like {"c0": 55, "c1": 555, "c2": 1797, "c3": 2142}.
            Empty dict if community_map.json does not exist.
        """
        if not self.community_map_exists():
            return {}
        try:
            with open(self._community_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            counts: dict[str, int] = {}
            for r in raw:
                lv = r.get("level", "unknown")
                # Handle both enum value strings and raw strings
                if isinstance(lv, dict):
                    lv = lv.get("value", "unknown")
                counts[str(lv)] = counts.get(str(lv), 0) + 1
            return counts
        except Exception as e:
            log.warning("Could not get community counts", error=str(e))
            return {}

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def delete_graph(self) -> bool:
        """Delete graph.pkl. Returns True if deleted, False if not found."""
        if self._graph_path.exists():
            self._graph_path.unlink()
            log.info("Deleted graph.pkl")
            return True
        return False

    def delete_community_map(self) -> bool:
        """Delete community_map.json. Returns True if deleted, False if not found."""
        if self._community_map_path.exists():
            self._community_map_path.unlink()
            log.info("Deleted community_map.json")
            return True
        return False

    def delete_all(self) -> None:
        """Delete all graph store files. Used when force_reindex=True."""
        self.delete_graph()
        self.delete_community_map()
        log.info("All GraphStore files deleted")

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a summary of graph artifact sizes and record counts."""
        stats: dict = {
            "artifacts_dir": str(self.artifacts_dir),
            "graph": self.get_graph_stats(),
            "community_map": {
                "exists": self.community_map_exists(),
                "counts_by_level": self.get_community_counts() if self.community_map_exists() else {},
                "size_mb": round(
                    self._community_map_path.stat().st_size / 1_048_576, 3
                ) if self.community_map_exists() else 0.0,
            },
        }
        return stats

    def __repr__(self) -> str:
        return (
            f"GraphStore("
            f"artifacts_dir={str(self.artifacts_dir)!r}, "
            f"graph_exists={self.graph_exists()}, "
            f"community_map_exists={self.community_map_exists()})"
        )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """JSON serializer for datetime and other non-standard types."""
    from datetime import datetime, date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── Factory function ───────────────────────────────────────────────────────────

def get_graph_store() -> GraphStore:
    """Build a GraphStore from application settings."""
    from app.config import get_settings
    settings = get_settings()
    return GraphStore(artifacts_dir=settings.artifacts_dir)


__all__ = [
    "GraphStore",
    "get_graph_store",
    "GRAPH_FILENAME",
    "COMMUNITY_MAP_FILENAME",
]