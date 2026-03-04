"""
storage/__init__.py — Public re-exports for all storage classes.

All pipeline file I/O goes through these classes.
Never import json/pickle/pathlib directly in pipeline or query code.
"""

from app.storage.artifact_store import (
    ArtifactStore,
    get_artifact_store,
    CHUNKS_FILENAME,
    EXTRACTIONS_FILENAME,
)
from app.storage.graph_store import (
    GraphStore,
    get_graph_store,
    GRAPH_FILENAME,
    COMMUNITY_MAP_FILENAME,
)
from app.storage.summary_store import (
    SummaryStore,
    get_summary_store,
    SUMMARIES_FILENAME,
)
from app.storage.cache_manager import (
    CacheManager,
    get_cache_manager,
    ALL_STAGES,
    STATE_FILENAME,
)

__all__ = [
    # ArtifactStore
    "ArtifactStore",
    "get_artifact_store",
    "CHUNKS_FILENAME",
    "EXTRACTIONS_FILENAME",
    # GraphStore
    "GraphStore",
    "get_graph_store",
    "GRAPH_FILENAME",
    "COMMUNITY_MAP_FILENAME",
    # SummaryStore
    "SummaryStore",
    "get_summary_store",
    "SUMMARIES_FILENAME",
    # CacheManager
    "CacheManager",
    "get_cache_manager",
    "ALL_STAGES",
    "STATE_FILENAME",
]