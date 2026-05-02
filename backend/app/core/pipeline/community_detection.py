"""
core/pipeline/community_detection.py — Leiden hierarchical community detection.

Implements paper Section 3.1.3:
  "We use the Leiden algorithm (Traag et al., 2019) to detect hierarchical
   communities in the knowledge graph, yielding a hierarchy of communities
   at multiple levels of resolution (C0–C3)."

Library: graspologic
  The paper uses graspologic's hierarchical_leiden() which wraps the Leiden
  algorithm and produces a HierarchicalCluster object.

  Install: pip install graspologic

Fallback (when graspologic is not available):
  Uses NetworkX's Louvain community detection as a single-level fallback.
  This produces a single level (C0) rather than a full C0–C3 hierarchy.
  All tests use this fallback to avoid requiring graspologic.

Community ID format:
  "comm_{level}_{index:04d}"  e.g. "comm_c1_0045"

Parent-child relationships:
  C0 communities contain C1 communities, which contain C2, which contain C3.
  Each CommunitySchema records its parent_community_id and child_community_ids.
  This is used by the summarization stage for the sub-community substitution
  strategy (higher-level summaries use sub-community summaries as context
  when the raw element content is too large for the 8k window).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from app.models.graph_models import CommunityLevel, CommunitySchema
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Optional dependency flags ──────────────────────────────────────────────────
try:
    from graspologic.partition import hierarchical_leiden
    _HAS_GRASPOLOGIC = True
except ImportError:
    _HAS_GRASPOLOGIC = False
    log.warning(
        "graspologic not installed — using single-level Louvain fallback. "
        "Install graspologic for full hierarchical Leiden support."
    )

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

# ── Level name constants ───────────────────────────────────────────────────────
_LEVEL_NAMES = ["c0", "c1", "c2", "c3"]


class CommunityDetection:
    """
    Hierarchical community detection for the knowledge graph.

    Uses graspologic's hierarchical_leiden() if available, falling back
    to NetworkX Louvain for single-level detection.

    Usage:
        detector = CommunityDetection()
        communities = detector.detect(graph, max_levels=3)
        # → list[CommunitySchema] for all levels
    """

    def __init__(
        self,
        max_cluster_size: int = 10,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            max_cluster_size: Maximum nodes per community in Leiden.
                              Smaller values produce more granular communities.
                              graspologic default: 10.
            random_seed:      Random seed for reproducibility.
        """
        self.max_cluster_size = max_cluster_size
        self.random_seed      = random_seed

    def detect(
        self,
        graph: Any,    # networkx.Graph
        max_levels: int = 3,
    ) -> list[CommunitySchema]:
        """
        Run community detection on the graph and return CommunitySchema list.

        Args:
            graph:      NetworkX Graph with node/edge attributes.
            max_levels: Maximum hierarchy depth. Default: 3 (C0–C3).

        Returns:
            Flat list of CommunitySchema for all levels (C0 through C{max_levels}).
            Ordered by level then by community index within each level.

        Raises:
            ImportError: If neither graspologic nor networkx is available.
        """
        if not _HAS_NETWORKX:
            raise ImportError("networkx is required for community detection")

        if graph.number_of_nodes() == 0:
            log.warning("Empty graph — no communities to detect")
            return []

        # Sparse graph guard: if no edges, Leiden/Louvain will crash.
        # Treat each connected component as its own C1 community instead.
        if graph.number_of_edges() == 0:
            log.warning(
                "Graph has no edges — treating each node as its own community",
                nodes=graph.number_of_nodes(),
            )
            return self._single_node_communities(graph)

        t0 = time.monotonic()
        log.info(
            "Community detection started",
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            max_levels=max_levels,
            algorithm="leiden" if _HAS_GRASPOLOGIC else "louvain_fallback",
        )

        if _HAS_GRASPOLOGIC:
            communities = self._detect_leiden(graph, max_levels)
        else:
            communities = self._detect_louvain_fallback(graph)

        # Add parent-child links
        _build_parent_child_links(communities)

        # Update graph nodes with community_ids
        _annotate_graph_nodes(graph, communities)

        elapsed = time.monotonic() - t0
        counts_by_level = _count_by_level(communities)

        log.info(
            "Community detection complete",
            total_communities=len(communities),
            counts_by_level=counts_by_level,
            elapsed_seconds=round(elapsed, 2),
        )

        return communities

    def _detect_leiden(
        self,
        graph: Any,
        max_levels: int,
    ) -> list[CommunitySchema]:
        """Run hierarchical Leiden via graspologic."""        # Convert graph to edge list for graspologic
        # graspologic expects integer node IDs
        node_list = list(graph.nodes())
        node_to_int = {node: i for i, node in enumerate(node_list)}

        edge_list = [
            (node_to_int[u], node_to_int[v], graph.edges[u, v].get("weight", 1.0))
            for u, v in graph.edges()
        ]

        # Leiden requires at least one edge — fall back if graph is too sparse
        if not edge_list:
            log.warning("No edges for Leiden — falling back to single-node communities")
            return self._single_node_communities(graph)

        try:
            # Run hierarchical Leiden
            hierarchy = hierarchical_leiden(
                edge_list,
                max_cluster_size=self.max_cluster_size,
                random_seed=self.random_seed,
            )
        except Exception as e:
            log.warning(
                "Leiden failed — falling back to Louvain",
                error=str(e),
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
            )
            return self._detect_louvain_fallback(graph)

        # Convert graspologic hierarchy to CommunitySchema list
        return self._parse_graspologic_hierarchy(
            hierarchy=hierarchy,
            node_list=node_list,
            max_levels=max_levels,
            graph=graph,
        )

    def _single_node_communities(self, graph: Any) -> list[CommunitySchema]:
        """
        Fallback when graph has no edges or Leiden/Louvain fails.
        Treats each node as its own C1 community so the pipeline can continue.
        """
        communities = []
        for idx, node_id in enumerate(sorted(graph.nodes())):
            communities.append(CommunitySchema(
                community_id=f"comm_c1_{idx:04d}",
                level=CommunityLevel.C1,
                level_index=idx,
                node_ids=[node_id],
                edge_ids=[],
                node_count=1,
                edge_count=0,
            ))
        log.info(
            "Single-node community fallback complete",
            communities=len(communities),
        )
        return communities

    def _parse_graspologic_hierarchy(
        self,
        hierarchy,
        node_list: list[str],
        max_levels: int,
        graph: Any,
    ) -> list[CommunitySchema]:
        """
        Parse a graspologic HierarchicalCluster into CommunitySchema list.

        graspologic returns a list of (node_id, cluster_id, level) tuples
        where level=0 is the most granular (leaf) and higher levels are coarser.

        We invert this: the paper's C0 is the coarsest, C3 is most granular.
        """
        # Build level → {community_int_id → set of node_ids}
        levels: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

        for partition in hierarchy:
            # graspologic HierarchicalCluster has .node, .cluster, .level attributes
            node_int  = partition.node
            cluster   = partition.cluster
            level     = partition.level

            if level > max_levels:
                continue

            if node_int < len(node_list):
                node_str = node_list[node_int]
                levels[level][cluster].add(node_str)

        # Convert to CommunitySchema
        # Invert levels: graspologic level 0 → paper C3 (most granular)
        # graspologic highest level → paper C0 (coarsest)
        all_communities: list[CommunitySchema] = []
        graspologic_levels = sorted(levels.keys())
        num_levels = min(len(graspologic_levels), max_levels + 1)

        for paper_level_idx, grasp_level in enumerate(reversed(graspologic_levels[:num_levels])):
            level_name = _LEVEL_NAMES[min(paper_level_idx, len(_LEVEL_NAMES) - 1)]
            level_enum = CommunityLevel(level_name)
            clusters = levels[grasp_level]

            for comm_idx, (cluster_int, node_ids) in enumerate(
                sorted(clusters.items())
            ):
                if not node_ids:
                    continue

                node_id_list = sorted(node_ids)
                edge_ids = _get_internal_edge_ids(graph, node_ids)

                all_communities.append(CommunitySchema(
                    community_id=f"comm_{level_name}_{comm_idx:04d}",
                    level=level_enum,
                    level_index=comm_idx,
                    node_ids=node_id_list,
                    edge_ids=edge_ids,
                    node_count=len(node_ids),
                    edge_count=len(edge_ids),
                ))

        return all_communities

    def _detect_louvain_fallback(self, graph: Any) -> list[CommunitySchema]:
        """
        Single-level community detection using NetworkX Louvain.

        Used when graspologic is not available. Produces only C1 communities
        (no hierarchy). This is sufficient for testing and development.
        """
        try:
            import networkx.algorithms.community as nx_comm
            # Use Louvain if available (NetworkX >= 2.7)
            partition = nx_comm.louvain_communities(
                graph, weight="weight", seed=self.random_seed
            )
        except AttributeError:
            # Older NetworkX: use greedy modularity
            import networkx.algorithms.community as nx_comm
            partition = nx_comm.greedy_modularity_communities(graph, weight="weight")

        communities: list[CommunitySchema] = []

        # Create a single level (C1) — treat as the working level
        level_enum = CommunityLevel.C1

        for comm_idx, node_set in enumerate(sorted(partition, key=len, reverse=True)):
            node_ids = sorted(node_set)
            edge_ids = _get_internal_edge_ids(graph, set(node_ids))

            communities.append(CommunitySchema(
                community_id=f"comm_c1_{comm_idx:04d}",
                level=level_enum,
                level_index=comm_idx,
                node_ids=node_ids,
                edge_ids=edge_ids,
                node_count=len(node_ids),
                edge_count=len(edge_ids),
            ))

        log.info(
            "Louvain fallback complete",
            communities=len(communities),
            level="c1",
        )
        return communities

    def get_stats(self, communities: list[CommunitySchema]) -> dict:
        """Return summary statistics for a list of communities."""
        if not communities:
            return {"total": 0}

        counts = _count_by_level(communities)
        sizes  = [c.node_count for c in communities]

        return {
            "total":           len(communities),
            "counts_by_level": counts,
            "avg_size":        round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "min_size":        min(sizes) if sizes else 0,
            "max_size":        max(sizes) if sizes else 0,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_internal_edge_ids(graph: Any, node_set: set[str]) -> list[str]:
    """Return edge IDs for all edges where both endpoints are in node_set."""
    edge_ids = []
    for u, v, data in graph.edges(data=True):
        if u in node_set and v in node_set:
            edge_ids.append(data.get("edge_id", f"{u}__{v}"))
    return edge_ids


def _build_parent_child_links(communities: list[CommunitySchema]) -> None:
    """
    Build parent-child community relationships.

    A community at level C{k} is the parent of C{k+1} communities whose
    nodes are a subset of the parent's nodes.

    Modifies communities in-place.
    """
    # Group by level
    by_level: dict[str, list[CommunitySchema]] = defaultdict(list)
    for comm in communities:
        level_val = comm.level.value if hasattr(comm.level, "value") else str(comm.level)
        by_level[level_val].append(comm)

    # For each adjacent level pair, find node set inclusion
    level_names = [lv for lv in _LEVEL_NAMES if lv in by_level]
    comm_by_id = {c.community_id: c for c in communities}

    for i in range(len(level_names) - 1):
        parent_level = level_names[i]
        child_level  = level_names[i + 1]

        parents  = by_level[parent_level]
        children = by_level[child_level]

        for parent in parents:
            parent_nodes = set(parent.node_ids)
            for child in children:
                child_nodes = set(child.node_ids)
                # Child's nodes are a subset of parent's nodes
                if child_nodes <= parent_nodes or child_nodes & parent_nodes:
                    child.parent_community_id = parent.community_id
                    if child.community_id not in parent.child_community_ids:
                        parent.child_community_ids.append(child.community_id)


def _annotate_graph_nodes(
    graph: Any,
    communities: list[CommunitySchema],
) -> None:
    """
    Update graph node attributes with community membership at each level.

    After this, each node has:
      node["community_ids"] = {"c0": "comm_c0_0003", "c1": "comm_c1_0045", ...}
    """
    for comm in communities:
        level_val = comm.level.value if hasattr(comm.level, "value") else str(comm.level)
        for node_id in comm.node_ids:
            if graph.has_node(node_id):
                if "community_ids" not in graph.nodes[node_id]:
                    graph.nodes[node_id]["community_ids"] = {}
                graph.nodes[node_id]["community_ids"][level_val] = comm.community_id


def _count_by_level(communities: list[CommunitySchema]) -> dict[str, int]:
    """Count communities per level."""
    counts: dict[str, int] = {}
    for comm in communities:
        lv = comm.level.value if hasattr(comm.level, "value") else str(comm.level)
        counts[lv] = counts.get(lv, 0) + 1
    return counts


# ── Factory function ───────────────────────────────────────────────────────────

def get_community_detection() -> CommunityDetection:
    """Build a CommunityDetection from application settings."""
    from app.config import get_settings
    settings = get_settings()
    return CommunityDetection(max_cluster_size=10, random_seed=42)


__all__ = [
    "CommunityDetection",
    "get_community_detection",
]