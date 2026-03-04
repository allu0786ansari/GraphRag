"""
core/pipeline/graph_builder.py — Build NetworkX knowledge graph from extractions.

Implements paper Section 3.1.3:
  "We convert the extracted entities and relationships into a graph. Entities
   with the same name are merged into a single node. Their descriptions are
   concatenated and the node's degree (number of relationships) is computed."

Input:  list[ChunkExtraction] from extraction stage
Output: NetworkX Graph with:
  - Nodes: one per unique entity (normalized name as node ID)
    Attributes: name, entity_type, description, degree, source_chunk_ids,
                claims, mention_count, community_ids (set later by community detection)
  - Edges: one per unique relationship pair
    Attributes: description, weight (co-occurrence count), source_chunk_ids,
                combined_degree (set after all nodes are added)

Merging rules:
  - Entity node IDs are normalized (lowercase, stripped).
  - When the same entity appears in multiple chunks, descriptions are
    concatenated (separated by ". ") for later LLM summarization.
  - Edge weight = number of chunks where this relationship pair appeared.
  - Entity type = the most commonly assigned type across all extractions.

combined_degree (paper Section 3.1.4):
  combined_degree = source_node_degree + target_node_degree
  Used during community summarization to prioritize which edges to include
  in the context window (highest combined_degree edges first).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from app.models.graph_models import ChunkExtraction
from app.utils.logger import get_logger

log = get_logger(__name__)

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False
    log.warning("networkx not installed — GraphBuilder will not function")


class GraphBuilder:
    """
    Builds a NetworkX knowledge graph from a list of ChunkExtraction objects.

    Usage:
        builder = GraphBuilder()
        graph = builder.build(extractions)
        # graph is a networkx.Graph
    """

    def build(
        self,
        extractions: list[ChunkExtraction],
        min_entity_mentions: int = 1,
    ) -> Any:  # networkx.Graph
        """
        Build a NetworkX Graph from all chunk extractions.

        Args:
            extractions:           Extractions from the extraction stage.
                                   Only extractions with extraction_completed=True are used.
            min_entity_mentions:   Filter out entities seen fewer than this many
                                   times across all chunks. Default: 1 (keep all).
                                   Increase to 2+ to filter noise for large corpora.

        Returns:
            networkx.Graph with node and edge attributes per NodeSchema/EdgeSchema.
        """
        if not _HAS_NETWORKX:
            raise ImportError("networkx is required for GraphBuilder")

        t0 = time.monotonic()
        successful = [e for e in extractions if e.extraction_completed]

        if not successful:
            log.warning("No successful extractions to build graph from")
            return nx.Graph()

        log.info(
            "Graph construction started",
            total_extractions=len(extractions),
            successful=len(successful),
            min_entity_mentions=min_entity_mentions,
        )

        # ── Pass 1: Collect entity data ────────────────────────────────────────
        # entity_data[normalized_name] = {
        #   "name": display_name, "types": Counter, "descriptions": list,
        #   "source_chunk_ids": set, "claims": list, "mention_count": int
        # }
        entity_data: dict[str, dict] = {}

        for ext in successful:
            for entity in ext.entities:
                node_id = _normalize(entity.name)
                if not node_id:
                    continue

                if node_id not in entity_data:
                    entity_data[node_id] = {
                        "name":            entity.name,
                        "types":           Counter(),
                        "descriptions":    [],
                        "source_chunk_ids": set(),
                        "claims":          [],
                        "mention_count":   0,
                    }

                data = entity_data[node_id]
                data["types"][entity.entity_type] += 1
                if entity.description and entity.description not in data["descriptions"]:
                    data["descriptions"].append(entity.description)
                data["source_chunk_ids"].add(entity.source_chunk_id)
                data["mention_count"] += 1

                # Prefer longer/more specific display name (most common)
                if len(entity.name) > len(data["name"]):
                    data["name"] = entity.name

            for claim in ext.claims:
                node_id = _normalize(claim.subject_entity)
                if node_id in entity_data:
                    entity_data[node_id]["claims"].append(claim.claim_description)

        # Apply mention filter
        if min_entity_mentions > 1:
            before = len(entity_data)
            entity_data = {
                k: v for k, v in entity_data.items()
                if v["mention_count"] >= min_entity_mentions
            }
            log.debug(
                "Entity mention filter applied",
                before=before,
                after=len(entity_data),
                threshold=min_entity_mentions,
            )

        # ── Pass 2: Collect relationship data ──────────────────────────────────
        # rel_data[(src_id, tgt_id)] = {
        #   "descriptions": list, "weights": int, "source_chunk_ids": set
        # }
        rel_data: dict[tuple[str, str], dict] = {}

        for ext in successful:
            for rel in ext.relationships:
                src_id = _normalize(rel.source_entity)
                tgt_id = _normalize(rel.target_entity)

                # Skip self-loops and relationships where either entity was filtered
                if not src_id or not tgt_id or src_id == tgt_id:
                    continue
                if src_id not in entity_data or tgt_id not in entity_data:
                    continue

                # Use sorted pair to avoid duplicate (A→B) and (B→A) edges
                # in an undirected graph context
                key = (min(src_id, tgt_id), max(src_id, tgt_id))

                if key not in rel_data:
                    rel_data[key] = {
                        "descriptions": [],
                        "weight":       0,
                        "source_chunk_ids": set(),
                        "strength_sum": 0,
                        "strength_count": 0,
                    }

                data = rel_data[key]
                if rel.description and rel.description not in data["descriptions"]:
                    data["descriptions"].append(rel.description)
                data["weight"] += 1
                data["source_chunk_ids"].add(rel.source_chunk_id)
                data["strength_sum"] += rel.strength
                data["strength_count"] += 1

        # ── Pass 3: Build the graph ────────────────────────────────────────────
        graph = nx.Graph()

        for node_id, data in entity_data.items():
            description = " ".join(data["descriptions"])
            if not description:
                description = f"Entity: {data['name']}"

            graph.add_node(
                node_id,
                name=data["name"],
                entity_type=data["types"].most_common(1)[0][0] if data["types"] else "UNKNOWN",
                description=description,
                degree=0,           # set below
                source_chunk_ids=sorted(data["source_chunk_ids"]),
                claims=data["claims"],
                mention_count=data["mention_count"],
                community_ids={},   # filled by community detection stage
            )

        for (src_id, tgt_id), data in rel_data.items():
            description = " ".join(data["descriptions"])
            avg_strength = (
                data["strength_sum"] / data["strength_count"]
                if data["strength_count"] > 0 else 1.0
            )

            graph.add_edge(
                src_id,
                tgt_id,
                edge_id=f"{src_id}__{tgt_id}",
                description=description,
                weight=float(data["weight"]),
                source_chunk_ids=sorted(data["source_chunk_ids"]),
                combined_degree=0,  # set below
                avg_strength=round(avg_strength, 2),
            )

        # ── Pass 4: Compute degrees and combined_degree ────────────────────────
        for node_id in graph.nodes:
            deg = graph.degree(node_id)
            graph.nodes[node_id]["degree"] = deg

        for src_id, tgt_id in graph.edges:
            src_deg = graph.nodes[src_id]["degree"]
            tgt_deg = graph.nodes[tgt_id]["degree"]
            graph.edges[src_id, tgt_id]["combined_degree"] = src_deg + tgt_deg

        elapsed = time.monotonic() - t0
        log.info(
            "Graph construction complete",
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            elapsed_seconds=round(elapsed, 2),
        )

        return graph

    def get_graph_stats(self, graph: Any) -> dict:
        """Return summary statistics about a built graph."""
        if not _HAS_NETWORKX or graph is None:
            return {}

        degrees = [d for _, d in graph.degree()]
        weights = [graph.edges[u, v].get("weight", 1.0) for u, v in graph.edges]
        entity_types: dict[str, int] = {}
        for node in graph.nodes:
            etype = graph.nodes[node].get("entity_type", "UNKNOWN")
            entity_types[etype] = entity_types.get(etype, 0) + 1

        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
            "avg_edge_weight": round(sum(weights) / len(weights), 2) if weights else 0.0,
            "entity_type_distribution": dict(
                sorted(entity_types.items(), key=lambda x: -x[1])
            ),
            "top_nodes_by_degree": sorted(
                [(n, graph.nodes[n].get("name", n), graph.nodes[n].get("degree", 0))
                 for n in graph.nodes],
                key=lambda x: -x[2],
            )[:10],
        }

    def get_node_context(
        self,
        graph: Any,
        node_id: str,
        max_edges: int = 50,
    ) -> str:
        """
        Build context text for a node (entity + its top edges by combined_degree).

        Used during community summarization to build context windows.
        Returns edge descriptions sorted by combined_degree descending.
        """
        if not graph.has_node(node_id):
            return ""

        node_data = graph.nodes[node_id]
        context_parts = [
            f"Entity: {node_data.get('name', node_id)}",
            f"Type: {node_data.get('entity_type', 'UNKNOWN')}",
            f"Description: {node_data.get('description', '')}",
        ]

        # Add edges sorted by combined_degree (paper Section 3.1.4)
        edges = []
        for nbr in graph.neighbors(node_id):
            edge_data = graph.edges[node_id, nbr]
            edges.append((
                edge_data.get("combined_degree", 0),
                nbr,
                edge_data.get("description", ""),
            ))

        edges.sort(key=lambda x: -x[0])
        for i, (_, nbr_id, desc) in enumerate(edges[:max_edges]):
            nbr_name = graph.nodes[nbr_id].get("name", nbr_id)
            context_parts.append(f"Relationship with {nbr_name}: {desc}")

        return "\n".join(context_parts)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """
    Normalize an entity name to a graph node ID.

    - Lowercase
    - Replace whitespace sequences with single underscore
    - Remove characters that are not alphanumeric or underscore
    - Strip leading/trailing underscores
    """
    if not name or not name.strip():
        return ""
    normalized = name.strip().lower()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w]", "", normalized)
    normalized = normalized.strip("_")
    return normalized


# ── Factory function ───────────────────────────────────────────────────────────

def get_graph_builder() -> GraphBuilder:
    """Build a GraphBuilder (no dependencies needed)."""
    return GraphBuilder()


__all__ = [
    "GraphBuilder",
    "get_graph_builder",
]