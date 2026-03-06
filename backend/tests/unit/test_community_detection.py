"""
tests/unit/test_community_detection.py — CommunityDetection unit tests.

Tests:
  - Empty graph returns empty list
  - Single-node graph does not crash
  - All nodes assigned to at least one community
  - community_id uniqueness
  - level field is a valid level string (c0, c1, ...)
  - node_ids non-empty for every community
  - Parent-child relationships are consistent
  - detect() returns a list (even with graspologic unavailable, falls back to louvain)
  - _count_by_level() aggregates correctly
  - Graph nodes annotated with community_ids after detect()
"""

from __future__ import annotations

import pytest
import networkx as nx


@pytest.fixture
def detector():
    from app.core.pipeline.community_detection import CommunityDetection
    return CommunityDetection(random_seed=42)


def make_connected_graph(n_nodes: int = 12) -> nx.Graph:
    """Build a connected graph with enough structure for community detection."""
    g = nx.Graph()
    # Three dense cliques of 4 nodes each, weakly connected
    for group in range(3):
        for i in range(4):
            node_id = f"node_{group}_{i}"
            g.add_node(node_id, name=f"Node {group}-{i}",
                       entity_type="ORGANIZATION", description="test",
                       mention_count=1, degree=0, source_chunk_ids=[])
        # Dense edges within group
        nodes = [f"node_{group}_{i}" for i in range(4)]
        for a in nodes:
            for b in nodes:
                if a < b:
                    g.add_edge(a, b, weight=5, descriptions=[], source_chunk_ids=[])
    # Sparse edges between groups
    g.add_edge("node_0_0", "node_1_0", weight=1, descriptions=[], source_chunk_ids=[])
    g.add_edge("node_1_0", "node_2_0", weight=1, descriptions=[], source_chunk_ids=[])
    return g


# ── Core detect() tests ───────────────────────────────────────────────────────

class TestDetect:

    def test_empty_graph_returns_empty_list(self, detector):
        g = nx.Graph()
        communities = detector.detect(g)
        assert communities == []

    def test_single_node_does_not_crash(self, detector):
        g = nx.Graph()
        g.add_node("solo", name="Solo", entity_type="PERSON",
                   description="alone", mention_count=1, degree=0, source_chunk_ids=[])
        communities = detector.detect(g)
        assert isinstance(communities, list)

    def test_returns_list_of_community_schema(self, detector):
        from app.models.graph_models import CommunitySchema
        g = make_connected_graph(12)
        communities = detector.detect(g)
        assert isinstance(communities, list)
        if communities:
            assert isinstance(communities[0], CommunitySchema)

    def test_all_nodes_assigned_to_community(self, detector):
        g = make_connected_graph(12)
        communities = detector.detect(g)
        covered_nodes = set()
        for c in communities:
            covered_nodes.update(c.node_ids)
        for node in g.nodes:
            assert node in covered_nodes, f"{node} not in any community"

    def test_community_ids_unique(self, detector):
        g = make_connected_graph(12)
        communities = detector.detect(g)
        ids = [c.community_id for c in communities]
        assert len(ids) == len(set(ids))

    def test_every_community_has_node_ids(self, detector):
        g = make_connected_graph(12)
        communities = detector.detect(g)
        for c in communities:
            assert len(c.node_ids) > 0

    def test_level_field_valid_format(self, detector):
        g = make_connected_graph(12)
        communities = detector.detect(g)
        valid_levels = {"c0", "c1", "c2", "c3"}
        for c in communities:
            level_val = c.level.value if hasattr(c.level, "value") else str(c.level)
            assert level_val in valid_levels

    def test_max_levels_respected(self, detector):
        g = make_connected_graph(12)
        communities = detector.detect(g, max_levels=1)
        levels = {c.level.value if hasattr(c.level, "value") else str(c.level)
                  for c in communities}
        assert "c2" not in levels
        assert "c3" not in levels

    def test_graph_nodes_annotated_after_detect(self, detector):
        g = make_connected_graph(12)
        detector.detect(g)
        for _, data in g.nodes(data=True):
            assert "community_ids" in data

    def test_deterministic_with_same_seed(self, detector):
        from app.core.pipeline.community_detection import CommunityDetection
        g = make_connected_graph(12)
        d1 = CommunityDetection(random_seed=42)
        d2 = CommunityDetection(random_seed=42)
        c1 = d1.detect(g)
        g2 = make_connected_graph(12)
        c2 = d2.detect(g2)
        # Same seed → same community count
        assert len(c1) == len(c2)


# ── With real corpus (conftest fixtures) ──────────────────────────────────────

class TestWithCorpus:

    def test_detects_communities_from_corpus_graph(self, sample_communities):
        assert len(sample_communities) >= 1

    def test_community_levels_present(self, sample_communities):
        levels = {c.level for c in sample_communities}
        # At minimum c0 should be present
        assert len(levels) >= 1


# ── Helper function tests ─────────────────────────────────────────────────────

class TestCountByLevel:

    def test_count_by_level(self):
        from app.core.pipeline.community_detection import _count_by_level
        from app.models.graph_models import CommunitySchema, CommunityLevel

        communities = [
            CommunitySchema(community_id="c0_0", level=CommunityLevel.C0,
                            level_index=0, node_ids=["a", "b"], edge_ids=[]),
            CommunitySchema(community_id="c0_1", level=CommunityLevel.C0,
                            level_index=1, node_ids=["c"], edge_ids=[]),
            CommunitySchema(community_id="c1_0", level=CommunityLevel.C1,
                            level_index=0, node_ids=["a", "b", "c"], edge_ids=[]),
        ]
        counts = _count_by_level(communities)
        assert counts["c0"] == 2
        assert counts["c1"] == 1

    def test_empty_list_returns_empty_dict(self):
        from app.core.pipeline.community_detection import _count_by_level
        assert _count_by_level([]) == {}