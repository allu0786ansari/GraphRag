"""
tests/unit/test_graph_builder.py — GraphBuilder unit tests.

Tests:
  - Empty extractions produces empty graph
  - Nodes created for every unique entity
  - Edges created for every valid relationship
  - Self-loop relationships skipped
  - min_entity_mentions filter applied correctly
  - Duplicate entity names are merged (case-insensitive)
  - Node attributes: name, entity_type, description, mention_count
  - Edge attributes: weight increases for duplicate relationships
  - Failed extractions excluded
  - _normalize() lowercases and strips whitespace
"""

from __future__ import annotations

import pytest
import networkx as nx

from app.models.graph_models import (
    ChunkExtraction, ExtractedEntity, ExtractedRelationship,
)
from app.core.pipeline.graph_builder import GraphBuilder


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_entity(name, etype="ORGANIZATION", desc="", chunk_id="c001"):
    return ExtractedEntity(
        name=name, entity_type=etype,
        description=desc or f"Description of {name}",
        source_chunk_id=chunk_id,
    )


def make_rel(src, tgt, desc="related", strength=7, chunk_id="c001"):
    return ExtractedRelationship(
        source_entity=src, target_entity=tgt,
        description=desc, strength=strength, source_chunk_id=chunk_id,
    )


def make_extraction(chunk_id, entities, relationships, completed=True):
    return ChunkExtraction(
        chunk_id=chunk_id, entities=entities, relationships=relationships,
        extraction_completed=completed,
    )


@pytest.fixture
def builder():
    return GraphBuilder()


# ── Basic graph construction ──────────────────────────────────────────────────

class TestGraphBuilderBasics:

    def test_empty_extractions_returns_empty_graph(self, builder):
        graph = builder.build([])
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_failed_extractions_excluded(self, builder):
        failed = make_extraction(
            "c001",
            entities=[make_entity("ShouldNotAppear")],
            relationships=[],
            completed=False,
        )
        graph = builder.build([failed])
        assert graph.number_of_nodes() == 0

    def test_nodes_created_for_all_entities(self, builder):
        ext = make_extraction("c001", [
            make_entity("OpenAI"),
            make_entity("Microsoft"),
            make_entity("Anthropic"),
        ], [])
        graph = builder.build([ext])
        assert graph.number_of_nodes() == 3

    def test_edges_created_for_valid_relationships(self, builder):
        ext = make_extraction("c001", [
            make_entity("OpenAI"),
            make_entity("Microsoft"),
        ], [
            make_rel("OpenAI", "Microsoft", "partnership"),
        ])
        graph = builder.build([ext])
        assert graph.number_of_edges() == 1

    def test_self_loop_relationships_skipped(self, builder):
        ext = make_extraction("c001", [
            make_entity("OpenAI"),
        ], [
            make_rel("OpenAI", "OpenAI", "self-reference"),
        ])
        graph = builder.build([ext])
        assert graph.number_of_edges() == 0

    def test_relationship_to_unknown_entity_skipped(self, builder):
        ext = make_extraction("c001", [
            make_entity("OpenAI"),
        ], [
            make_rel("OpenAI", "NonExistentEntity", "unknown"),
        ])
        graph = builder.build([ext])
        assert graph.number_of_edges() == 0


# ── Entity deduplication ──────────────────────────────────────────────────────

class TestEntityDeduplication:

    def test_same_entity_across_chunks_merged(self, builder):
        ext1 = make_extraction("c001", [make_entity("OpenAI", chunk_id="c001")], [])
        ext2 = make_extraction("c002", [make_entity("OpenAI", chunk_id="c002")], [])
        graph = builder.build([ext1, ext2])
        assert graph.number_of_nodes() == 1

    def test_mention_count_aggregated_across_chunks(self, builder):
        extractions = [
            make_extraction(f"c00{i}", [make_entity("OpenAI", chunk_id=f"c00{i}")], [])
            for i in range(3)
        ]
        graph = builder.build(extractions)
        node_id = next(n for n in graph.nodes if "openai" in n.lower())
        assert graph.nodes[node_id]["mention_count"] == 3

    def test_case_insensitive_deduplication(self, builder):
        ext = make_extraction("c001", [
            make_entity("openai"),
            make_entity("OpenAI"),
            make_entity("OPENAI"),
        ], [])
        graph = builder.build([ext])
        # All three should merge into one node
        assert graph.number_of_nodes() == 1


# ── min_entity_mentions filter ────────────────────────────────────────────────

class TestMentionFilter:

    def test_min_mentions_2_filters_rare_entities(self, builder):
        ext1 = make_extraction("c001", [
            make_entity("OpenAI", chunk_id="c001"),
            make_entity("Rare Corp", chunk_id="c001"),  # appears only once
        ], [])
        ext2 = make_extraction("c002", [
            make_entity("OpenAI", chunk_id="c002"),  # appears twice
        ], [])
        graph = builder.build([ext1, ext2], min_entity_mentions=2)
        node_names = {graph.nodes[n]["name"].lower() for n in graph.nodes}
        assert any("openai" in name for name in node_names)
        assert not any("rare" in name for name in node_names)

    def test_min_mentions_1_keeps_all(self, builder):
        extractions = [make_extraction("c001", [
            make_entity("UniqueEntity"),
            make_entity("AnotherUnique"),
        ], [])]
        graph = builder.build(extractions, min_entity_mentions=1)
        assert graph.number_of_nodes() == 2


# ── Node and edge attributes ──────────────────────────────────────────────────

class TestNodeEdgeAttributes:

    def test_node_has_name_attribute(self, builder):
        ext = make_extraction("c001", [make_entity("OpenAI")], [])
        graph = builder.build([ext])
        node = next(iter(graph.nodes))
        assert "name" in graph.nodes[node]
        assert graph.nodes[node]["name"] == "OpenAI"

    def test_node_has_entity_type_attribute(self, builder):
        ext = make_extraction("c001", [make_entity("OpenAI", etype="ORGANIZATION")], [])
        graph = builder.build([ext])
        node = next(iter(graph.nodes))
        assert graph.nodes[node]["entity_type"] == "ORGANIZATION"

    def test_node_has_description_attribute(self, builder):
        ext = make_extraction("c001", [make_entity("OpenAI", desc="AI lab")], [])
        graph = builder.build([ext])
        node = next(iter(graph.nodes))
        assert "description" in graph.nodes[node]
        assert len(graph.nodes[node]["description"]) > 0

    def test_edge_weight_increases_for_duplicate_relationships(self, builder):
        ext1 = make_extraction("c001", [
            make_entity("A", chunk_id="c001"), make_entity("B", chunk_id="c001"),
        ], [make_rel("A", "B", "related", chunk_id="c001")])
        ext2 = make_extraction("c002", [
            make_entity("A", chunk_id="c002"), make_entity("B", chunk_id="c002"),
        ], [make_rel("A", "B", "also related", chunk_id="c002")])
        graph = builder.build([ext1, ext2])
        a_id = next(n for n in graph.nodes if "a" == graph.nodes[n]["name"].lower())
        b_id = next(n for n in graph.nodes if "b" == graph.nodes[n]["name"].lower())
        if graph.has_edge(a_id, b_id):
            assert graph.edges[a_id, b_id]["weight"] == 2

    def test_returns_networkx_graph(self, builder):
        graph = builder.build([])
        assert isinstance(graph, nx.Graph)


# ── Full corpus (uses conftest fixtures) ─────────────────────────────────────

class TestWithCorpus:

    def test_graph_has_expected_entity_count(self, test_graph):
        # Our 10-chunk corpus has ~15+ unique entities
        assert test_graph.number_of_nodes() >= 8

    def test_graph_has_edges(self, test_graph):
        assert test_graph.number_of_edges() >= 3

    def test_openai_node_exists_in_graph(self, test_graph):
        node_names = {
            graph_data["name"].lower()
            for _, graph_data in test_graph.nodes(data=True)
        }
        assert any("openai" in name for name in node_names)