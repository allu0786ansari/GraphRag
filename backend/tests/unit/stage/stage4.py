"""
tests/unit/test_stage4_storage.py — Stage 4 storage layer tests.

Test strategy:
  - All tests use pytest's tmp_path fixture for isolated temp directories.
  - No mocking needed — all four stores use real file I/O (the whole point).
  - Tests verify: save → reload roundtrip, atomic writes, corrupt record
    handling, cache behaviour, resume logic, and cleanup.
  - NetworkX is used for graph tests (lightweight, no API calls needed).

Coverage targets:
  artifact_store.py  : 95%+
  graph_store.py     : 95%+
  summary_store.py   : 95%+
  cache_manager.py   : 95%+
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def artifacts_dir(tmp_path) -> Path:
    """Fresh temp directory for each test — completely isolated."""
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


# ── Model factories ────────────────────────────────────────────────────────────

def make_chunk(index: int = 0, source: str = "doc_001.json") -> "ChunkSchema":
    from app.models.graph_models import ChunkSchema
    return ChunkSchema(
        chunk_id=f"doc_001_{index:04d}",
        source_document=source,
        text=f"This is the text of chunk {index}. It discusses various topics.",
        token_count=15,
        start_char=index * 100,
        end_char=(index * 100) + 65,
        chunk_index=index,
        total_chunks_in_doc=10,
    )


def make_extraction(chunk_id: str = "doc_001_0000") -> "ChunkExtraction":
    from app.models.graph_models import (
        ChunkExtraction, ExtractedEntity, ExtractedRelationship,
    )
    return ChunkExtraction(
        chunk_id=chunk_id,
        entities=[
            ExtractedEntity(
                name="OpenAI",
                entity_type="ORGANIZATION",
                description="AI research company.",
                source_chunk_id=chunk_id,
            )
        ],
        relationships=[
            ExtractedRelationship(
                source_entity="Microsoft",
                target_entity="OpenAI",
                description="Microsoft invested in OpenAI.",
                strength=9,
                source_chunk_id=chunk_id,
            )
        ],
        gleaning_rounds_completed=2,
        extraction_completed=True,
    )


def make_community(level: str = "c1", index: int = 0) -> "CommunitySchema":
    from app.models.graph_models import CommunitySchema, CommunityLevel
    level_enum = CommunityLevel(level)
    return CommunitySchema(
        community_id=f"comm_{level}_{index:04d}",
        level=level_enum,
        level_index=index,
        node_ids=["openai", "microsoft", "sam_altman"],
        edge_ids=["microsoft__openai"],
        node_count=3,
        edge_count=1,
    )


def make_summary(level: str = "c1", index: int = 0) -> "CommunitySummary":
    from app.models.graph_models import CommunitySummary, CommunityFinding, CommunityLevel
    level_enum = CommunityLevel(level)
    return CommunitySummary(
        community_id=f"comm_{level}_{index:04d}",
        level=level_enum,
        title=f"Community {index} at level {level}",
        summary=f"This community covers AI investment topics (community {index}).",
        impact_rating=7.5,
        rating_explanation="Central to AI industry dynamics.",
        findings=[
            CommunityFinding(
                finding_id=0,
                summary="Microsoft invested $10B in OpenAI.",
                explanation="This made Microsoft the primary commercial partner.",
            )
        ],
        node_ids=["openai", "microsoft"],
        context_tokens_used=4500,
    )


def make_nx_graph() -> "networkx.Graph":
    import networkx as nx
    g = nx.Graph()
    g.add_node("openai",    name="OpenAI",    entity_type="ORGANIZATION",
               description="AI research company.", degree=5)
    g.add_node("microsoft", name="Microsoft", entity_type="ORGANIZATION",
               description="Technology company.",   degree=5)
    g.add_node("sam_altman", name="Sam Altman", entity_type="PERSON",
               description="CEO of OpenAI.",       degree=3)
    g.add_edge("openai", "microsoft",
               description="Microsoft invested in OpenAI.", weight=9.0, combined_degree=10)
    g.add_edge("openai", "sam_altman",
               description="Sam Altman leads OpenAI.",     weight=8.0, combined_degree=8)
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:

    def test_artifact_store_imports(self):
        from app.storage.artifact_store import (
            ArtifactStore, get_artifact_store,
            CHUNKS_FILENAME, EXTRACTIONS_FILENAME,
        )
        assert ArtifactStore is not None
        assert CHUNKS_FILENAME == "chunks.json"
        assert EXTRACTIONS_FILENAME == "extractions.json"

    def test_graph_store_imports(self):
        from app.storage.graph_store import (
            GraphStore, get_graph_store,
            GRAPH_FILENAME, COMMUNITY_MAP_FILENAME,
        )
        assert GraphStore is not None
        assert GRAPH_FILENAME == "graph.pkl"
        assert COMMUNITY_MAP_FILENAME == "community_map.json"

    def test_summary_store_imports(self):
        from app.storage.summary_store import (
            SummaryStore, get_summary_store, SUMMARIES_FILENAME,
        )
        assert SummaryStore is not None
        assert SUMMARIES_FILENAME == "community_summaries.json"

    def test_cache_manager_imports(self):
        from app.storage.cache_manager import (
            CacheManager, get_cache_manager, ALL_STAGES, STATE_FILENAME,
        )
        assert CacheManager is not None
        assert STATE_FILENAME == "pipeline_state.json"
        assert len(ALL_STAGES) == 6

    def test_storage_init_exports(self):
        from app.storage import (
            ArtifactStore, GraphStore, SummaryStore, CacheManager,
            get_artifact_store, get_graph_store,
            get_summary_store, get_cache_manager,
            CHUNKS_FILENAME, EXTRACTIONS_FILENAME,
            GRAPH_FILENAME, COMMUNITY_MAP_FILENAME,
            SUMMARIES_FILENAME, STATE_FILENAME, ALL_STAGES,
        )
        assert all(x is not None for x in [
            ArtifactStore, GraphStore, SummaryStore, CacheManager
        ])

    def test_no_circular_imports(self):
        import importlib
        for module in [
            "app.storage.artifact_store",
            "app.storage.graph_store",
            "app.storage.summary_store",
            "app.storage.cache_manager",
        ]:
            mod = importlib.import_module(module)
            assert mod is not None


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT STORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtifactStoreInit:

    def test_creates_directory(self, tmp_path):
        from app.storage.artifact_store import ArtifactStore
        new_dir = tmp_path / "new" / "nested" / "dir"
        store = ArtifactStore(artifacts_dir=new_dir)
        assert new_dir.exists()

    def test_repr(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        r = repr(store)
        assert "ArtifactStore" in r
        assert "chunks_exist=False" in r

    def test_initial_state(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        assert store.chunks_exist() is False
        assert store.extractions_exist() is False
        assert store.chunks_count() == 0
        assert store.extractions_count() == 0


class TestArtifactStoreChunks:

    def test_save_and_load_roundtrip(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        chunks = [make_chunk(i) for i in range(5)]

        store.save_chunks(chunks)
        assert store.chunks_exist() is True
        assert store.chunks_count() == 5

        loaded = store.load_chunks()
        assert len(loaded) == 5
        assert loaded[0].chunk_id == "doc_001_0000"
        assert loaded[4].chunk_id == "doc_001_0004"

    def test_chunk_fields_preserved(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        chunk = make_chunk(0)
        chunk.metadata = {"date": "2024-01-15", "category": "tech"}

        store.save_chunks([chunk])
        loaded = store.load_chunks()

        assert loaded[0].source_document == "doc_001.json"
        assert loaded[0].token_count == 15
        assert loaded[0].metadata["date"] == "2024-01-15"

    def test_save_overwrites_existing(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(0)])
        assert store.chunks_count() == 1

        store.save_chunks([make_chunk(0), make_chunk(1), make_chunk(2)])
        assert store.chunks_count() == 3

    def test_load_chunks_file_not_found(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        with pytest.raises(FileNotFoundError, match="chunks.json"):
            store.load_chunks()

    def test_chunks_file_is_valid_json(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(0)])

        chunks_path = artifacts_dir / "chunks.json"
        with open(chunks_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["chunk_id"] == "doc_001_0000"

    def test_load_chunks_iter(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(i) for i in range(10)])

        loaded = list(store.load_chunks_iter())
        assert len(loaded) == 10
        assert loaded[5].chunk_id == "doc_001_0005"

    def test_delete_chunks(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(0)])
        assert store.chunks_exist() is True

        result = store.delete_chunks()
        assert result is True
        assert store.chunks_exist() is False

    def test_delete_nonexistent_chunks_returns_false(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        assert store.delete_chunks() is False

    def test_corrupt_chunk_skipped(self, artifacts_dir):
        """A corrupt record in chunks.json is skipped, not raised."""
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)

        # Write a file with one valid and one corrupt record
        chunks_path = artifacts_dir / "chunks.json"
        with open(chunks_path, "w") as f:
            json.dump([
                make_chunk(0).model_dump(mode="json"),
                {"chunk_id": "bad", "missing_required_field": True},
            ], f)

        loaded = store.load_chunks()
        assert len(loaded) == 1  # corrupt record skipped
        assert loaded[0].chunk_id == "doc_001_0000"


class TestArtifactStoreExtractions:

    def test_save_and_load_roundtrip(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        extractions = [make_extraction(f"doc_001_{i:04d}") for i in range(3)]

        store.save_extractions(extractions)
        assert store.extractions_exist() is True
        assert store.extractions_count() == 3

        loaded = store.load_extractions()
        assert len(loaded) == 3
        assert loaded[0].chunk_id == "doc_001_0000"

    def test_extraction_fields_preserved(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        ext = make_extraction("doc_001_0000")

        store.save_extractions([ext])
        loaded = store.load_extractions()

        assert len(loaded[0].entities) == 1
        assert loaded[0].entities[0].name == "OpenAI"
        assert len(loaded[0].relationships) == 1
        assert loaded[0].relationships[0].strength == 9
        assert loaded[0].gleaning_rounds_completed == 2

    def test_append_extraction(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)

        store.append_extraction(make_extraction("doc_001_0000"))
        assert store.extractions_count() == 1

        store.append_extraction(make_extraction("doc_001_0001"))
        assert store.extractions_count() == 2

        store.append_extraction(make_extraction("doc_001_0002"))
        assert store.extractions_count() == 3

    def test_append_starts_fresh_when_no_file(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        assert not store.extractions_exist()

        store.append_extraction(make_extraction("doc_001_0000"))
        assert store.extractions_exist()
        assert store.extractions_count() == 1

    def test_save_extractions_batch(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)

        batch1 = [make_extraction(f"doc_001_{i:04d}") for i in range(5)]
        batch2 = [make_extraction(f"doc_001_{i:04d}") for i in range(5, 10)]

        store.save_extractions_batch(batch1)
        assert store.extractions_count() == 5

        store.save_extractions_batch(batch2)
        assert store.extractions_count() == 10

    def test_save_extractions_batch_empty(self, artifacts_dir):
        """Empty batch should not change anything."""
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_extractions_batch([])
        assert not store.extractions_exist()

    def test_load_extractions_as_dict(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        extractions = [make_extraction(f"doc_001_{i:04d}") for i in range(3)]
        store.save_extractions(extractions)

        result = store.load_extractions_as_dict()
        assert isinstance(result, dict)
        assert "doc_001_0000" in result
        assert "doc_001_0002" in result
        assert result["doc_001_0001"].chunk_id == "doc_001_0001"

    def test_get_extracted_chunk_ids(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        extractions = [make_extraction(f"chunk_{i}") for i in range(5)]
        store.save_extractions(extractions)

        ids = store.get_extracted_chunk_ids()
        assert ids == {"chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"}

    def test_get_extracted_chunk_ids_empty(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        assert store.get_extracted_chunk_ids() == set()

    def test_load_extractions_file_not_found(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        with pytest.raises(FileNotFoundError, match="extractions.json"):
            store.load_extractions()

    def test_delete_extractions(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_extractions([make_extraction("c0")])
        assert store.extractions_exist()

        result = store.delete_extractions()
        assert result is True
        assert not store.extractions_exist()


class TestArtifactStoreAtomicWrite:

    def test_file_is_valid_after_save(self, artifacts_dir):
        """After save, the file must be a valid complete JSON array."""
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        chunks = [make_chunk(i) for i in range(20)]
        store.save_chunks(chunks)

        with open(artifacts_dir / "chunks.json") as f:
            content = f.read()
        data = json.loads(content)
        assert len(data) == 20

    def test_no_tmp_files_remain_after_save(self, artifacts_dir):
        """Temp files from atomic write should be cleaned up."""
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(i) for i in range(5)])

        tmp_files = list(artifacts_dir.glob("*.tmp*"))
        assert len(tmp_files) == 0


class TestArtifactStoreDeleteAll:

    def test_delete_all(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(0)])
        store.save_extractions([make_extraction("doc_001_0000")])

        store.delete_all()
        assert not store.chunks_exist()
        assert not store.extractions_exist()


class TestArtifactStoreStats:

    def test_get_stats(self, artifacts_dir):
        from app.storage.artifact_store import ArtifactStore
        store = ArtifactStore(artifacts_dir=artifacts_dir)
        store.save_chunks([make_chunk(i) for i in range(3)])

        stats = store.get_stats()
        assert stats["chunks"]["exists"] is True
        assert stats["chunks"]["count"] == 3
        assert stats["extractions"]["exists"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphStoreInit:

    def test_creates_directory(self, tmp_path):
        from app.storage.graph_store import GraphStore
        new_dir = tmp_path / "new_graph_dir"
        store = GraphStore(artifacts_dir=new_dir)
        assert new_dir.exists()

    def test_initial_state(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        assert store.graph_exists() is False
        assert store.community_map_exists() is False

    def test_repr(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        r = repr(store)
        assert "GraphStore" in r
        assert "graph_exists=False" in r


class TestGraphStoreSaveLoadGraph:

    def test_save_and_load_roundtrip(self, artifacts_dir):
        import networkx as nx
        from app.storage.graph_store import GraphStore

        store = GraphStore(artifacts_dir=artifacts_dir)
        graph = make_nx_graph()

        store.save_graph(graph)
        assert store.graph_exists() is True

        loaded = store.load_graph()
        assert loaded.number_of_nodes() == 3
        assert loaded.number_of_edges() == 2

    def test_node_attributes_preserved(self, artifacts_dir):
        import networkx as nx
        from app.storage.graph_store import GraphStore

        store = GraphStore(artifacts_dir=artifacts_dir)
        graph = make_nx_graph()
        store.save_graph(graph)

        loaded = store.load_graph()
        assert loaded.nodes["openai"]["name"] == "OpenAI"
        assert loaded.nodes["openai"]["entity_type"] == "ORGANIZATION"
        assert loaded.nodes["openai"]["degree"] == 5

    def test_edge_attributes_preserved(self, artifacts_dir):
        import networkx as nx
        from app.storage.graph_store import GraphStore

        store = GraphStore(artifacts_dir=artifacts_dir)
        graph = make_nx_graph()
        store.save_graph(graph)

        loaded = store.load_graph()
        edge_data = loaded.edges["openai", "microsoft"]
        assert edge_data["weight"] == 9.0
        assert edge_data["combined_degree"] == 10

    def test_load_graph_file_not_found(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        with pytest.raises(FileNotFoundError, match="graph.pkl"):
            store.load_graph()

    def test_graph_pkl_is_binary_file(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        store.save_graph(make_nx_graph())

        graph_path = artifacts_dir / "graph.pkl"
        assert graph_path.exists()
        with open(graph_path, "rb") as f:
            first_bytes = f.read(4)
        # Pickle files start with specific magic bytes (protocol header)
        assert len(first_bytes) == 4

    def test_save_overwrites_existing(self, artifacts_dir):
        import networkx as nx
        from app.storage.graph_store import GraphStore

        store = GraphStore(artifacts_dir=artifacts_dir)
        small_graph = nx.Graph()
        small_graph.add_node("a")
        store.save_graph(small_graph)

        big_graph = make_nx_graph()
        store.save_graph(big_graph)

        loaded = store.load_graph()
        assert loaded.number_of_nodes() == 3  # big_graph, not small_graph

    def test_get_graph_stats(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        store.save_graph(make_nx_graph())

        stats = store.get_graph_stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 2
        assert stats["exists"] is True

    def test_get_graph_stats_not_built(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        stats = store.get_graph_stats()
        assert stats["exists"] is False
        assert stats["nodes"] == 0

    def test_delete_graph(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        store.save_graph(make_nx_graph())
        assert store.graph_exists() is True

        result = store.delete_graph()
        assert result is True
        assert store.graph_exists() is False

    def test_delete_nonexistent_graph_returns_false(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        assert store.delete_graph() is False


class TestGraphStoreCommunityMap:

    def test_save_and_load_roundtrip(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)

        communities = (
            [make_community("c0", i) for i in range(3)] +
            [make_community("c1", i) for i in range(8)] +
            [make_community("c2", i) for i in range(15)]
        )
        store.save_community_map(communities)
        assert store.community_map_exists() is True

        loaded = store.load_community_map()
        assert len(loaded) == 26

    def test_community_fields_preserved(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        comm = make_community("c1", 0)
        store.save_community_map([comm])

        loaded = store.load_community_map()
        assert loaded[0].community_id == "comm_c1_0000"
        assert loaded[0].node_count == 3
        assert "openai" in loaded[0].node_ids

    def test_load_community_map_by_level(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)

        communities = (
            [make_community("c0", i) for i in range(2)] +
            [make_community("c1", i) for i in range(5)] +
            [make_community("c2", i) for i in range(10)]
        )
        store.save_community_map(communities)

        by_level = store.load_community_map_by_level()
        assert len(by_level["c0"]) == 2
        assert len(by_level["c1"]) == 5
        assert len(by_level["c2"]) == 10

    def test_get_community_counts(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)

        communities = (
            [make_community("c0", i) for i in range(3)] +
            [make_community("c1", i) for i in range(7)]
        )
        store.save_community_map(communities)

        counts = store.get_community_counts()
        assert counts.get("c0") == 3
        assert counts.get("c1") == 7

    def test_get_community_counts_empty(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        assert store.get_community_counts() == {}

    def test_load_community_map_file_not_found(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        with pytest.raises(FileNotFoundError, match="community_map.json"):
            store.load_community_map()

    def test_community_map_is_valid_json(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        store.save_community_map([make_community("c1", 0)])

        with open(artifacts_dir / "community_map.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["community_id"] == "comm_c1_0000"

    def test_delete_community_map(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        store.save_community_map([make_community()])
        assert store.community_map_exists()

        result = store.delete_community_map()
        assert result is True
        assert not store.community_map_exists()

    def test_corrupt_community_record_skipped(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)

        community_path = artifacts_dir / "community_map.json"
        with open(community_path, "w") as f:
            json.dump([
                make_community("c1", 0).model_dump(mode="json"),
                {"community_id": "bad", "missing_fields": True},
            ], f)

        loaded = store.load_community_map()
        assert len(loaded) == 1


class TestGraphStoreDeleteAll:

    def test_delete_all(self, artifacts_dir):
        from app.storage.graph_store import GraphStore
        store = GraphStore(artifacts_dir=artifacts_dir)
        store.save_graph(make_nx_graph())
        store.save_community_map([make_community()])

        store.delete_all()
        assert not store.graph_exists()
        assert not store.community_map_exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY STORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummaryStoreInit:

    def test_creates_directory(self, tmp_path):
        from app.storage.summary_store import SummaryStore
        new_dir = tmp_path / "new_summary_dir"
        store = SummaryStore(artifacts_dir=new_dir)
        assert new_dir.exists()

    def test_initial_state(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        assert store.summaries_exist() is False
        assert store.total_summaries() == 0
        assert store._cache is None

    def test_repr(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        r = repr(store)
        assert "SummaryStore" in r
        assert "cache_loaded=False" in r


class TestSummaryStoreSaveLoad:

    def test_save_and_load_roundtrip(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)

        summaries = (
            [make_summary("c0", i) for i in range(3)] +
            [make_summary("c1", i) for i in range(8)]
        )
        store.save_summaries(summaries)
        assert store.summaries_exist() is True
        assert store.total_summaries() == 11

        loaded = store.load_summaries()
        assert len(loaded) == 11

    def test_summary_fields_preserved(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        summary = make_summary("c1", 0)
        store.save_summaries([summary])

        loaded = store.load_summaries()
        assert loaded[0].community_id == "comm_c1_0000"
        assert loaded[0].title == "Community 0 at level c1"
        assert loaded[0].impact_rating == 7.5
        assert len(loaded[0].findings) == 1
        assert loaded[0].findings[0].summary == "Microsoft invested $10B in OpenAI."

    def test_load_summaries_by_level(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)

        summaries = (
            [make_summary("c0", i) for i in range(2)] +
            [make_summary("c1", i) for i in range(5)] +
            [make_summary("c2", i) for i in range(10)]
        )
        store.save_summaries(summaries)

        c0 = store.load_summaries_by_level("c0")
        c1 = store.load_summaries_by_level("c1")
        c2 = store.load_summaries_by_level("c2")
        c3 = store.load_summaries_by_level("c3")

        assert len(c0) == 2
        assert len(c1) == 5
        assert len(c2) == 10
        assert len(c3) == 0  # none saved at c3

    def test_load_summary_by_id(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 3)])

        result = store.load_summary_by_id("comm_c1_0003")
        assert result is not None
        assert result.community_id == "comm_c1_0003"

    def test_load_summary_by_id_not_found(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 0)])

        result = store.load_summary_by_id("nonexistent_id")
        assert result is None

    def test_load_summaries_file_not_found(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        with pytest.raises(FileNotFoundError, match="community_summaries.json"):
            store.load_summaries()

    def test_save_overwrites_and_invalidates_cache(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 0)])
        store.load_summaries()  # populate cache
        assert store._cache is not None

        store.save_summaries([make_summary("c1", 0), make_summary("c1", 1)])
        assert store._cache is None  # cache invalidated by save

        loaded = store.load_summaries()
        assert len(loaded) == 2

    def test_append_summaries(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c0", 0)])
        store.append_summaries([make_summary("c1", 0), make_summary("c1", 1)])
        store.append_summaries([make_summary("c2", 0)])

        assert store.total_summaries() == 4

    def test_append_empty_does_nothing(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 0)])
        store.append_summaries([])
        assert store.total_summaries() == 1

    def test_corrupt_summary_record_skipped(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)

        summaries_path = artifacts_dir / "community_summaries.json"
        with open(summaries_path, "w") as f:
            json.dump([
                make_summary("c1", 0).model_dump(mode="json"),
                {"community_id": "bad", "missing_required": True},
            ], f)

        loaded = store.load_summaries()
        assert len(loaded) == 1


class TestSummaryStoreCache:

    def test_second_load_uses_cache(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir, use_cache=True)
        store.save_summaries([make_summary("c1", i) for i in range(5)])

        store.load_summaries()
        assert store._cache is not None

        # Delete the file — second load must use cache
        (artifacts_dir / "community_summaries.json").unlink()
        loaded2 = store.load_summaries()
        assert len(loaded2) == 5  # from cache, not disk

    def test_no_cache_mode(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir, use_cache=False)
        store.save_summaries([make_summary("c1", 0)])

        store.load_summaries()
        assert store._cache is None  # cache disabled

    def test_invalidate_cache(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 0)])
        store.load_summaries()
        assert store._cache is not None

        store.invalidate_cache()
        assert store._cache is None
        assert store._cache_by_id == {}
        assert store._cache_by_level == {}

    def test_warm_cache(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", i) for i in range(3)])
        assert store._cache is None

        store.warm_cache()
        assert store._cache is not None
        assert len(store._cache) == 3

    def test_warm_cache_no_file(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        # Should not raise — logs a warning
        store.warm_cache()
        assert store._cache is None


class TestSummaryStorePagination:

    def test_paginate_first_page(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", i) for i in range(25)])

        page, total = store.load_summaries_paginated("c1", page=1, page_size=10)
        assert len(page) == 10
        assert total == 25

    def test_paginate_last_page(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", i) for i in range(25)])

        page, total = store.load_summaries_paginated("c1", page=3, page_size=10)
        assert len(page) == 5  # 25 - 20 = 5 remaining
        assert total == 25

    def test_paginate_nonexistent_level(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 0)])

        page, total = store.load_summaries_paginated("c3", page=1, page_size=10)
        assert len(page) == 0
        assert total == 0


class TestSummaryStoreCountsAndStats:

    def test_get_summary_counts(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries(
            [make_summary("c0", i) for i in range(3)] +
            [make_summary("c1", i) for i in range(7)]
        )
        counts = store.get_summary_counts()
        assert counts["c0"] == 3
        assert counts["c1"] == 7

    def test_get_summary_counts_from_cache(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", i) for i in range(5)])
        store.load_summaries()  # populate cache

        counts = store.get_summary_counts()
        assert counts["c1"] == 5

    def test_get_stats(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", i) for i in range(3)])

        stats = store.get_stats()
        assert stats["summaries"]["exists"] is True
        assert stats["summaries"]["total"] == 3

    def test_delete_summaries(self, artifacts_dir):
        from app.storage.summary_store import SummaryStore
        store = SummaryStore(artifacts_dir=artifacts_dir)
        store.save_summaries([make_summary("c1", 0)])
        store.load_summaries()  # populate cache

        result = store.delete_summaries()
        assert result is True
        assert not store.summaries_exist()
        assert store._cache is None  # cache cleared


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheManagerInit:

    def test_creates_directory(self, tmp_path):
        from app.storage.cache_manager import CacheManager
        new_dir = tmp_path / "new_cache_dir"
        cache = CacheManager(artifacts_dir=new_dir)
        assert new_dir.exists()

    def test_initial_state_no_file(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        assert not cache.state_exists()
        assert cache.run_id == ""

    def test_repr(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=100)
        r = repr(cache)
        assert "CacheManager" in r
        assert "100" in r


class TestCacheManagerInitializeRun:

    def test_fresh_run(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        run_id = cache.initialize_run(total_chunks=50)

        assert run_id.startswith("run_")
        assert cache.state_exists()
        assert cache.get_progress()["total_chunks"] == 50

    def test_custom_run_id(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        run_id = cache.initialize_run(total_chunks=10, run_id="custom_run_001")
        assert run_id == "custom_run_001"

    def test_resume_preserves_extracted_chunks(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager

        # Simulate partial run
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)
        cache.mark_extracted("chunk_0")
        cache.mark_extracted("chunk_1")

        # New CacheManager instance simulates restart
        cache2 = CacheManager(artifacts_dir=artifacts_dir)
        cache2.initialize_run(total_chunks=10)

        assert cache2.is_extracted("chunk_0")
        assert cache2.is_extracted("chunk_1")
        assert not cache2.is_extracted("chunk_2")

    def test_force_reset_clears_state(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)
        cache.mark_extracted("chunk_0")

        cache.initialize_run(total_chunks=10, force_reset=True)
        assert not cache.is_extracted("chunk_0")
        assert cache.get_progress()["extracted"] == 0

    def test_state_persisted_to_disk(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager, STATE_FILENAME
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)

        state_file = artifacts_dir / STATE_FILENAME
        assert state_file.exists()

        with open(state_file) as f:
            state = json.load(f)
        assert state["total_chunks"] == 5


class TestCacheManagerChunkTracking:

    def test_mark_extracted(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        assert not cache.is_extracted("chunk_0")
        cache.mark_extracted("chunk_0")
        assert cache.is_extracted("chunk_0")

    def test_mark_extracted_idempotent(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        cache.mark_extracted("chunk_0")
        cache.mark_extracted("chunk_0")  # second call should not duplicate
        cache.mark_extracted("chunk_0")

        progress = cache.get_progress()
        assert progress["extracted"] == 1

    def test_mark_failed(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        assert not cache.is_failed("chunk_5")
        cache.mark_failed("chunk_5", "Rate limit exceeded after 3 retries")
        assert cache.is_failed("chunk_5")

    def test_mark_failed_stores_error_message(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        cache.mark_failed("chunk_5", "API timeout")
        failed = cache.get_failed_chunks()
        assert "chunk_5" in failed
        assert "API timeout" in failed["chunk_5"]["error"]

    def test_mark_extracted_removes_from_failed(self, artifacts_dir):
        """A chunk that previously failed and is now re-processed should
        be moved from failed to extracted."""
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        cache.mark_failed("chunk_3", "Transient error")
        assert cache.is_failed("chunk_3")

        cache.mark_extracted("chunk_3")
        assert cache.is_extracted("chunk_3")
        assert not cache.is_failed("chunk_3")

    def test_persistence_survives_restart(self, artifacts_dir):
        """mark_extracted persists immediately — survives process restart."""
        from app.storage.cache_manager import CacheManager

        cache1 = CacheManager(artifacts_dir=artifacts_dir)
        cache1.initialize_run(total_chunks=20)
        for i in range(5):
            cache1.mark_extracted(f"chunk_{i}")

        # New instance simulates process restart
        cache2 = CacheManager(artifacts_dir=artifacts_dir)
        for i in range(5):
            assert cache2.is_extracted(f"chunk_{i}"), f"chunk_{i} not found after restart"
        assert not cache2.is_extracted("chunk_5")


class TestCacheManagerFilterPending:

    def test_filter_all_pending(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)

        chunks = [make_chunk(i) for i in range(5)]
        pending = cache.filter_pending_chunks(chunks)
        assert len(pending) == 5

    def test_filter_excludes_extracted(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        for i in range(3):
            cache.mark_extracted(f"doc_001_{i:04d}")

        chunks = [make_chunk(i) for i in range(10)]
        pending = cache.filter_pending_chunks(chunks)

        assert len(pending) == 7
        pending_ids = {c.chunk_id for c in pending}
        for i in range(3):
            assert f"doc_001_{i:04d}" not in pending_ids

    def test_filter_excludes_failed(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        cache.mark_failed("doc_001_0002", "Unrecoverable error")

        chunks = [make_chunk(i) for i in range(10)]
        pending = cache.filter_pending_chunks(chunks)

        assert len(pending) == 9
        assert "doc_001_0002" not in {c.chunk_id for c in pending}

    def test_filter_pending_chunk_ids(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)
        cache.mark_extracted("chunk_0")
        cache.mark_extracted("chunk_1")

        all_ids = [f"chunk_{i}" for i in range(5)]
        pending = cache.filter_pending_chunk_ids(all_ids)
        assert set(pending) == {"chunk_2", "chunk_3", "chunk_4"}

    def test_resume_scenario_end_to_end(self, artifacts_dir):
        """
        Full resume scenario:
          Run 1: Process 6/10 chunks, then crash.
          Run 2: Resume, only 4 remaining chunks processed.
        """
        from app.storage.cache_manager import CacheManager

        # ── Run 1 ──────────────────────────────────────────────────────────────
        cache1 = CacheManager(artifacts_dir=artifacts_dir)
        cache1.initialize_run(total_chunks=10)
        all_chunks = [make_chunk(i) for i in range(10)]

        # Process first 6
        for chunk in all_chunks[:6]:
            cache1.mark_extracted(chunk.chunk_id)

        # Simulate crash — cache1 is abandoned

        # ── Run 2 ──────────────────────────────────────────────────────────────
        cache2 = CacheManager(artifacts_dir=artifacts_dir)
        cache2.initialize_run(total_chunks=10)
        pending = cache2.filter_pending_chunks(all_chunks)

        assert len(pending) == 4
        pending_ids = {c.chunk_id for c in pending}
        for i in range(6):
            assert make_chunk(i).chunk_id not in pending_ids
        for i in range(6, 10):
            assert make_chunk(i).chunk_id in pending_ids


class TestCacheManagerStageTracking:

    def test_mark_stage_complete(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)

        assert not cache.is_stage_complete("chunking")
        cache.mark_stage_complete("chunking")
        assert cache.is_stage_complete("chunking")

    def test_get_completed_stages(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)

        cache.mark_stage_complete("chunking")
        cache.mark_stage_complete("extraction")

        completed = cache.get_completed_stages()
        assert "chunking" in completed
        assert "extraction" in completed
        assert "graph_construction" not in completed

    def test_get_next_pending_stage(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)

        assert cache.get_next_pending_stage() == "chunking"
        cache.mark_stage_complete("chunking")
        assert cache.get_next_pending_stage() == "extraction"
        cache.mark_stage_complete("extraction")
        assert cache.get_next_pending_stage() == "graph_construction"

    def test_all_stages_complete_returns_none(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager, ALL_STAGES
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)

        for stage in ALL_STAGES:
            cache.mark_stage_complete(stage)

        assert cache.get_next_pending_stage() is None

    def test_reset_stage(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)
        cache.mark_stage_complete("extraction")

        cache.reset_stage("extraction")
        assert not cache.is_stage_complete("extraction")

    def test_reset_extraction_stage_clears_chunk_state(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)
        cache.mark_extracted("chunk_0")
        cache.mark_stage_complete("extraction")

        cache.reset_stage("extraction")
        assert not cache.is_extracted("chunk_0")
        assert cache.get_progress()["extracted"] == 0

    def test_stage_completion_persists_across_restart(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache1 = CacheManager(artifacts_dir=artifacts_dir)
        cache1.initialize_run(total_chunks=5)
        cache1.mark_stage_complete("chunking")

        cache2 = CacheManager(artifacts_dir=artifacts_dir)
        assert cache2.is_stage_complete("chunking")


class TestCacheManagerProgress:

    def test_get_progress(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=100)

        for i in range(30):
            cache.mark_extracted(f"chunk_{i}")
        for i in range(5):
            cache.mark_failed(f"bad_chunk_{i}", "error")

        progress = cache.get_progress()
        assert progress["total_chunks"] == 100
        assert progress["extracted"] == 30
        assert progress["failed"] == 5
        assert progress["pending"] == 65
        assert progress["pct_complete"] == 30.0

    def test_extraction_completion_rate(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)

        for i in range(4):
            cache.mark_extracted(f"chunk_{i}")

        rate = cache.extraction_completion_rate()
        assert rate == 0.4

    def test_completion_rate_zero_when_no_chunks(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        assert cache.extraction_completion_rate() == 0.0


class TestCacheManagerReset:

    def test_reset_all(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager, STATE_FILENAME
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=10)
        cache.mark_extracted("chunk_0")
        cache.mark_stage_complete("chunking")

        cache.reset_all()

        assert not cache.state_exists()
        assert not cache.is_extracted("chunk_0")
        assert not cache.is_stage_complete("chunking")
        assert cache.run_id == ""

    def test_delete_state_file(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=5)
        assert cache.state_exists()

        result = cache.delete_state_file()
        assert result is True
        assert not cache.state_exists()

    def test_delete_state_file_not_exists(self, artifacts_dir):
        from app.storage.cache_manager import CacheManager
        cache = CacheManager(artifacts_dir=artifacts_dir)
        assert cache.delete_state_file() is False


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: ALL STORES WORKING TOGETHER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStorageIntegration:
    """
    End-to-end test simulating the full pipeline storage flow:
    chunking → extraction → graph → community → summarization → query.
    """

    def test_full_pipeline_artifact_lifecycle(self, artifacts_dir):
        """
        Simulate saving and loading every artifact type in pipeline order.
        Verifies that all stores work with the same artifacts_dir.
        """
        from app.storage.artifact_store import ArtifactStore
        from app.storage.graph_store import GraphStore
        from app.storage.summary_store import SummaryStore
        from app.storage.cache_manager import CacheManager

        artifact_store = ArtifactStore(artifacts_dir=artifacts_dir)
        graph_store    = GraphStore(artifacts_dir=artifacts_dir)
        summary_store  = SummaryStore(artifacts_dir=artifacts_dir)
        cache          = CacheManager(artifacts_dir=artifacts_dir)

        # ── Stage 1: Chunking ──────────────────────────────────────────────────
        chunks = [make_chunk(i) for i in range(10)]
        artifact_store.save_chunks(chunks)
        cache.initialize_run(total_chunks=10)
        cache.mark_stage_complete("chunking")

        assert artifact_store.chunks_exist()
        assert cache.is_stage_complete("chunking")

        # ── Stage 2: Extraction ────────────────────────────────────────────────
        for chunk in chunks:
            extraction = make_extraction(chunk.chunk_id)
            artifact_store.append_extraction(extraction)
            cache.mark_extracted(chunk.chunk_id)
        cache.mark_stage_complete("extraction")

        assert artifact_store.extractions_exist()
        assert artifact_store.extractions_count() == 10
        assert cache.is_stage_complete("extraction")
        assert cache.extraction_completion_rate() == 1.0

        # ── Stage 3: Graph construction ────────────────────────────────────────
        graph = make_nx_graph()
        graph_store.save_graph(graph)
        cache.mark_stage_complete("graph_construction")

        assert graph_store.graph_exists()
        loaded_graph = graph_store.load_graph()
        assert loaded_graph.number_of_nodes() == 3

        # ── Stage 4: Community detection ───────────────────────────────────────
        communities = (
            [make_community("c0", i) for i in range(2)] +
            [make_community("c1", i) for i in range(5)]
        )
        graph_store.save_community_map(communities)
        cache.mark_stage_complete("community_detection")

        counts = graph_store.get_community_counts()
        assert counts["c0"] == 2
        assert counts["c1"] == 5

        # ── Stage 5: Summarization ─────────────────────────────────────────────
        summaries = (
            [make_summary("c0", i) for i in range(2)] +
            [make_summary("c1", i) for i in range(5)]
        )
        summary_store.save_summaries(summaries)
        cache.mark_stage_complete("summarization")

        assert summary_store.summaries_exist()
        c1_summaries = summary_store.load_summaries_by_level("c1")
        assert len(c1_summaries) == 5

        # ── Final state ────────────────────────────────────────────────────────
        progress = cache.get_progress()
        assert progress["extracted"] == 10
        assert progress["pct_complete"] == 100.0

        completed = cache.get_completed_stages()
        assert "chunking" in completed
        assert "extraction" in completed
        assert "graph_construction" in completed
        assert "community_detection" in completed
        assert "summarization" in completed

    def test_force_reindex_wipes_all_artifacts(self, artifacts_dir):
        """force_reindex=True should wipe all stores cleanly."""
        from app.storage.artifact_store import ArtifactStore
        from app.storage.graph_store import GraphStore
        from app.storage.summary_store import SummaryStore
        from app.storage.cache_manager import CacheManager

        # Create all artifacts
        ArtifactStore(artifacts_dir=artifacts_dir).save_chunks([make_chunk(0)])
        GraphStore(artifacts_dir=artifacts_dir).save_graph(make_nx_graph())
        SummaryStore(artifacts_dir=artifacts_dir).save_summaries([make_summary()])
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=1)
        cache.mark_extracted("doc_001_0000")

        # Wipe everything
        ArtifactStore(artifacts_dir=artifacts_dir).delete_all()
        GraphStore(artifacts_dir=artifacts_dir).delete_all()
        SummaryStore(artifacts_dir=artifacts_dir).delete_all()
        cache.reset_all()

        # Verify clean state
        assert not ArtifactStore(artifacts_dir=artifacts_dir).chunks_exist()
        assert not GraphStore(artifacts_dir=artifacts_dir).graph_exists()
        assert not SummaryStore(artifacts_dir=artifacts_dir).summaries_exist()
        assert not CacheManager(artifacts_dir=artifacts_dir).state_exists()

    def test_all_files_live_in_artifacts_dir(self, artifacts_dir):
        """All storage files must be in the configured artifacts_dir — no strays."""
        from app.storage.artifact_store import ArtifactStore
        from app.storage.graph_store import GraphStore
        from app.storage.summary_store import SummaryStore
        from app.storage.cache_manager import CacheManager

        ArtifactStore(artifacts_dir=artifacts_dir).save_chunks([make_chunk(0)])
        GraphStore(artifacts_dir=artifacts_dir).save_graph(make_nx_graph())
        SummaryStore(artifacts_dir=artifacts_dir).save_summaries([make_summary()])
        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=1)

        all_files = list(artifacts_dir.iterdir())
        for f in all_files:
            assert f.parent == artifacts_dir, f"File outside artifacts_dir: {f}"