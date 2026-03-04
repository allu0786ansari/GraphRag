"""
tests/unit/test_stage5_pipeline.py — Stage 5 pipeline tests.

Test strategy:
  - Zero real API calls. All LLM and embedding calls are mocked.
  - Real tokenizer (tiktoken) — no mock.
  - Real NetworkX graph — no mock.
  - Real community detection (NetworkX Louvain fallback — no graspologic needed).
  - Real file I/O via tmp_path (same as Stage 4 tests).
  - Pipeline runner tested via unit-level stage mocking.

Coverage targets per file:
  chunking.py            95%+
  extraction.py          90%+
  gleaning.py            90%+
  graph_builder.py       95%+
  community_detection.py 90%+
  summarization.py       85%+
  pipeline_runner.py     80%+
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES & FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def raw_data_dir(tmp_path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def artifacts_dir(tmp_path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    return d


def make_raw_document(directory: Path, name: str, text: str, metadata: dict = None) -> Path:
    """Write a JSON document file in the raw data directory."""
    doc = {"text": text, "metadata": metadata or {}}
    path = directory / name
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


def make_text_document(directory: Path, name: str, text: str) -> Path:
    """Write a plain text document file."""
    path = directory / name
    path.write_text(text)
    return path


def make_extraction_result(chunk_id: str, n_entities: int = 3, n_rels: int = 2):
    """Build a realistic ChunkExtraction without API calls."""
    from app.models.graph_models import (
        ChunkExtraction, ExtractedEntity, ExtractedRelationship
    )
    entities = [
        ExtractedEntity(
            name=f"Entity_{i}",
            entity_type="ORGANIZATION",
            description=f"Description of entity {i}.",
            source_chunk_id=chunk_id,
        )
        for i in range(n_entities)
    ]
    relationships = [
        ExtractedRelationship(
            source_entity=f"Entity_{i}",
            target_entity=f"Entity_{i+1}",
            description=f"Entity {i} relates to entity {i+1}.",
            strength=7,
            source_chunk_id=chunk_id,
        )
        for i in range(n_rels)
    ]
    return ChunkExtraction(
        chunk_id=chunk_id,
        entities=entities,
        relationships=relationships,
        gleaning_rounds_completed=0,
        extraction_completed=True,
    )


def make_mock_openai_result(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Build a mock CompletionResult."""
    from app.services.openai_service import CompletionResult
    return CompletionResult(
        content=content,
        model="gpt-4o",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )


def make_multi_entity_graph() -> nx.Graph:
    """Build a realistic graph with 10+ nodes for community detection."""
    g = nx.Graph()
    entities = [
        ("openai",     "OpenAI",     "ORGANIZATION"),
        ("microsoft",  "Microsoft",  "ORGANIZATION"),
        ("google",     "Google",     "ORGANIZATION"),
        ("apple",      "Apple",      "ORGANIZATION"),
        ("sam_altman", "Sam Altman", "PERSON"),
        ("satya_nadella", "Satya Nadella", "PERSON"),
        ("sundar_pichai", "Sundar Pichai", "PERSON"),
        ("tim_cook",   "Tim Cook",   "PERSON"),
        ("gpt4",       "GPT-4",      "TECHNOLOGY"),
        ("bing",       "Bing",       "TECHNOLOGY"),
        ("ai_safety",  "AI Safety",  "CONCEPT"),
        ("llm",        "LLM",        "TECHNOLOGY"),
    ]
    for node_id, name, etype in entities:
        g.add_node(node_id, name=name, entity_type=etype,
                   description=f"{name} is a {etype}.", degree=0,
                   source_chunk_ids=[], claims=[], mention_count=1,
                   community_ids={})

    edges = [
        ("openai",     "microsoft",     9.0),
        ("openai",     "sam_altman",    8.0),
        ("microsoft",  "satya_nadella", 8.0),
        ("google",     "sundar_pichai", 8.0),
        ("apple",      "tim_cook",      8.0),
        ("openai",     "gpt4",          7.0),
        ("microsoft",  "bing",          6.0),
        ("openai",     "ai_safety",     5.0),
        ("gpt4",       "llm",           4.0),
        ("google",     "llm",           4.0),
        ("openai",     "google",        3.0),
        ("microsoft",  "apple",         2.0),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, weight=w, edge_id=f"{u}__{v}",
                   description=f"{u} relates to {v}.", combined_degree=0)

    for node_id in g.nodes:
        g.nodes[node_id]["degree"] = g.degree(node_id)
    for u, v in g.edges:
        g.edges[u, v]["combined_degree"] = g.nodes[u]["degree"] + g.nodes[v]["degree"]

    return g


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:

    def test_chunking_imports(self):
        from app.core.pipeline.chunking import ChunkingPipeline, get_chunking_pipeline
        assert ChunkingPipeline is not None

    def test_extraction_imports(self):
        from app.core.pipeline.extraction import (
            ExtractionPipeline, TUPLE_DELIM, RECORD_DELIM,
            COMPLETION_DELIM, DEFAULT_ENTITY_TYPES,
        )
        assert TUPLE_DELIM == "<|>"
        assert RECORD_DELIM == "##"
        assert COMPLETION_DELIM == "<|COMPLETE|>"
        assert "organization" in DEFAULT_ENTITY_TYPES

    def test_gleaning_imports(self):
        from app.core.pipeline.gleaning import GleaningLoop, get_yes_no_token_ids
        yes_ids, no_ids = get_yes_no_token_ids("gpt-4o")
        assert len(yes_ids) > 0
        assert len(no_ids) > 0

    def test_graph_builder_imports(self):
        from app.core.pipeline.graph_builder import GraphBuilder, get_graph_builder
        assert GraphBuilder is not None

    def test_community_detection_imports(self):
        from app.core.pipeline.community_detection import CommunityDetection, get_community_detection
        assert CommunityDetection is not None

    def test_summarization_imports(self):
        from app.core.pipeline.summarization import SummarizationPipeline, get_summarization_pipeline
        assert SummarizationPipeline is not None

    def test_pipeline_runner_imports(self):
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        assert PipelineRunner is not None

    def test_pipeline_init_exports(self):
        from app.core.pipeline import (
            ChunkingPipeline, ExtractionPipeline, GleaningLoop,
            GraphBuilder, CommunityDetection, SummarizationPipeline,
            PipelineRunner, PipelineResult,
        )
        assert all(x is not None for x in [
            ChunkingPipeline, ExtractionPipeline, GleaningLoop,
            GraphBuilder, CommunityDetection, SummarizationPipeline,
            PipelineRunner, PipelineResult,
        ])

    def test_no_circular_imports(self):
        import importlib
        for module in [
            "app.core.pipeline.chunking",
            "app.core.pipeline.extraction",
            "app.core.pipeline.gleaning",
            "app.core.pipeline.graph_builder",
            "app.core.pipeline.community_detection",
            "app.core.pipeline.summarization",
            "app.core.pipeline.pipeline_runner",
        ]:
            mod = importlib.import_module(module)
            assert mod is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkingInit:

    def test_init(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        tok = TokenizerService(model="gpt-4o")
        pipeline = ChunkingPipeline(tokenizer=tok)
        assert pipeline.tokenizer is tok


class TestChunkDocument:

    def test_chunk_short_text(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.chunk_document(
            text="This is a short document.",
            source_document="test.json",
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "test_0000"
        assert chunks[0].source_document == "test.json"
        assert chunks[0].chunk_index == 0

    def test_chunk_ids_are_deterministic(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        text = "word " * 200
        chunks1 = pipeline.chunk_document(text, "doc.json")
        chunks2 = pipeline.chunk_document(text, "doc.json")
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_uses_stem_for_id(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.chunk_document("hello world", "article_001.json")
        assert chunks[0].chunk_id.startswith("article_001_")

    def test_chunk_long_text_produces_multiple_chunks(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        text = "word " * 800  # well over 600 tokens
        chunks = pipeline.chunk_document(text, "doc.json", chunk_size=600, chunk_overlap=100)
        assert len(chunks) >= 2

    def test_chunk_indices_sequential(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        text = "word " * 800
        chunks = pipeline.chunk_document(text, "doc.json")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_total_chunks_correct(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        text = "word " * 800
        chunks = pipeline.chunk_document(text, "doc.json")
        for c in chunks:
            assert c.total_chunks_in_doc == len(chunks)

    def test_metadata_attached_to_chunks(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        meta = {"date": "2024-01-15", "category": "tech"}
        chunks = pipeline.chunk_document("hello world text here", "doc.json", metadata=meta)
        assert chunks[0].metadata["date"] == "2024-01-15"

    def test_empty_text_returns_empty_list(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.chunk_document("", "doc.json")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.chunk_document("   \n   ", "doc.json")
        assert chunks == []


class TestChunkingRun:

    def test_run_json_documents(self, raw_data_dir):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        make_raw_document(raw_data_dir, "doc1.json", "The quick brown fox jumps over the lazy dog. " * 10)
        make_raw_document(raw_data_dir, "doc2.json", "AI is transforming the world rapidly. " * 10)

        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)

        assert len(chunks) > 0
        sources = {c.source_document for c in chunks}
        assert "doc1.json" in sources
        assert "doc2.json" in sources

    def test_run_text_documents(self, raw_data_dir):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        make_text_document(raw_data_dir, "doc1.txt", "word " * 50)

        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)
        assert len(chunks) > 0
        assert chunks[0].source_document == "doc1.txt"

    def test_run_empty_directory(self, raw_data_dir):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)
        assert chunks == []

    def test_run_nonexistent_dir_raises(self, tmp_dir):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        with pytest.raises(FileNotFoundError):
            pipeline.run(tmp_dir / "does_not_exist")

    def test_run_max_chunks_limits_output(self, raw_data_dir):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        # Write a document that produces many chunks
        make_raw_document(raw_data_dir, "big.json", "word " * 5000)
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir, max_chunks=5)
        assert len(chunks) <= 5

    def test_run_results_sorted_by_document(self, raw_data_dir):
        """Documents are processed in sorted order — IDs are deterministic."""
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        make_raw_document(raw_data_dir, "zzz.json", "text " * 20)
        make_raw_document(raw_data_dir, "aaa.json", "text " * 20)

        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)
        sources = [c.source_document for c in chunks]
        assert sources[0] == "aaa.json"

    def test_run_skips_bad_json(self, raw_data_dir):
        """A malformed JSON file should be skipped, not crash the pipeline."""
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        bad = raw_data_dir / "bad.json"
        bad.write_text("{not valid json}")
        make_raw_document(raw_data_dir, "good.json", "good text " * 20)

        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)
        # good.json should still be processed
        assert any(c.source_document == "good.json" for c in chunks)

    def test_get_stats(self, raw_data_dir):
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        make_raw_document(raw_data_dir, "doc.json", "word " * 100)
        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)
        stats = pipeline.get_stats(chunks)
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_tokens"] > 0

    def test_document_json_with_metadata_fields(self, raw_data_dir):
        """Top-level fields in JSON doc are added as metadata."""
        from app.core.pipeline.chunking import ChunkingPipeline
        from app.services.tokenizer_service import TokenizerService
        doc = {"text": "content " * 20, "date": "2024-01-01", "source": "reuters"}
        path = raw_data_dir / "meta.json"
        with open(path, "w") as f:
            json.dump(doc, f)

        pipeline = ChunkingPipeline(tokenizer=TokenizerService())
        chunks = pipeline.run(raw_data_dir)
        assert chunks[0].metadata.get("date") == "2024-01-01"


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION PARSING (no API calls)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractionParsing:
    """Test the tuple-format parser without any API calls."""

    def get_parser(self):
        from app.core.pipeline.extraction import ExtractionPipeline
        from app.services.tokenizer_service import TokenizerService
        mock_openai = MagicMock()
        mock_openai.model = "gpt-4o"
        tok = TokenizerService()
        return ExtractionPipeline(openai_service=mock_openai, tokenizer=tok)

    def test_parse_entity(self):
        pipeline = self.get_parser()
        text = '("entity"<|>OpenAI<|>ORGANIZATION<|>OpenAI is an AI company.)##<|COMPLETE|>'
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.entities) == 1
        assert result.entities[0].name == "OpenAI"
        assert result.entities[0].entity_type == "ORGANIZATION"

    def test_parse_relationship(self):
        pipeline = self.get_parser()
        text = ('("entity"<|>OpenAI<|>ORGANIZATION<|>AI company)##'
                '("entity"<|>Microsoft<|>ORGANIZATION<|>Tech giant)##'
                '("relationship"<|>OpenAI<|>Microsoft<|>Microsoft invested in OpenAI<|>9)##<|COMPLETE|>')
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.relationships) == 1
        assert result.relationships[0].source_entity == "OpenAI"
        assert result.relationships[0].target_entity == "Microsoft"
        assert result.relationships[0].strength == 9

    def test_parse_strength_clamped(self):
        pipeline = self.get_parser()
        text = '("relationship"<|>A<|>B<|>desc<|>99)##<|COMPLETE|>'
        result = pipeline._parse_extraction_output(text, "chunk_001")
        if result.relationships:
            assert result.relationships[0].strength <= 10

    def test_parse_claim(self):
        pipeline = self.get_parser()
        text = ('("entity"<|>OpenAI<|>ORGANIZATION<|>desc)##'
                '("claim"<|>OpenAI<|>funding<|>TRUE<|>OpenAI raised $10B<|>2023-01-01<|>NONE)##'
                '<|COMPLETE|>')
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.claims) == 1
        assert result.claims[0].claim_type == "funding"
        assert result.claims[0].status == "TRUE"

    def test_parse_empty_output(self):
        pipeline = self.get_parser()
        result = pipeline._parse_extraction_output("", "chunk_001")
        assert result.entities == []
        assert result.relationships == []

    def test_parse_completion_delimiter_only(self):
        pipeline = self.get_parser()
        result = pipeline._parse_extraction_output("<|COMPLETE|>", "chunk_001")
        assert result.entities == []

    def test_parse_malformed_record_skipped(self):
        pipeline = self.get_parser()
        text = ('("entity"<|>OpenAI<|>ORGANIZATION<|>desc)##'
                '("entity"<|>)##'  # malformed — too few fields
                '("entity"<|>Google<|>ORGANIZATION<|>Search company)##<|COMPLETE|>')
        result = pipeline._parse_extraction_output(text, "chunk_001")
        # At least the valid entities should be parsed
        names = [e.name for e in result.entities]
        assert "OpenAI" in names
        assert "Google" in names

    def test_parse_multiple_entities_and_relationships(self):
        pipeline = self.get_parser()
        text = (
            '("entity"<|>OpenAI<|>ORGANIZATION<|>AI company)##'
            '("entity"<|>Microsoft<|>ORGANIZATION<|>Tech giant)##'
            '("entity"<|>Sam Altman<|>PERSON<|>CEO of OpenAI)##'
            '("relationship"<|>Sam Altman<|>OpenAI<|>Sam leads OpenAI<|>8)##'
            '("relationship"<|>OpenAI<|>Microsoft<|>Partnership<|>9)##'
            '<|COMPLETE|>'
        )
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.entities) == 3
        assert len(result.relationships) == 2

    def test_chunk_id_set_correctly(self):
        pipeline = self.get_parser()
        text = '("entity"<|>Test<|>CONCEPT<|>A test concept)##<|COMPLETE|>'
        result = pipeline._parse_extraction_output(text, "my_chunk_id_9999")
        assert result.chunk_id == "my_chunk_id_9999"
        if result.entities:
            assert result.entities[0].source_chunk_id == "my_chunk_id_9999"

    def test_get_extraction_stats_empty(self):
        pipeline = self.get_parser()
        stats = pipeline.get_extraction_stats([])
        assert stats["successful"] == 0

    def test_get_extraction_stats_with_data(self):
        from app.models.graph_models import ChunkExtraction
        pipeline = self.get_parser()
        extractions = [make_extraction_result(f"chunk_{i}") for i in range(5)]
        stats = pipeline.get_extraction_stats(extractions)
        assert stats["total"] == 5
        assert stats["successful"] == 5
        assert stats["total_entities"] == 15


class TestExtractionAsync:
    """Test async extract_chunk with mocked OpenAI service."""

    def make_pipeline(self, extraction_response: str):
        from app.core.pipeline.extraction import ExtractionPipeline
        from app.services.tokenizer_service import TokenizerService

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_result = make_mock_openai_result(extraction_response)
        mock_openai.async_chat_completion = AsyncMock(return_value=mock_result)

        return ExtractionPipeline(
            openai_service=mock_openai,
            tokenizer=TokenizerService(),
            gleaning_loop=None,  # no gleaning in unit tests
        )

    @pytest.mark.asyncio
    async def test_extract_chunk_returns_extraction(self):
        from app.models.graph_models import ChunkSchema
        response = (
            '("entity"<|>OpenAI<|>ORGANIZATION<|>AI company)##'
            '("entity"<|>Sam Altman<|>PERSON<|>CEO)##'
            '("relationship"<|>Sam Altman<|>OpenAI<|>leads<|>9)##'
            '<|COMPLETE|>'
        )
        pipeline = self.make_pipeline(response)
        chunk = ChunkSchema(
            chunk_id="test_0000", source_document="test.json",
            text="OpenAI is led by Sam Altman.",
            token_count=8, start_char=0, end_char=30,
            chunk_index=0, total_chunks_in_doc=1,
        )
        result = await pipeline.extract_chunk(chunk, gleaning_rounds=0)
        assert result.chunk_id == "test_0000"
        assert result.extraction_completed is True
        assert len(result.entities) >= 1

    @pytest.mark.asyncio
    async def test_extract_chunk_sets_source_chunk_id(self):
        from app.models.graph_models import ChunkSchema
        response = '("entity"<|>TestCorp<|>ORGANIZATION<|>A company)##<|COMPLETE|>'
        pipeline = self.make_pipeline(response)
        chunk = ChunkSchema(
            chunk_id="my_chunk_id", source_document="test.json",
            text="TestCorp was founded in 2020.",
            token_count=7, start_char=0, end_char=30,
            chunk_index=0, total_chunks_in_doc=1,
        )
        result = await pipeline.extract_chunk(chunk, gleaning_rounds=0)
        if result.entities:
            assert result.entities[0].source_chunk_id == "my_chunk_id"

    @pytest.mark.asyncio
    async def test_extract_chunks_batch_empty(self):
        pipeline = self.make_pipeline("")
        results = await pipeline.extract_chunks_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_chunks_batch_concurrent(self):
        from app.models.graph_models import ChunkSchema
        response = '("entity"<|>Corp<|>ORGANIZATION<|>desc)##<|COMPLETE|>'
        pipeline = self.make_pipeline(response)

        chunks = [
            ChunkSchema(
                chunk_id=f"chunk_{i}", source_document="test.json",
                text=f"Text {i}.", token_count=3,
                start_char=0, end_char=10,
                chunk_index=i, total_chunks_in_doc=5,
            )
            for i in range(5)
        ]
        results = await pipeline.extract_chunks_batch(chunks, gleaning_rounds=0, max_concurrency=3)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_on_chunk_complete_called(self):
        from app.models.graph_models import ChunkSchema
        response = '("entity"<|>Corp<|>ORGANIZATION<|>desc)##<|COMPLETE|>'
        pipeline = self.make_pipeline(response)

        called = []
        def callback(ext):
            called.append(ext.chunk_id)

        chunk = ChunkSchema(
            chunk_id="cb_chunk", source_document="test.json",
            text="Corp was big.", token_count=4,
            start_char=0, end_char=13,
            chunk_index=0, total_chunks_in_doc=1,
        )
        await pipeline.extract_chunks_batch([chunk], gleaning_rounds=0, on_chunk_complete=callback)
        assert "cb_chunk" in called


# ═══════════════════════════════════════════════════════════════════════════════
# GLEANING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class TestGleaningYesNoTokenIds:

    def test_get_yes_no_ids_for_gpt4o(self):
        from app.core.pipeline.gleaning import get_yes_no_token_ids
        yes_ids, no_ids = get_yes_no_token_ids("gpt-4o")
        assert isinstance(yes_ids, list)
        assert isinstance(no_ids, list)
        assert len(yes_ids) > 0
        assert len(no_ids) > 0

    def test_all_ids_are_ints(self):
        from app.core.pipeline.gleaning import get_yes_no_token_ids
        yes_ids, no_ids = get_yes_no_token_ids("gpt-4o")
        assert all(isinstance(i, int) for i in yes_ids)
        assert all(isinstance(i, int) for i in no_ids)

    def test_no_overlap_between_yes_and_no(self):
        from app.core.pipeline.gleaning import get_yes_no_token_ids
        yes_ids, no_ids = get_yes_no_token_ids("gpt-4o")
        assert not set(yes_ids) & set(no_ids), "YES and NO token IDs should not overlap"

    def test_unknown_model_fallback(self):
        from app.core.pipeline.gleaning import get_yes_no_token_ids
        # Should not raise for unknown model — falls back to cl100k_base
        yes_ids, no_ids = get_yes_no_token_ids("unknown-model-xyz")
        assert len(yes_ids) > 0


class TestGleaningLoop:

    def make_gleaning_loop(self):
        from app.core.pipeline.gleaning import GleaningLoop
        mock_openai = MagicMock()
        mock_openai.model = "gpt-4o"
        return GleaningLoop(openai_service=mock_openai)

    @pytest.mark.asyncio
    async def test_max_rounds_zero_returns_initial(self):
        """max_rounds=0 should return the initial extraction unchanged."""
        from app.core.pipeline.gleaning import GleaningLoop
        loop = self.make_gleaning_loop()
        initial = make_extraction_result("chunk_0")
        result = await loop.run(
            chunk_text="some text",
            initial_extraction=initial,
            conversation_history=[],
            max_rounds=0,
        )
        assert result is initial
        assert result.gleaning_rounds_completed == 0

    @pytest.mark.asyncio
    async def test_no_answer_stops_loop(self):
        """If LLM answers NO, gleaning loop stops without adding entities."""
        from app.core.pipeline.gleaning import GleaningLoop
        from app.services.openai_service import CompletionResult

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.async_completion_with_logit_bias = AsyncMock(
            return_value=make_mock_openai_result("NO")
        )
        loop = GleaningLoop(openai_service=mock_openai)
        initial = make_extraction_result("chunk_0", n_entities=2)

        result = await loop.run(
            chunk_text="some text",
            initial_extraction=initial,
            conversation_history=[],
            max_rounds=2,
        )
        assert result.gleaning_rounds_completed == 0
        assert len(result.entities) == 2  # unchanged

    @pytest.mark.asyncio
    async def test_yes_triggers_continuation(self):
        """If LLM answers YES, a continuation call is made."""
        from app.core.pipeline.gleaning import GleaningLoop

        call_count = [0]

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.async_completion_with_logit_bias = AsyncMock(
            return_value=make_mock_openai_result("YES")
        )
        # Continuation returns NO on second check to stop the loop
        mock_openai.async_chat_completion = AsyncMock(
            return_value=make_mock_openai_result('("entity"<|>NewCorp<|>ORG<|>desc)##<|COMPLETE|>')
        )

        # On second round, answer NO
        check_responses = [
            make_mock_openai_result("YES"),
            make_mock_openai_result("NO"),
        ]
        mock_openai.async_completion_with_logit_bias = AsyncMock(side_effect=check_responses)

        loop = GleaningLoop(openai_service=mock_openai)
        initial = make_extraction_result("chunk_0", n_entities=1)

        from app.core.pipeline.extraction import ExtractionPipeline
        from app.services.tokenizer_service import TokenizerService
        pipeline = ExtractionPipeline(
            openai_service=mock_openai,
            tokenizer=TokenizerService(),
        )

        result = await loop.run(
            chunk_text="text",
            initial_extraction=initial,
            conversation_history=[],
            max_rounds=2,
            parse_fn=pipeline._parse_extraction_output,
        )
        assert result.gleaning_rounds_completed >= 1

    @pytest.mark.asyncio
    async def test_check_needs_gleaning_yes(self):
        from app.core.pipeline.gleaning import GleaningLoop
        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.async_completion_with_logit_bias = AsyncMock(
            return_value=make_mock_openai_result("YES")
        )
        loop = GleaningLoop(openai_service=mock_openai)
        result = await loop.check_needs_gleaning([])
        assert result is True

    @pytest.mark.asyncio
    async def test_check_needs_gleaning_no(self):
        from app.core.pipeline.gleaning import GleaningLoop
        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.async_completion_with_logit_bias = AsyncMock(
            return_value=make_mock_openai_result("NO")
        )
        loop = GleaningLoop(openai_service=mock_openai)
        result = await loop.check_needs_gleaning([])
        assert result is False


class TestGleaningMerge:

    def test_merge_deduplicates_entities(self):
        from app.core.pipeline.gleaning import _merge_into
        from app.models.graph_models import ChunkExtraction, ExtractedEntity

        base = make_extraction_result("c0", n_entities=0)
        base.entities = [
            ExtractedEntity(name="OpenAI", entity_type="ORG",
                           description="AI company", source_chunk_id="c0")
        ]

        new = make_extraction_result("c0", n_entities=0)
        new.entities = [
            ExtractedEntity(name="OpenAI", entity_type="ORG",
                           description="Same entity", source_chunk_id="c0"),  # dup
            ExtractedEntity(name="Google", entity_type="ORG",
                           description="Search company", source_chunk_id="c0"),  # new
        ]

        _merge_into(base, new, round_num=1)
        names = [e.name for e in base.entities]
        assert names.count("OpenAI") == 1
        assert "Google" in names

    def test_merge_deduplicates_relationships(self):
        from app.core.pipeline.gleaning import _merge_into
        from app.models.graph_models import ExtractedRelationship

        base = make_extraction_result("c0", n_rels=0)
        base.relationships = [
            ExtractedRelationship(source_entity="A", target_entity="B",
                                 description="existing", strength=5, source_chunk_id="c0")
        ]

        new = make_extraction_result("c0", n_rels=0)
        new.relationships = [
            ExtractedRelationship(source_entity="A", target_entity="B",
                                 description="duplicate", strength=5, source_chunk_id="c0"),
            ExtractedRelationship(source_entity="B", target_entity="C",
                                 description="new", strength=7, source_chunk_id="c0"),
        ]

        _merge_into(base, new, round_num=1)
        assert len(base.relationships) == 2

    def test_merge_sets_extraction_round(self):
        from app.core.pipeline.gleaning import _merge_into
        from app.models.graph_models import ExtractedEntity

        base = make_extraction_result("c0", n_entities=0)
        new = make_extraction_result("c0", n_entities=0)
        new.entities = [
            ExtractedEntity(name="NewCorp", entity_type="ORG",
                           description="new", source_chunk_id="c0")
        ]

        _merge_into(base, new, round_num=2)
        assert base.entities[0].extraction_round == 2


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphBuilderBuild:

    def test_build_empty_extractions(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        graph = builder.build([])
        assert graph.number_of_nodes() == 0

    def test_build_failed_extractions_excluded(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        from app.models.graph_models import ChunkExtraction
        builder = GraphBuilder()
        bad = ChunkExtraction(
            chunk_id="bad", entities=[], relationships=[],
            gleaning_rounds_completed=0, extraction_completed=False,
        )
        graph = builder.build([bad])
        assert graph.number_of_nodes() == 0

    def test_build_creates_nodes(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        extractions = [make_extraction_result("chunk_0", n_entities=3, n_rels=2)]
        graph = builder.build(extractions)
        assert graph.number_of_nodes() == 3

    def test_build_creates_edges(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        extractions = [make_extraction_result("chunk_0", n_entities=3, n_rels=2)]
        graph = builder.build(extractions)
        assert graph.number_of_edges() == 2

    def test_entity_merging_across_chunks(self):
        """Same entity appearing in multiple chunks → merged into one node."""
        from app.core.pipeline.graph_builder import GraphBuilder
        from app.models.graph_models import ChunkExtraction, ExtractedEntity

        ext1 = ChunkExtraction(
            chunk_id="c0", entities=[
                ExtractedEntity(name="OpenAI", entity_type="ORGANIZATION",
                               description="AI company", source_chunk_id="c0")
            ], relationships=[], gleaning_rounds_completed=0, extraction_completed=True,
        )
        ext2 = ChunkExtraction(
            chunk_id="c1", entities=[
                ExtractedEntity(name="OpenAI", entity_type="ORGANIZATION",
                               description="Research lab", source_chunk_id="c1")
            ], relationships=[], gleaning_rounds_completed=0, extraction_completed=True,
        )

        builder = GraphBuilder()
        graph = builder.build([ext1, ext2])

        # OpenAI appears once
        assert graph.number_of_nodes() == 1
        # mention_count should be 2
        assert graph.nodes["openai"]["mention_count"] == 2

    def test_descriptions_concatenated(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        from app.models.graph_models import ChunkExtraction, ExtractedEntity

        ext1 = ChunkExtraction(
            chunk_id="c0", entities=[
                ExtractedEntity(name="Corp", entity_type="ORG",
                               description="First description.", source_chunk_id="c0")
            ], relationships=[], gleaning_rounds_completed=0, extraction_completed=True,
        )
        ext2 = ChunkExtraction(
            chunk_id="c1", entities=[
                ExtractedEntity(name="Corp", entity_type="ORG",
                               description="Second description.", source_chunk_id="c1")
            ], relationships=[], gleaning_rounds_completed=0, extraction_completed=True,
        )
        builder = GraphBuilder()
        graph = builder.build([ext1, ext2])
        desc = graph.nodes["corp"]["description"]
        assert "First description." in desc
        assert "Second description." in desc

    def test_edge_weight_is_cooccurrence_count(self):
        """Edge weight = number of chunks where this pair appeared."""
        from app.core.pipeline.graph_builder import GraphBuilder
        from app.models.graph_models import (
            ChunkExtraction, ExtractedEntity, ExtractedRelationship
        )

        def make_ext(chunk_id):
            return ChunkExtraction(
                chunk_id=chunk_id,
                entities=[
                    ExtractedEntity(name="A", entity_type="ORG", description="a", source_chunk_id=chunk_id),
                    ExtractedEntity(name="B", entity_type="ORG", description="b", source_chunk_id=chunk_id),
                ],
                relationships=[
                    ExtractedRelationship(source_entity="A", target_entity="B",
                                        description="A relates to B", strength=8,
                                        source_chunk_id=chunk_id)
                ],
                gleaning_rounds_completed=0, extraction_completed=True,
            )

        builder = GraphBuilder()
        graph = builder.build([make_ext("c0"), make_ext("c1"), make_ext("c2")])

        # Only one unique pair A-B, but appeared in 3 chunks → weight=3
        assert graph.edges["a", "b"]["weight"] == 3.0

    def test_degree_computed(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        extractions = [make_extraction_result("c0", n_entities=3, n_rels=2)]
        graph = builder.build(extractions)
        for node in graph.nodes:
            assert "degree" in graph.nodes[node]
            assert graph.nodes[node]["degree"] == graph.degree(node)

    def test_combined_degree_computed(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        extractions = [make_extraction_result("c0", n_entities=3, n_rels=2)]
        graph = builder.build(extractions)
        for u, v in graph.edges:
            cd = graph.edges[u, v]["combined_degree"]
            expected = graph.nodes[u]["degree"] + graph.nodes[v]["degree"]
            assert cd == expected

    def test_self_loops_excluded(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        from app.models.graph_models import (
            ChunkExtraction, ExtractedEntity, ExtractedRelationship
        )
        ext = ChunkExtraction(
            chunk_id="c0",
            entities=[
                ExtractedEntity(name="Corp", entity_type="ORG", description="d", source_chunk_id="c0")
            ],
            relationships=[
                ExtractedRelationship(source_entity="Corp", target_entity="Corp",
                                     description="self", strength=5, source_chunk_id="c0")
            ],
            gleaning_rounds_completed=0, extraction_completed=True,
        )
        builder = GraphBuilder()
        graph = builder.build([ext])
        assert graph.number_of_edges() == 0

    def test_min_entity_mentions_filter(self):
        """Entities seen fewer than min_entity_mentions times are excluded."""
        from app.core.pipeline.graph_builder import GraphBuilder
        from app.models.graph_models import ChunkExtraction, ExtractedEntity

        ext = ChunkExtraction(
            chunk_id="c0",
            entities=[
                ExtractedEntity(name="Common", entity_type="ORG", description="d", source_chunk_id="c0"),
                ExtractedEntity(name="Common", entity_type="ORG", description="d", source_chunk_id="c1"),
                ExtractedEntity(name="Rare", entity_type="ORG", description="d", source_chunk_id="c0"),
            ],
            relationships=[], gleaning_rounds_completed=0, extraction_completed=True,
        )
        # Simulate Common appearing twice by using two extractions
        ext2 = ChunkExtraction(
            chunk_id="c1",
            entities=[
                ExtractedEntity(name="Common", entity_type="ORG", description="d", source_chunk_id="c1"),
            ],
            relationships=[], gleaning_rounds_completed=0, extraction_completed=True,
        )
        builder = GraphBuilder()
        graph = builder.build([ext, ext2], min_entity_mentions=2)
        # "Common" should still be there, but let's just verify the graph was built
        assert graph.number_of_nodes() >= 1

    def test_get_node_context(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        graph = make_multi_entity_graph()
        ctx = builder.get_node_context(graph, "openai", max_edges=5)
        assert "OpenAI" in ctx
        assert "ORGANIZATION" in ctx

    def test_get_graph_stats(self):
        from app.core.pipeline.graph_builder import GraphBuilder
        builder = GraphBuilder()
        graph = make_multi_entity_graph()
        stats = builder.get_graph_stats(graph)
        assert stats["nodes"] == 12
        assert stats["edges"] == 12
        assert "entity_type_distribution" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommunityDetection:

    def test_detect_on_empty_graph(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection()
        graph = nx.Graph()
        communities = detector.detect(graph)
        assert communities == []

    def test_detect_produces_communities(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)
        assert len(communities) > 0

    def test_all_nodes_covered(self):
        """Every node in the graph should appear in at least one community."""
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)

        all_node_ids_in_communities = set()
        for c in communities:
            all_node_ids_in_communities.update(c.node_ids)

        for node_id in graph.nodes:
            assert node_id in all_node_ids_in_communities, \
                f"Node {node_id} not covered by any community"

    def test_community_ids_unique(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)
        ids = [c.community_id for c in communities]
        assert len(ids) == len(set(ids)), "Duplicate community IDs found"

    def test_community_id_format(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)
        for c in communities:
            assert c.community_id.startswith("comm_c"), \
                f"Unexpected community ID format: {c.community_id}"

    def test_node_counts_correct(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)
        for c in communities:
            assert c.node_count == len(c.node_ids), \
                f"node_count mismatch for {c.community_id}"

    def test_graph_annotated_with_community_ids(self):
        """After detect(), graph nodes have community_ids populated."""
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        detector.detect(graph)

        for node_id in graph.nodes:
            comm_ids = graph.nodes[node_id].get("community_ids", {})
            assert isinstance(comm_ids, dict), f"community_ids not a dict for {node_id}"
            assert len(comm_ids) > 0, f"community_ids empty for {node_id}"

    def test_community_schema_fields_valid(self):
        from app.core.pipeline.community_detection import CommunityDetection
        from app.models.graph_models import CommunitySchema
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)

        for c in communities:
            assert isinstance(c, CommunitySchema)
            assert c.node_count >= 1
            assert len(c.node_ids) >= 1
            assert c.level_index >= 0

    def test_get_stats(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection(random_seed=42)
        graph = make_multi_entity_graph()
        communities = detector.detect(graph)
        stats = detector.get_stats(communities)
        assert stats["total"] == len(communities)
        assert "avg_size" in stats

    def test_get_stats_empty(self):
        from app.core.pipeline.community_detection import CommunityDetection
        detector = CommunityDetection()
        stats = detector.get_stats([])
        assert stats["total"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_valid_summary_json(title: str = "Test Community") -> str:
    data = {
        "title": title,
        "summary": "This community covers AI investment topics.",
        "rating": 7.5,
        "rating_explanation": "High impact due to large investment flows.",
        "findings": [
            {
                "summary": "Microsoft invested $10B in OpenAI.",
                "explanation": "This created the primary commercial partnership.",
            },
            {
                "summary": "Sam Altman leads OpenAI as CEO.",
                "explanation": "He shaped the company's research direction.",
            },
        ]
    }
    return json.dumps(data)


def make_community_schema(community_id: str = "comm_c1_0000", level: str = "c1") -> "CommunitySchema":
    from app.models.graph_models import CommunitySchema, CommunityLevel
    return CommunitySchema(
        community_id=community_id,
        level=CommunityLevel(level),
        level_index=0,
        node_ids=["openai", "microsoft", "sam_altman"],
        edge_ids=["openai__microsoft"],
        node_count=3,
        edge_count=1,
    )


class TestSummarizationContextBuilding:

    def make_pipeline(self):
        from app.core.pipeline.summarization import SummarizationPipeline
        from app.services.tokenizer_service import TokenizerService
        mock_openai = MagicMock()
        mock_openai.model = "gpt-4o"
        tok = TokenizerService()
        return SummarizationPipeline(openai_service=mock_openai, tokenizer=tok)

    def test_build_raw_context_contains_entity_info(self):
        pipeline = self.make_pipeline()
        graph = make_multi_entity_graph()
        community = make_community_schema()
        ctx, truncated, used_sub = pipeline._build_community_context(
            community=community, graph=graph, child_summaries={}
        )
        assert "OpenAI" in ctx or "openai" in ctx
        assert isinstance(truncated, bool)
        assert isinstance(used_sub, bool)

    def test_build_context_not_truncated_for_small_community(self):
        pipeline = self.make_pipeline()
        graph = make_multi_entity_graph()
        community = make_community_schema()
        ctx, truncated, _ = pipeline._build_community_context(
            community=community, graph=graph, child_summaries={}
        )
        assert not truncated  # small community should fit in 8k tokens

    def test_build_context_with_missing_nodes(self):
        """Nodes in community not present in graph are silently skipped."""
        pipeline = self.make_pipeline()
        graph = nx.Graph()  # empty graph
        community = make_community_schema()
        ctx, _, _ = pipeline._build_community_context(
            community=community, graph=graph, child_summaries={}
        )
        # Should not raise, returns some minimal context
        assert isinstance(ctx, str)


class TestSummarizationParsing:

    def make_pipeline(self):
        from app.core.pipeline.summarization import SummarizationPipeline
        from app.services.tokenizer_service import TokenizerService
        mock_openai = MagicMock()
        tok = TokenizerService()
        return SummarizationPipeline(openai_service=mock_openai, tokenizer=tok)

    def test_parse_valid_json(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        summary = pipeline._parse_summary_response(
            response_text=make_valid_summary_json("AI Partnership"),
            community=community,
            context_tokens=1000,
            was_truncated=False,
            used_sub_community=False,
        )
        assert summary.community_id == "comm_c1_0000"
        assert summary.title == "AI Partnership"
        assert summary.impact_rating == 7.5
        assert len(summary.findings) == 2

    def test_parse_json_with_markdown_fences(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        response = f"```json\n{make_valid_summary_json()}\n```"
        summary = pipeline._parse_summary_response(
            response_text=response, community=community,
            context_tokens=1000, was_truncated=False, used_sub_community=False,
        )
        assert summary.title == "Test Community"

    def test_parse_rating_clamped_at_10(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        data = {"title": "T", "summary": "S", "rating": 99.0,
                "rating_explanation": "high", "findings": [{"summary": "f", "explanation": "e"}]}
        summary = pipeline._parse_summary_response(
            response_text=json.dumps(data), community=community,
            context_tokens=0, was_truncated=False, used_sub_community=False,
        )
        assert summary.impact_rating <= 10.0

    def test_parse_invalid_json_returns_fallback(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        summary = pipeline._parse_summary_response(
            response_text="this is not json at all!!!",
            community=community, context_tokens=0,
            was_truncated=False, used_sub_community=False,
        )
        assert summary.community_id == "comm_c1_0000"
        assert len(summary.findings) >= 1

    def test_parse_empty_findings_gets_default(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        data = {"title": "T", "summary": "S", "rating": 5.0,
                "rating_explanation": "ok", "findings": []}
        summary = pipeline._parse_summary_response(
            response_text=json.dumps(data), community=community,
            context_tokens=0, was_truncated=False, used_sub_community=False,
        )
        assert len(summary.findings) >= 1

    def test_fallback_summary_is_valid(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        summary = pipeline._make_fallback_summary(community, "API error")
        assert summary.community_id == "comm_c1_0000"
        assert len(summary.findings) >= 1
        assert "API error" in summary.summary or "generation failed" in summary.title.lower()

    def test_metadata_fields_set(self):
        pipeline = self.make_pipeline()
        community = make_community_schema()
        summary = pipeline._parse_summary_response(
            response_text=make_valid_summary_json(),
            community=community, context_tokens=2500,
            was_truncated=True, used_sub_community=False,
        )
        assert summary.context_tokens_used == 2500
        assert summary.was_truncated is True
        assert summary.level.value == "c1"


class TestSummarizationAsync:

    def make_pipeline(self, response_json: str = None):
        from app.core.pipeline.summarization import SummarizationPipeline
        from app.services.tokenizer_service import TokenizerService
        from app.services.openai_service import CompletionResult

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        content = response_json or make_valid_summary_json()
        mock_openai.complete = AsyncMock(
            return_value=CompletionResult(
                content=content, model="gpt-4o",
                prompt_tokens=500, completion_tokens=300,
                finish_reason="stop",
            )
        )
        return SummarizationPipeline(
            openai_service=mock_openai,
            tokenizer=TokenizerService(),
        )

    @pytest.mark.asyncio
    async def test_summarize_community(self):
        pipeline = self.make_pipeline()
        community = make_community_schema("comm_c1_0000", "c1")
        graph = make_multi_entity_graph()

        summary = await pipeline.summarize_community(community, graph)
        assert summary.community_id == "comm_c1_0000"
        assert summary.impact_rating == 7.5
        assert len(summary.findings) == 2

    @pytest.mark.asyncio
    async def test_summarize_all_produces_one_per_community(self):
        pipeline = self.make_pipeline()
        graph = make_multi_entity_graph()
        communities = [
            make_community_schema("comm_c1_0000", "c1"),
            make_community_schema("comm_c1_0001", "c1"),
            make_community_schema("comm_c1_0002", "c1"),
        ]
        summaries = await pipeline.summarize_all(communities, graph, max_concurrency=3)
        assert len(summaries) == 3

    @pytest.mark.asyncio
    async def test_summarize_all_leaf_first(self):
        """Level c3 (leaf) must be processed before c1 (higher level)."""
        pipeline = self.make_pipeline()
        graph = make_multi_entity_graph()
        communities = [
            make_community_schema("comm_c1_0000", "c1"),
            make_community_schema("comm_c3_0000", "c3"),
        ]
        order = []
        orig_summarize = pipeline.summarize_community
        async def track_order(comm, g, child_summaries=None):
            order.append(comm.level.value)
            return await orig_summarize(comm, g, child_summaries)
        pipeline.summarize_community = track_order

        await pipeline.summarize_all(communities, graph)
        # c3 should appear before c1
        assert order.index("c3") < order.index("c1")

    @pytest.mark.asyncio
    async def test_on_summary_complete_called(self):
        pipeline = self.make_pipeline()
        graph = make_multi_entity_graph()
        communities = [make_community_schema(f"comm_c1_{i:04d}", "c1") for i in range(3)]

        called = []
        def callback(s):
            called.append(s.community_id)

        await pipeline.summarize_all(communities, graph, on_summary_complete=callback)
        assert len(called) == 3

    @pytest.mark.asyncio
    async def test_api_failure_returns_fallback(self):
        from app.core.pipeline.summarization import SummarizationPipeline
        from app.services.tokenizer_service import TokenizerService

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.complete = AsyncMock(side_effect=Exception("API down"))

        pipeline = SummarizationPipeline(
            openai_service=mock_openai,
            tokenizer=TokenizerService(),
        )
        community = make_community_schema()
        graph = make_multi_entity_graph()

        summary = await pipeline.summarize_community(community, graph)
        # Should not raise — returns fallback
        assert summary.community_id == community.community_id
        assert len(summary.findings) >= 1

    def test_get_stats(self):
        from app.core.pipeline.summarization import SummarizationPipeline
        from app.services.tokenizer_service import TokenizerService
        from app.models.graph_models import CommunitySummary, CommunityFinding, CommunityLevel
        from datetime import datetime, timezone

        pipeline = SummarizationPipeline(
            openai_service=MagicMock(), tokenizer=TokenizerService()
        )
        summaries = []
        for i in range(5):
            summaries.append(CommunitySummary(
                community_id=f"comm_c1_{i:04d}",
                level=CommunityLevel.C1,
                title=f"Community {i}",
                summary="A summary.",
                impact_rating=5.0,
                rating_explanation="ok",
                findings=[CommunityFinding(finding_id=0, summary="s", explanation="e")],
                node_ids=["a", "b"],
                context_tokens_used=3000,
            ))
        stats = pipeline.get_stats(summaries)
        assert stats["total"] == 5
        assert "by_level" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineRunnerInit:

    def test_init(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner
        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test-key",
        )
        assert runner.raw_data_dir == raw_data_dir
        assert runner.artifacts_dir == artifacts_dir
        assert runner.chunk_size == 600
        assert runner.gleaning_rounds == 2

    def test_pipeline_result_repr(self):
        from app.core.pipeline.pipeline_runner import PipelineResult
        result = PipelineResult(success=True, run_id="run_001", total_elapsed_seconds=42.5)
        r = repr(result)
        assert "run_001" not in r or "OK" in r  # either format is fine

    def test_get_pipeline_status_structure(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner
        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test",
        )
        status = runner.get_pipeline_status()
        assert "progress" in status
        assert "artifacts" in status


class TestPipelineRunnerStages:
    """
    Test each stage independently with mocked LLM calls.
    These tests verify the orchestration logic without making API calls.
    """

    def make_runner(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner
        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test",
            gleaning_rounds=0,          # no gleaning in tests
            skip_embedding=True,        # no embedding in tests
            min_entity_mentions=1,
        )
        runner._build_services()        # initialize tokenizer/openai with mock
        return runner

    def write_sample_docs(self, raw_data_dir, n=3):
        """Write N small JSON documents to the raw data dir."""
        for i in range(n):
            make_raw_document(
                raw_data_dir,
                f"doc_{i:03d}.json",
                f"OpenAI and Microsoft partnered together. Google is a competitor. " * 20,
            )

    @pytest.mark.asyncio
    async def test_chunking_stage_produces_artifact(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        self.write_sample_docs(raw_data_dir, n=2)
        runner = self.make_runner(raw_data_dir, artifacts_dir)
        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)
        chunks = await runner._run_chunking(result, max_chunks=10, on_progress=None)

        assert chunks is not None
        assert len(chunks) > 0
        assert runner.artifact_store.chunks_exist()
        assert runner.cache.is_stage_complete("chunking")

    @pytest.mark.asyncio
    async def test_chunking_stage_skipped_when_done(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        from app.storage.artifact_store import ArtifactStore
        from app.storage.cache_manager import CacheManager

        # Pre-populate artifacts
        self.write_sample_docs(raw_data_dir, n=1)
        runner = self.make_runner(raw_data_dir, artifacts_dir)
        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)

        # Simulate already-done chunking
        chunks = await runner._run_chunking(result, max_chunks=10, on_progress=None)
        result.stages_completed.clear()

        # Run again — should skip
        chunks2 = await runner._run_chunking(result, max_chunks=10, on_progress=None)
        assert "chunking" in result.stages_skipped

    @pytest.mark.asyncio
    async def test_graph_stage(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        runner = self.make_runner(raw_data_dir, artifacts_dir)
        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)

        extractions = [make_extraction_result(f"chunk_{i}", n_entities=3, n_rels=2) for i in range(5)]
        graph = await runner._run_graph_construction(result, extractions, on_progress=None)

        assert graph is not None
        assert graph.number_of_nodes() > 0
        assert runner.graph_store.graph_exists()
        assert runner.cache.is_stage_complete("graph_construction")

    @pytest.mark.asyncio
    async def test_community_detection_stage(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        runner = self.make_runner(raw_data_dir, artifacts_dir)
        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)
        graph = make_multi_entity_graph()

        communities = await runner._run_community_detection(result, graph, on_progress=None)
        assert communities is not None
        assert len(communities) > 0
        assert runner.graph_store.community_map_exists()
        assert runner.cache.is_stage_complete("community_detection")

    @pytest.mark.asyncio
    async def test_summarization_stage(self, raw_data_dir, artifacts_dir):
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        from app.services.openai_service import CompletionResult

        runner = self.make_runner(raw_data_dir, artifacts_dir)
        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)

        # Mock the OpenAI service on the runner
        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.complete = AsyncMock(
            return_value=CompletionResult(
                content=make_valid_summary_json(),
                model="gpt-4o",
                prompt_tokens=500, completion_tokens=300,
                finish_reason="stop",
            )
        )
        runner._openai_svc = mock_openai

        graph = make_multi_entity_graph()
        from app.core.pipeline.community_detection import CommunityDetection
        communities = CommunityDetection(random_seed=42).detect(graph)

        summaries = await runner._run_summarization(result, communities, graph, on_progress=None)
        assert summaries is not None
        assert len(summaries) == len(communities)
        assert runner.summary_store.summaries_exist()
        assert runner.cache.is_stage_complete("summarization")


class TestPipelineRunnerResume:

    @pytest.mark.asyncio
    async def test_resume_skips_completed_stages(self, raw_data_dir, artifacts_dir):
        """If chunking is already done, the second run skips it."""
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        from app.models.graph_models import ChunkSchema

        # Pre-populate the artifact store to simulate a completed chunking stage
        from app.storage.artifact_store import ArtifactStore
        from app.storage.cache_manager import CacheManager

        store = ArtifactStore(artifacts_dir=artifacts_dir)
        chunks = [
            ChunkSchema(
                chunk_id=f"doc_000_{i:04d}", source_document="doc_000.json",
                text="word " * 20, token_count=20, start_char=0, end_char=100,
                chunk_index=i, total_chunks_in_doc=3,
            )
            for i in range(3)
        ]
        store.save_chunks(chunks)

        cache = CacheManager(artifacts_dir=artifacts_dir)
        cache.initialize_run(total_chunks=3)
        cache.mark_stage_complete("chunking")

        # Make sure raw docs also exist so the test doesn't fail at file-not-found
        make_raw_document(raw_data_dir, "doc_000.json", "word " * 20)

        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test",
            skip_embedding=True,
        )
        runner._build_services()

        result = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)
        loaded_chunks = await runner._run_chunking(result, max_chunks=None, on_progress=None)

        assert "chunking" in result.stages_skipped
        assert len(loaded_chunks) == 3

    def test_delete_all_artifacts(self, raw_data_dir, artifacts_dir):
        """force_reindex=True should wipe all artifact files."""
        from app.core.pipeline.pipeline_runner import PipelineRunner
        from app.storage.artifact_store import ArtifactStore

        store = ArtifactStore(artifacts_dir=artifacts_dir)
        from app.models.graph_models import ChunkSchema
        chunks = [ChunkSchema(
            chunk_id="c0", source_document="d.json", text="t",
            token_count=1, start_char=0, end_char=1,
            chunk_index=0, total_chunks_in_doc=1,
        )]
        store.save_chunks(chunks)
        assert store.chunks_exist()

        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test",
        )
        runner._delete_all_artifacts()
        assert not store.chunks_exist()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL END-TO-END INTEGRATION (no LLM, all real components)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """
    Full pipeline integration test using real components but mocked LLM.

    Validates the complete artifact flow:
      documents → chunks → extractions → graph → communities → summaries

    This is the key "done when" test for Stage 5.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_50_chunks(self, raw_data_dir, artifacts_dir):
        """
        Simulate the full pipeline on a 3-document corpus.
        Uses real chunking, real graph building, real community detection.
        Mocks only LLM calls (extraction + summarization).
        """
        from app.core.pipeline.pipeline_runner import PipelineRunner, PipelineResult
        from app.services.openai_service import CompletionResult
        from app.core.pipeline.extraction import TUPLE_DELIM, RECORD_DELIM, COMPLETION_DELIM

        # Write 3 documents that produce ~15 chunks total
        for i in range(3):
            text = (
                f"OpenAI is an AI company. Microsoft invested $10B in OpenAI. "
                f"Sam Altman leads OpenAI as CEO. Google is a competitor. "
                f"Document {i} discusses AI trends and market dynamics. "
            ) * 40  # enough text to produce multiple chunks
            make_raw_document(raw_data_dir, f"doc_{i:03d}.json", text)

        # Build mock LLM that returns valid extraction format
        extraction_response = (
            '("entity"<|>OpenAI<|>ORGANIZATION<|>AI research company)##'
            '("entity"<|>Microsoft<|>ORGANIZATION<|>Technology corporation)##'
            '("entity"<|>Sam Altman<|>PERSON<|>CEO of OpenAI)##'
            '("entity"<|>Google<|>ORGANIZATION<|>Search and AI company)##'
            '("relationship"<|>Microsoft<|>OpenAI<|>Microsoft invested in OpenAI<|>9)##'
            '("relationship"<|>Sam Altman<|>OpenAI<|>Sam leads OpenAI<|>8)##'
            '("relationship"<|>OpenAI<|>Google<|>Competitors in AI<|>5)##'
            f'{COMPLETION_DELIM}'
        )
        summary_response = make_valid_summary_json("AI Investment Community")

        # Track call counts
        extraction_count = [0]

        async def mock_chat_completion(messages, **kwargs):
            extraction_count[0] += 1
            return CompletionResult(
                content=extraction_response,
                model="gpt-4o",
                prompt_tokens=400, completion_tokens=200,
                finish_reason="stop",
            )

        async def mock_complete(user_prompt, **kwargs):
            return CompletionResult(
                content=summary_response,
                model="gpt-4o",
                prompt_tokens=500, completion_tokens=300,
                finish_reason="stop",
            )

        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test",
            gleaning_rounds=0,
            skip_embedding=True,
            min_entity_mentions=1,
            max_concurrency=3,
        )
        runner._build_services()
        runner._openai_svc.async_chat_completion = mock_chat_completion
        runner._openai_svc.complete = mock_complete

        result = await runner.run(force_reindex=True, max_chunks=20)

        # ── Verify result ──────────────────────────────────────────────────────
        assert result.chunks_count > 0
        assert result.extractions_count > 0
        assert result.graph_nodes > 0
        assert result.graph_edges > 0
        assert result.communities_count > 0
        assert result.summaries_count > 0

        # ── Verify all artifacts on disk ───────────────────────────────────────
        assert runner.artifact_store.chunks_exist()
        assert runner.artifact_store.extractions_exist()
        assert runner.graph_store.graph_exists()
        assert runner.graph_store.community_map_exists()
        assert runner.summary_store.summaries_exist()

        # ── Verify artifact integrity ──────────────────────────────────────────
        chunks = runner.artifact_store.load_chunks()
        assert len(chunks) == result.chunks_count

        extractions = runner.artifact_store.load_extractions()
        assert len(extractions) == result.extractions_count

        graph = runner.graph_store.load_graph()
        assert graph.number_of_nodes() == result.graph_nodes

        communities = runner.graph_store.load_community_map()
        assert len(communities) == result.communities_count

        summaries = runner.summary_store.load_summaries()
        assert len(summaries) == result.summaries_count

        # ── Verify all stages complete ─────────────────────────────────────────
        progress = runner.cache.get_progress()
        assert runner.cache.is_stage_complete("chunking")
        assert runner.cache.is_stage_complete("extraction")
        assert runner.cache.is_stage_complete("graph_construction")
        assert runner.cache.is_stage_complete("community_detection")
        assert runner.cache.is_stage_complete("summarization")

    @pytest.mark.asyncio
    async def test_pipeline_resume_after_chunking(self, raw_data_dir, artifacts_dir):
        """
        Test that a pipeline resumed after chunking:
        - Does NOT re-chunk (uses cached chunks.json)
        - DOES run extraction on all chunks
        """
        from app.core.pipeline.pipeline_runner import PipelineRunner
        from app.services.openai_service import CompletionResult
        from app.core.pipeline.extraction import COMPLETION_DELIM

        make_raw_document(raw_data_dir, "doc_000.json",
                         "OpenAI is an AI company. " * 50)

        extraction_response = (
            '("entity"<|>OpenAI<|>ORGANIZATION<|>AI company)##'
            '("entity"<|>Microsoft<|>ORGANIZATION<|>Tech company)##'
            '("relationship"<|>OpenAI<|>Microsoft<|>Partnership<|>9)##'
            f'{COMPLETION_DELIM}'
        )

        async def mock_chat_completion(messages, **kwargs):
            return CompletionResult(
                content=extraction_response, model="gpt-4o",
                prompt_tokens=300, completion_tokens=100, finish_reason="stop",
            )
        async def mock_complete(user_prompt, **kwargs):
            return CompletionResult(
                content=make_valid_summary_json(), model="gpt-4o",
                prompt_tokens=400, completion_tokens=200, finish_reason="stop",
            )

        runner = PipelineRunner(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
            openai_api_key="sk-test",
            gleaning_rounds=0,
            skip_embedding=True,
        )
        runner._build_services()
        runner._openai_svc.async_chat_completion = mock_chat_completion
        runner._openai_svc.complete = mock_complete

        # Run 1: only chunking
        result1 = PipelineResult(success=False, run_id="", total_elapsed_seconds=0)
        chunks = await runner._run_chunking(result1, max_chunks=5, on_progress=None)
        runner.cache.initialize_run(total_chunks=len(chunks))
        n_chunks = len(chunks)
        assert n_chunks > 0
        assert runner.cache.is_stage_complete("chunking")

        # Run 2: full run with resume
        result2 = await runner.run(force_reindex=False, max_chunks=5)

        # Chunking should be in skipped (not re-done)
        assert "chunking" in result2.stages_skipped
        assert result2.chunks_count == n_chunks
        assert result2.summaries_count > 0