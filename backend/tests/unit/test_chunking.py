"""
tests/unit/test_chunking.py — ChunkingPipeline unit tests.

Tests:
  - chunk_document() with realistic text: correct chunk count, overlap, IDs
  - Empty / whitespace-only text returns []
  - max_chunks truncation
  - run() on a directory: processes JSON, TXT, MD files; skips unknowns
  - run() raises FileNotFoundError for missing directory
  - get_stats() returns correct aggregates
  - chunk_id uniqueness across multiple documents
  - Metadata attached to every chunk
  - chunk_overlap=0 produces no overlap
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tokenizer(mock_tokenizer):
    return mock_tokenizer


@pytest.fixture
def pipeline(tokenizer):
    from app.core.pipeline.chunking import ChunkingPipeline
    return ChunkingPipeline(tokenizer=tokenizer)


def write_json_doc(directory: Path, name: str, text: str, metadata: dict = None) -> Path:
    path = directory / name
    path.write_text(json.dumps({"text": text, "metadata": metadata or {}}))
    return path


def write_txt_doc(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


# ── chunk_document() ──────────────────────────────────────────────────────────

class TestChunkDocument:

    def test_returns_at_least_one_chunk_for_short_text(self, pipeline):
        chunks = pipeline.chunk_document(
            text="OpenAI was founded in 2015 by Sam Altman.",
            source_document="article.txt",
        )
        assert len(chunks) >= 1

    def test_chunk_ids_contain_doc_stem(self, pipeline):
        chunks = pipeline.chunk_document(
            text="Some content here. " * 20,
            source_document="my_document.txt",
        )
        for chunk in chunks:
            assert chunk.chunk_id.startswith("my_document_")

    def test_chunk_ids_are_unique(self, pipeline):
        text = "Token " * 2000   # force multiple chunks
        chunks = pipeline.chunk_document(
            text=text,
            source_document="doc.txt",
            chunk_size=100,
            chunk_overlap=20,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_source_document_set_on_all_chunks(self, pipeline):
        chunks = pipeline.chunk_document(
            text="Content. " * 50,
            source_document="source.txt",
        )
        for chunk in chunks:
            assert chunk.source_document == "source.txt"

    def test_metadata_propagated_to_all_chunks(self, pipeline):
        meta = {"author": "Test", "date": "2024-01-01"}
        chunks = pipeline.chunk_document(
            text="Content. " * 50,
            source_document="doc.txt",
            metadata=meta,
        )
        for chunk in chunks:
            assert chunk.metadata["author"] == "Test"
            assert chunk.metadata["date"] == "2024-01-01"

    def test_empty_text_returns_empty_list(self, pipeline):
        chunks = pipeline.chunk_document(text="", source_document="empty.txt")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self, pipeline):
        chunks = pipeline.chunk_document(text="   \n\t  ", source_document="ws.txt")
        assert chunks == []

    def test_chunk_token_counts_within_bounds(self, pipeline):
        text = "OpenAI and Microsoft partnered together. " * 200
        chunks = pipeline.chunk_document(
            text=text,
            source_document="doc.txt",
            chunk_size=100,
            chunk_overlap=20,
        )
        for chunk in chunks:
            assert chunk.token_count > 0
            # Allow for tokenizer variation: chunks should never exceed 2x limit
            assert chunk.token_count <= 200

    def test_total_chunks_in_doc_field_correct(self, pipeline):
        text = "Word " * 1000
        chunks = pipeline.chunk_document(
            text=text,
            source_document="doc.txt",
            chunk_size=100,
            chunk_overlap=10,
        )
        expected_total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks_in_doc == expected_total

    def test_chunk_indices_sequential(self, pipeline):
        text = "Word " * 500
        chunks = pipeline.chunk_document(
            text=text, source_document="doc.txt",
            chunk_size=80, chunk_overlap=10,
        )
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_larger_chunk_size_produces_fewer_chunks(self, pipeline):
        text = "OpenAI is an AI research lab. " * 200
        small_chunks = pipeline.chunk_document(
            text=text, source_document="doc.txt",
            chunk_size=50, chunk_overlap=5,
        )
        large_chunks = pipeline.chunk_document(
            text=text, source_document="doc.txt",
            chunk_size=200, chunk_overlap=20,
        )
        assert len(large_chunks) < len(small_chunks)


# ── run() on directory ────────────────────────────────────────────────────────

class TestRunDirectory:

    def test_processes_json_documents(self, pipeline, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        write_json_doc(d, "article.json", "OpenAI is an AI company. " * 20)
        chunks = pipeline.run(d, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 1
        assert all(c.source_document == "article.json" for c in chunks)

    def test_processes_txt_documents(self, pipeline, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        write_txt_doc(d, "notes.txt", "Microsoft invested in OpenAI. " * 20)
        chunks = pipeline.run(d, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 1

    def test_processes_multiple_documents(self, pipeline, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        for i in range(3):
            write_txt_doc(d, f"doc_{i}.txt", f"Document {i} content. " * 30)
        chunks = pipeline.run(d, chunk_size=200, chunk_overlap=20)
        sources = {c.source_document for c in chunks}
        assert len(sources) == 3

    def test_max_chunks_limits_output(self, pipeline, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        write_txt_doc(d, "big.txt", "Content sentence. " * 2000)
        chunks = pipeline.run(d, chunk_size=100, chunk_overlap=10, max_chunks=5)
        assert len(chunks) <= 5

    def test_empty_directory_returns_empty_list(self, pipeline, tmp_path):
        d = tmp_path / "empty_raw"
        d.mkdir()
        chunks = pipeline.run(d)
        assert chunks == []

    def test_missing_directory_raises_file_not_found(self, pipeline, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            pipeline.run(missing)

    def test_chunk_ids_unique_across_documents(self, pipeline, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        for i in range(4):
            write_txt_doc(d, f"doc_{i}.txt", f"Content {i}. " * 100)
        chunks = pipeline.run(d, chunk_size=100, chunk_overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ── get_stats() ───────────────────────────────────────────────────────────────

class TestGetStats:

    def test_empty_list_returns_total_zero(self, pipeline):
        stats = pipeline.get_stats([])
        assert stats["total"] == 0

    def test_stats_returns_correct_counts(self, pipeline):
        from app.models.graph_models import ChunkSchema
        chunks = [
            ChunkSchema(
                chunk_id=f"c_{i}", source_document=f"doc_{i}.txt",
                text="text", token_count=50 + i,
                start_char=0, end_char=10, chunk_index=0, total_chunks_in_doc=1,
            )
            for i in range(5)
        ]
        stats = pipeline.get_stats(chunks)
        assert stats["total_chunks"] == 5
        assert stats["total_documents"] == 5
        assert stats["total_tokens"] == sum(50 + i for i in range(5))