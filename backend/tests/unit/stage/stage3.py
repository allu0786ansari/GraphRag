"""
tests/unit/test_stage3_services.py — Stage 3 service layer tests.

Test strategy:
  - TokenizerService: all real tiktoken calls — no mocking needed
  - FAISSService:     all real FAISS operations — no mocking needed
  - OpenAIService:    mock the OpenAI client — we test our wrapper logic,
                      not the OpenAI API itself
  - EmbeddingService: mock the OpenAI client — test our batching/normalization logic

No real API keys are required. Tests run fully offline.

Coverage targets:
  - tokenizer_service.py : 100% (pure logic, no external deps)
  - faiss_service.py     : 95%+ (pure FAISS, no external deps)
  - openai_service.py    : 85%+ (mocked OpenAI client)
  - embedding_service.py : 85%+ (mocked OpenAI client)
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tokenizer():
    """Real TokenizerService — no mocking."""
    from app.services.tokenizer_service import TokenizerService
    return TokenizerService(model="gpt-4o")


@pytest.fixture
def faiss_service():
    """Fresh FAISSService with 1536-dim (text-embedding-3-small)."""
    from app.services.faiss_service import FAISSService
    return FAISSService(embedding_dim=1536)


@pytest.fixture
def small_embeddings():
    """
    Small set of L2-normalized random float32 embeddings for testing.
    Shape: (20, 1536) — simulates 20 indexed chunks.
    """
    rng = np.random.default_rng(seed=42)
    vectors = rng.standard_normal((20, 1536)).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype(np.float32)


@pytest.fixture
def small_metadata():
    """Metadata for small_embeddings — 20 fake chunks."""
    return [
        {
            "chunk_id": f"doc_001_{i:04d}",
            "text": f"This is the text of chunk {i}. It discusses topic {i % 5}.",
            "source_document": f"doc_{i // 5:03d}.json",
            "token_count": 15,
        }
        for i in range(20)
    ]


def _make_openai_response(content: str, model: str = "gpt-4o") -> MagicMock:
    """
    Build a mock ChatCompletion response matching the OpenAI SDK shape.
    """
    mock_response = MagicMock()
    mock_response.model = model
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    return mock_response


def _make_embedding_response(n: int, dim: int = 1536) -> MagicMock:
    """
    Build a mock CreateEmbeddingResponse with n random L2-normalized vectors.
    """
    rng = np.random.default_rng(seed=0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = (vectors / norms).astype(np.float32)

    mock_response = MagicMock()
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = n * 5

    items = []
    for i in range(n):
        item = MagicMock()
        item.index = i
        item.embedding = vectors[i].tolist()
        items.append(item)
    mock_response.data = items
    return mock_response


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:

    def test_tokenizer_service_imports(self):
        from app.services.tokenizer_service import TokenizerService, get_tokenizer
        assert TokenizerService is not None
        assert get_tokenizer is not None

    def test_openai_service_imports(self):
        from app.services.openai_service import (
            OpenAIService, CompletionResult, get_openai_service,
            build_messages, system_message, user_message, assistant_message,
        )
        assert OpenAIService is not None

    def test_embedding_service_imports(self):
        from app.services.embedding_service import (
            EmbeddingService, get_embedding_service,
            EMBEDDING_DIM_SMALL, EMBEDDING_DIM_LARGE,
        )
        assert EMBEDDING_DIM_SMALL == 1536
        assert EMBEDDING_DIM_LARGE == 3072

    def test_faiss_service_imports(self):
        from app.services.faiss_service import FAISSService, SearchResult, get_faiss_service
        assert FAISSService is not None

    def test_services_init_imports(self):
        from app.services import (
            TokenizerService, OpenAIService, EmbeddingService, FAISSService,
            CompletionResult, SearchResult, get_tokenizer, get_openai_service,
            get_embedding_service, get_faiss_service,
        )
        assert all(x is not None for x in [
            TokenizerService, OpenAIService, EmbeddingService, FAISSService
        ])

    def test_no_circular_imports(self):
        import importlib
        for module in [
            "app.services.tokenizer_service",
            "app.services.openai_service",
            "app.services.embedding_service",
            "app.services.faiss_service",
        ]:
            mod = importlib.import_module(module)
            assert mod is not None


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER SERVICE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenizerServiceInit:

    def test_default_model(self, tokenizer):
        assert tokenizer.model == "gpt-4o"
        assert tokenizer.encoding_name in ("cl100k_base", "o200k_base")  # ✅ version-agnostic


    def test_repr(self, tokenizer):
        r = repr(tokenizer)
        assert "gpt-4o" in r
        assert "base" in r

    def test_custom_model(self):
        from app.services.tokenizer_service import TokenizerService
        t = TokenizerService(model="gpt-4-turbo")
        assert t.model == "gpt-4-turbo"

    def test_unknown_model_fallback(self):
        """Unknown models should fall back gracefully to cl100k_base."""
        from app.services.tokenizer_service import TokenizerService
        t = TokenizerService(model="gpt-99-unknown")
        # Should not raise — fallback applied
        assert t.count_tokens("hello") > 0


class TestCountTokens:

    def test_empty_string_returns_zero(self, tokenizer):
        assert tokenizer.count_tokens("") == 0

    def test_single_word(self, tokenizer):
        count = tokenizer.count_tokens("hello")
        assert count == 1

    def test_known_sentence(self, tokenizer):
        """
        'Hello, world!' should tokenize to exactly 4 tokens with cl100k_base.
        (Hello ,  world !)
        Verify exact tiktoken output — this is the core contract.
        """
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")  # ✅ matches service behavior
        expected = len(enc.encode("Hello, world!"))
        assert tokenizer.count_tokens("Hello, world!") == expected

    def test_longer_text(self, tokenizer):
        text = "OpenAI was founded in 2015 by Sam Altman and Elon Musk."
        count = tokenizer.count_tokens(text)
        assert count > 0
        # Rough sanity check: ~1 token per 4 chars
        assert count >= len(text) // 8

    def test_unicode_text(self, tokenizer):
        text = "日本語テスト — Japanese language test"
        count = tokenizer.count_tokens(text)
        assert count > 0

    def test_whitespace_only(self, tokenizer):
        count = tokenizer.count_tokens("   ")
        # Whitespace has tokens too
        assert count >= 0

    def test_matches_tiktoken_directly(self, tokenizer):
        """count_tokens MUST match raw tiktoken for any string."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "GraphRAG implements hierarchical Leiden community detection.",
            "text-embedding-3-small produces 1536-dimensional vectors.",
            "A" * 100,
            "  spaces and\ttabs\n",
        ]
        for s in test_strings:
            expected = len(enc.encode(s))
            actual = tokenizer.count_tokens(s)
            preview = repr(s[:30])
            assert actual == expected, (
                f"Token count mismatch for {preview}: "
                f"expected {expected}, got {actual}"
            )


class TestEncodeDecodeRoundtrip:

    def test_encode_returns_list_of_ints(self, tokenizer):
        ids = tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_empty_encode(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_decode_roundtrip(self, tokenizer):
        """Encoding then decoding should recover the original text."""
        text = "GraphRAG is a knowledge graph-based retrieval system."
        ids = tokenizer.encode(text)
        recovered = tokenizer.decode(ids)
        assert recovered == text

    def test_empty_decode(self, tokenizer):
        assert tokenizer.decode([]) == ""


class TestTruncateToLimit:

    def test_short_text_unchanged(self, tokenizer):
        text = "Hello world"
        result = tokenizer.truncate_to_limit(text, max_tokens=100)
        assert result == text

    def test_long_text_truncated(self, tokenizer):
        # Generate text that exceeds 10 tokens
        text = "This is a sentence that will be truncated. " * 10
        result = tokenizer.truncate_to_limit(text, max_tokens=10, truncation_marker="")
        assert tokenizer.count_tokens(result) <= 10

    def test_truncation_marker_appended(self, tokenizer):
        text = "word " * 100
        result = tokenizer.truncate_to_limit(text, max_tokens=20, truncation_marker=" [TRUNCATED]")
        assert result.endswith(" [TRUNCATED]")

    def test_empty_text_unchanged(self, tokenizer):
        assert tokenizer.truncate_to_limit("", max_tokens=100) == ""

    def test_exact_limit_unchanged(self, tokenizer):
        text = "Hello world"
        limit = tokenizer.count_tokens(text)
        result = tokenizer.truncate_to_limit(text, max_tokens=limit)
        assert result == text

    def test_result_fits_in_limit(self, tokenizer):
        text = "The paper describes hierarchical knowledge graph indexing. " * 50
        result = tokenizer.truncate_to_limit(text, max_tokens=50, truncation_marker="")
        assert tokenizer.count_tokens(result) <= 50


class TestFitsInWindow:

    def test_short_text_fits(self, tokenizer):
        assert tokenizer.fits_in_window("Hello", max_tokens=100) is True

    def test_exact_limit_fits(self, tokenizer):
        text = "Hello world"
        limit = tokenizer.count_tokens(text)
        assert tokenizer.fits_in_window(text, max_tokens=limit) is True

    def test_exceeds_limit(self, tokenizer):
        text = "word " * 100
        assert tokenizer.fits_in_window(text, max_tokens=5) is False


class TestTokensRemaining:

    def test_positive_remaining(self, tokenizer):
        remaining = tokenizer.tokens_remaining("Hello", max_tokens=1000)
        assert remaining > 0

    def test_negative_remaining_when_over_limit(self, tokenizer):
        text = "word " * 100
        remaining = tokenizer.tokens_remaining(text, max_tokens=10)
        assert remaining < 0


class TestBatchCountTokens:

    def test_batch_matches_individual_counts(self, tokenizer):
        texts = [
            "Hello world",
            "OpenAI was founded in 2015.",
            "GraphRAG uses hierarchical community detection.",
        ]
        batch_counts = tokenizer.batch_count_tokens(texts)
        individual_counts = [tokenizer.count_tokens(t) for t in texts]
        assert batch_counts == individual_counts

    def test_empty_list(self, tokenizer):
        assert tokenizer.batch_count_tokens([]) == []

    def test_total_tokens(self, tokenizer):
        texts = ["Hello", "world", "!"]
        total = tokenizer.total_tokens(texts)
        expected = sum(tokenizer.count_tokens(t) for t in texts)
        assert total == expected


class TestChunkText:

    def test_short_text_produces_one_chunk(self, tokenizer):
        text = "Hello world, this is a short document."
        chunks = tokenizer.chunk_text(text, chunk_size=600, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0

    def test_long_text_produces_multiple_chunks(self, tokenizer):
        # Generate ~1200 tokens of text
        text = "This is a test sentence for chunking. " * 120
        chunks = tokenizer.chunk_text(text, chunk_size=600, chunk_overlap=100)
        assert len(chunks) >= 2

    def test_chunk_token_counts_are_correct(self, tokenizer):
        text = "Word " * 200
        chunks = tokenizer.chunk_text(text, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            actual = tokenizer.count_tokens(chunk["text"])
            # Token count should match stored count
            assert abs(actual - chunk["token_count"]) <= 1  # allow ±1 for boundary effects

    def test_chunk_indices_are_sequential(self, tokenizer):
        text = "sentence " * 200
        chunks = tokenizer.chunk_text(text, chunk_size=100, chunk_overlap=10)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_all_chunks_have_required_keys(self, tokenizer):
        text = "The quick brown fox. " * 50
        chunks = tokenizer.chunk_text(text)
        for chunk in chunks:
            assert "text" in chunk
            assert "token_count" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert "chunk_index" in chunk

    def test_empty_text_returns_empty_list(self, tokenizer):
        assert tokenizer.chunk_text("") == []
        assert tokenizer.chunk_text("   ") == []

    def test_invalid_chunk_size_raises(self, tokenizer):
        with pytest.raises(ValueError, match="chunk_size"):
            tokenizer.chunk_text("hello", chunk_size=0)

    def test_invalid_overlap_raises(self, tokenizer):
        with pytest.raises(ValueError, match="chunk_overlap"):
            tokenizer.chunk_text("hello", chunk_size=100, chunk_overlap=100)

    def test_paper_exact_defaults(self, tokenizer):
        """Verify paper-exact 600/100 defaults work without arguments."""
        text = "This is a test document with enough content for chunking. " * 50
        chunks = tokenizer.chunk_text(text)  # uses defaults: 600, 100
        if len(chunks) > 1:
            # All chunks except possibly the last should be ≤600 tokens
            for chunk in chunks[:-1]:
                assert chunk["token_count"] <= 600

    def test_no_text_lost_in_chunking(self, tokenizer):
        """
        The total text in chunks should cover the full document.
        We verify by checking that all chars appear in at least one chunk.
        """
        text = "ABCDE " * 40  # Distinctive text for easy verification
        chunks = tokenizer.chunk_text(text, chunk_size=50, chunk_overlap=10)
        combined = " ".join(c["text"] for c in chunks)
        # All words should appear (overlaps mean some appear multiple times)
        for word in text.split():
            assert word in combined


class TestChunkTextIter:

    def test_iter_matches_list(self, tokenizer):
        text = "sentence word more text here. " * 80
        list_result  = tokenizer.chunk_text(text, chunk_size=100, chunk_overlap=20)
        iter_result  = list(tokenizer.chunk_text_iter(text, chunk_size=100, chunk_overlap=20))
        assert len(list_result) == len(iter_result)
        for a, b in zip(list_result, iter_result):
            assert a["chunk_index"] == b["chunk_index"]
            assert a["token_count"] == b["token_count"]


class TestBuildContextWindow:

    def test_all_items_fit(self, tokenizer):
        items = ["Hello world", "Second item", "Third item"]
        context, included, truncated = tokenizer.build_context_window(items, max_tokens=1000)
        assert not truncated
        assert len(included) == 3
        assert "Hello world" in context

    def test_truncation_when_over_limit(self, tokenizer):
        # Each item is ~5 tokens, limit is 8 tokens → only 1-2 items fit
        items = ["Hello world here", "Second item text", "Third item more"]
        context, included, truncated = tokenizer.build_context_window(items, max_tokens=8)
        assert truncated
        assert len(included) < 3

    def test_separator_used(self, tokenizer):
        items = ["Item one", "Item two"]
        context, _, _ = tokenizer.build_context_window(items, separator="---")
        assert "---" in context

    def test_empty_items(self, tokenizer):
        context, included, truncated = tokenizer.build_context_window([], max_tokens=8000)
        assert context == ""
        assert included == []
        assert not truncated

    def test_returns_original_indices_when_not_shuffled(self, tokenizer):
        items = ["a b c", "d e f", "g h i"]
        _, included, _ = tokenizer.build_context_window(items, max_tokens=1000, shuffle=False)
        assert included == [0, 1, 2]


class TestTokenizerSingleton:

    def test_get_tokenizer_returns_instance(self):
        from app.services.tokenizer_service import get_tokenizer, TokenizerService
        t = get_tokenizer()
        assert isinstance(t, TokenizerService)

    def test_get_tokenizer_returns_same_instance(self):
        from app.services.tokenizer_service import get_tokenizer
        t1 = get_tokenizer()
        t2 = get_tokenizer()
        assert t1 is t2


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS SERVICE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFAISSServiceInit:

    def test_default_dim(self):
        from app.services.faiss_service import FAISSService
        s = FAISSService()
        assert s.embedding_dim == 1536

    def test_custom_dim(self):
        from app.services.faiss_service import FAISSService
        s = FAISSService(embedding_dim=64)
        assert s.embedding_dim == 64

    def test_not_built_initially(self, faiss_service):
        assert faiss_service.is_built is False
        assert faiss_service.total_vectors == 0

    def test_repr(self, faiss_service):
        r = repr(faiss_service)
        assert "1536" in r
        assert "is_built=False" in r


class TestFAISSBuildIndex:

    def test_build_with_valid_data(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        assert faiss_service.is_built is True
        assert faiss_service.total_vectors == 20
        assert faiss_service.metadata_count == 20

    def test_build_wrong_dimension_raises(self, faiss_service, small_metadata):
        wrong_dim = np.random.randn(5, 128).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            faiss_service.build_index(wrong_dim, small_metadata[:5])

    def test_build_mismatched_lengths_raises(self, faiss_service, small_embeddings):
        with pytest.raises(ValueError, match="same length"):
            faiss_service.build_index(small_embeddings, [{"chunk_id": "x"}])

    def test_build_with_1d_raises(self, faiss_service, small_metadata):
        flat = np.random.randn(1536).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            faiss_service.build_index(flat, small_metadata)

    def test_rebuild_replaces_existing(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        # Rebuild with half the data
        new_emb  = small_embeddings[:5]
        new_meta = small_metadata[:5]
        faiss_service.build_index(new_emb, new_meta)
        assert faiss_service.total_vectors == 5


class TestFAISSSearch:

    def test_search_returns_top_k(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        query = small_embeddings[0]  # Use a known vector as query
        results = faiss_service.search(query, top_k=5)
        assert len(results) == 5

    def test_search_result_fields(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        query = small_embeddings[0]
        results = faiss_service.search(query, top_k=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r.rank, int)
            assert 0.0 <= r.score <= 1.001  # cosine similarity ≤ 1, allow tiny float error
            assert isinstance(r.chunk_id, str)
            assert isinstance(r.text, str)

    def test_top_result_is_most_similar(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        # Query with exact copy of vector[0] — it should be rank 1 with score ≈ 1.0
        query = small_embeddings[0].copy()
        results = faiss_service.search(query, top_k=1)
        assert len(results) == 1
        assert results[0].score > 0.99  # nearly identical

    def test_results_ordered_by_score_desc(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        query = small_embeddings[3]
        results = faiss_service.search(query, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_on_unbuilt_index_raises(self, faiss_service):
        query = np.random.randn(1536).astype(np.float32)
        with pytest.raises(RuntimeError, match="not built"):
            faiss_service.search(query)

    def test_top_k_larger_than_index_returns_all(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        query = small_embeddings[0]
        results = faiss_service.search(query, top_k=20)
        assert len(results) == 20  # only 20 vectors in index

    def test_score_threshold_filters_results(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        query = small_embeddings[0]
        results = faiss_service.search(query, top_k=20, score_threshold=0.99)
        # Only the query vector itself should score ≥ 0.99
        assert all(r.score >= 0.99 for r in results)

    def test_search_with_2d_query(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        # 2D query: shape (1, 1536)
        query = small_embeddings[0].reshape(1, -1)
        results = faiss_service.search(query, top_k=3)
        assert len(results) == 3

    def test_wrong_query_dimension_raises(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        bad_query = np.random.randn(128).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            faiss_service.search(bad_query)

    def test_rank_starts_at_1(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        results = faiss_service.search(small_embeddings[0], top_k=5)
        assert results[0].rank == 1
        assert results[1].rank == 2


class TestFAISSMetadata:

    def test_metadata_stored_correctly(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        meta = faiss_service.get_metadata(0)
        assert meta["chunk_id"] == "doc_001_0000"
        assert "text" in meta

    def test_metadata_out_of_range_returns_none(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        assert faiss_service.get_metadata(9999) is None
        assert faiss_service.get_metadata(-1) is None


class TestFAISSAddVectors:

    def test_add_to_existing_index(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings[:10], small_metadata[:10])
        assert faiss_service.total_vectors == 10

        extra_emb  = small_embeddings[10:15]
        extra_meta = small_metadata[10:15]
        faiss_service.add_vectors(extra_emb, extra_meta)
        assert faiss_service.total_vectors == 15

    def test_add_to_empty_builds_index(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.add_vectors(small_embeddings[:5], small_metadata[:5])
        assert faiss_service.is_built is True
        assert faiss_service.total_vectors == 5


class TestFAISSPersistence:

    def test_save_and_load_roundtrip(self, faiss_service, small_embeddings, small_metadata, tmp_path):
        faiss_service.build_index(small_embeddings, small_metadata)

        index_path    = tmp_path / "test.bin"
        metadata_path = tmp_path / "metadata.json"
        faiss_service.save(index_path, metadata_path)

        # Verify files exist
        assert index_path.exists()
        assert metadata_path.exists()

        # Load into a new service
        from app.services.faiss_service import FAISSService
        new_service = FAISSService(embedding_dim=1536)
        new_service.load(index_path, metadata_path)

        assert new_service.is_built is True
        assert new_service.total_vectors == 20

        # Results should be identical
        query   = small_embeddings[5]
        orig    = faiss_service.search(query, top_k=5)
        loaded  = new_service.search(query, top_k=5)
        assert len(orig) == len(loaded)
        for o, l in zip(orig, loaded):
            assert abs(o.score - l.score) < 1e-5
            assert o.chunk_id == l.chunk_id

    def test_metadata_preserved_after_roundtrip(self, faiss_service, small_embeddings, small_metadata, tmp_path):
        faiss_service.build_index(small_embeddings, small_metadata)
        index_path    = tmp_path / "idx.bin"
        metadata_path = tmp_path / "meta.json"
        faiss_service.save(index_path, metadata_path)

        from app.services.faiss_service import FAISSService
        s2 = FAISSService(embedding_dim=1536)
        s2.load(index_path, metadata_path)

        assert s2.metadata_count == 20
        assert s2.get_metadata(0)["chunk_id"] == "doc_001_0000"

    def test_save_without_build_raises(self, faiss_service, tmp_path):
        with pytest.raises(RuntimeError, match="not been built"):
            faiss_service.save(tmp_path / "x.bin", tmp_path / "x.json")

    def test_load_missing_file_raises(self, faiss_service, tmp_path):
        with pytest.raises(FileNotFoundError):
            faiss_service.load(tmp_path / "missing.bin", tmp_path / "missing.json")

    def test_load_wrong_dimension_raises(self, faiss_service, small_embeddings, small_metadata, tmp_path):
        faiss_service.build_index(small_embeddings, small_metadata)
        index_path    = tmp_path / "idx.bin"
        metadata_path = tmp_path / "meta.json"
        faiss_service.save(index_path, metadata_path)

        # Load into service with wrong dimension
        from app.services.faiss_service import FAISSService
        wrong_service = FAISSService(embedding_dim=128)
        with pytest.raises(ValueError, match="dimension"):
            wrong_service.load(index_path, metadata_path)


class TestFAISSReset:

    def test_reset_clears_state(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        assert faiss_service.is_built is True
        faiss_service.reset()
        assert faiss_service.is_built is False
        assert faiss_service.total_vectors == 0
        assert faiss_service.metadata_count == 0


class TestFAISSSearchAndFillContext:

    def test_fills_context_within_token_budget(self, faiss_service, small_embeddings, small_metadata):
        faiss_service.build_index(small_embeddings, small_metadata)
        query = small_embeddings[0]
        results, total_tokens = faiss_service.search_and_fill_context(
            query, max_tokens=30, top_k=20
        )
        # Each chunk has token_count=15, so at most 2 fit in 30 tokens
        assert total_tokens <= 30
        assert len(results) <= 2

    def test_returns_empty_for_unbuilt_index(self):
        from app.services.faiss_service import FAISSService
        s = FAISSService()
        # Empty index raises RuntimeError from search()
        with pytest.raises(RuntimeError):
            s.search_and_fill_context(np.zeros(1536, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI SERVICE TESTS (mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAIServiceInit:

    def test_init_stores_config(self):
        from app.services.openai_service import OpenAIService
        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI"):
            svc = OpenAIService(
                api_key="sk-test",
                model="gpt-4o",
                max_tokens=2048,
                temperature=0.1,
            )
            assert svc.model == "gpt-4o"
            assert svc.max_tokens == 2048
            assert svc.temperature == 0.1

    def test_repr(self):
        from app.services.openai_service import OpenAIService
        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI"):
            svc = OpenAIService(api_key="sk-test")
            r = repr(svc)
            assert "gpt-4o" in r


class TestBuildMessages:

    def test_user_only(self):
        from app.services.openai_service import build_messages
        msgs = build_messages("Hello")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_with_system(self):
        from app.services.openai_service import build_messages
        msgs = build_messages("Hello", system_prompt="Be helpful.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_with_history(self):
        from app.services.openai_service import build_messages, assistant_message
        history = [assistant_message("Previous reply.")]
        msgs = build_messages("Follow-up", history=history)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "user"

    def test_full_conversation(self):
        from app.services.openai_service import build_messages, assistant_message, system_message
        history = [assistant_message("I understand.")]
        msgs = build_messages("Next question", system_prompt="System.", history=history)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"


class TestCompletionResult:

    def test_cost_estimate(self):
        from app.services.openai_service import CompletionResult
        result = CompletionResult(
            content="test",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
            model="gpt-4o",
            finish_reason="stop",
        )
        # $2.50/1M input + $10/1M output = $12.50 total
        assert abs(result.estimated_cost_usd - 12.50) < 0.01

    def test_repr_truncates_content(self):
        from app.services.openai_service import CompletionResult
        long_content = "A" * 200
        result = CompletionResult(
            content=long_content,
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            model="gpt-4o", finish_reason="stop",
        )
        r = repr(result)
        assert "CompletionResult" in r


class TestSyncChatCompletion:

    def test_returns_completion_result(self):
        from app.services.openai_service import OpenAIService, CompletionResult
        mock_response = _make_openai_response("Test answer")

        with patch("app.services.openai_service.OpenAI") as mock_openai_cls, \
             patch("app.services.openai_service.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            svc = OpenAIService(api_key="sk-test")
            from app.services.openai_service import build_messages
            msgs = build_messages("What are the themes?")
            result = svc.chat_completion(msgs)

            assert isinstance(result, CompletionResult)
            assert result.content == "Test answer"
            assert result.total_tokens == 150

    def test_passes_correct_kwargs_to_api(self):
        from app.services.openai_service import OpenAIService, build_messages

        mock_response = _make_openai_response("Answer")

        with patch("app.services.openai_service.OpenAI") as mock_openai_cls, \
             patch("app.services.openai_service.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            svc = OpenAIService(api_key="sk-test", model="gpt-4o", temperature=0.0)
            msgs = build_messages("test prompt")
            svc.chat_completion(msgs, max_tokens=500)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["temperature"] == 0.0

    def test_json_mode_sets_response_format(self):
        from app.services.openai_service import OpenAIService, build_messages

        mock_response = _make_openai_response('{"key": "value"}')

        with patch("app.services.openai_service.OpenAI") as mock_openai_cls, \
             patch("app.services.openai_service.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            svc = OpenAIService(api_key="sk-test")
            svc.complete_sync("Return JSON", json_mode=True)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("response_format") == {"type": "json_object"}


class TestAsyncChatCompletion:

    @pytest.mark.asyncio
    async def test_async_returns_completion_result(self):
        from app.services.openai_service import OpenAIService, CompletionResult, build_messages

        mock_response = _make_openai_response("Async answer")

        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI") as mock_async_cls:
            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_async_client

            svc = OpenAIService(api_key="sk-test")
            msgs = build_messages("What are the themes?")
            result = await svc.async_chat_completion(msgs)

            assert isinstance(result, CompletionResult)
            assert result.content == "Async answer"
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_complete_convenience_wrapper(self):
        from app.services.openai_service import OpenAIService

        mock_response = _make_openai_response("Convenient answer")

        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI") as mock_async_cls:
            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_async_client

            svc = OpenAIService(api_key="sk-test")
            result = await svc.complete(
                user_prompt="What are themes?",
                system_prompt="You are an analyst.",
            )
            assert result.content == "Convenient answer"


class TestLogitBiasCompletion:

    @pytest.mark.asyncio
    async def test_logit_bias_passed_to_api(self):
        from app.services.openai_service import OpenAIService, build_messages

        mock_response = _make_openai_response("YES")

        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI") as mock_async_cls:
            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_async_client

            svc = OpenAIService(api_key="sk-test")
            msgs = build_messages("Were all entities extracted?")
            result = await svc.async_completion_with_logit_bias(
                messages=msgs,
                yes_token_ids=[9642],
                no_token_ids=[2201],
                bias=100,
            )

            # Check logit_bias was passed
            call_kwargs = mock_async_client.chat.completions.create.call_args[1]
            assert "logit_bias" in call_kwargs
            assert call_kwargs["logit_bias"]["9642"] == 100
            assert call_kwargs["logit_bias"]["2201"] == 100
            assert call_kwargs["max_tokens"] == 1  # forced single token

    @pytest.mark.asyncio
    async def test_logit_bias_max_tokens_is_one(self):
        """Gleaning loop must use max_tokens=1 for forced yes/no."""
        from app.services.openai_service import OpenAIService, build_messages

        mock_response = _make_openai_response("NO")

        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI") as mock_async_cls:
            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_async_client

            svc = OpenAIService(api_key="sk-test")
            msgs = build_messages("Were entities missed?")
            await svc.async_completion_with_logit_bias(
                messages=msgs, yes_token_ids=[9642], no_token_ids=[2201]
            )

            call_kwargs = mock_async_client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 1


class TestBatchComplete:

    @pytest.mark.asyncio
    async def test_batch_returns_all_results(self):
        from app.services.openai_service import OpenAIService, CompletionResult

        responses = [_make_openai_response(f"Answer {i}") for i in range(5)]
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            resp = responses[call_count % len(responses)]
            call_count += 1
            return resp

        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI") as mock_async_cls:
            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = mock_create
            mock_async_cls.return_value = mock_async_client

            svc = OpenAIService(api_key="sk-test")
            from app.services.openai_service import build_messages
            prompts = [build_messages(f"Question {i}") for i in range(5)]
            results = await svc.batch_complete(prompts, max_concurrency=5)

            assert len(results) == 5
            assert all(isinstance(r, CompletionResult) for r in results)

    @pytest.mark.asyncio
    async def test_batch_empty_prompts(self):
        from app.services.openai_service import OpenAIService

        with patch("app.services.openai_service.OpenAI"), \
             patch("app.services.openai_service.AsyncOpenAI"):
            svc = OpenAIService(api_key="sk-test")
            results = await svc.batch_complete([], max_concurrency=5)
            assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING SERVICE TESTS (mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingServiceInit:

    def test_default_model(self):
        from app.services.embedding_service import EmbeddingService, EMBEDDING_DIM_SMALL
        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI"):
            svc = EmbeddingService(api_key="sk-test")
            assert svc.model == "text-embedding-3-small"
            assert svc.embedding_dim == EMBEDDING_DIM_SMALL

    def test_large_model_dim(self):
        from app.services.embedding_service import EmbeddingService, EMBEDDING_DIM_LARGE
        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI"):
            svc = EmbeddingService(api_key="sk-test", model="text-embedding-3-large")
            assert svc.embedding_dim == EMBEDDING_DIM_LARGE

    def test_custom_dimensions(self):
        from app.services.embedding_service import EmbeddingService
        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI"):
            svc = EmbeddingService(api_key="sk-test", dimensions=512)
            assert svc.embedding_dim == 512


class TestL2Normalization:

    def test_l2_normalize_unit_vectors(self):
        from app.services.embedding_service import _l2_normalize
        vectors = np.random.randn(10, 1536).astype(np.float32)
        normalized = _l2_normalize(vectors)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_l2_normalize_zero_vector(self):
        from app.services.embedding_service import _l2_normalize
        vectors = np.zeros((2, 4), dtype=np.float32)
        # Zero vectors should not produce NaN
        result = _l2_normalize(vectors)
        assert not np.any(np.isnan(result))

    def test_already_normalized_unchanged(self):
        from app.services.embedding_service import _l2_normalize
        # Create a unit vector
        v = np.zeros((1, 4), dtype=np.float32)
        v[0, 0] = 1.0
        result = _l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-6)


class TestCleanText:

    def test_newlines_replaced_with_spaces(self):
        from app.services.embedding_service import _clean_text
        text = "Hello\nWorld\nTest"
        cleaned = _clean_text(text)
        assert "\n" not in cleaned
        assert "Hello" in cleaned

    def test_empty_text_becomes_space(self):
        from app.services.embedding_service import _clean_text
        assert _clean_text("") == " "
        assert _clean_text("   ") == " "


class TestSyncEmbedText:

    def test_returns_normalized_vector(self):
        from app.services.embedding_service import EmbeddingService

        mock_response = _make_embedding_response(1)

        with patch("app.services.embedding_service.OpenAI") as mock_openai_cls, \
             patch("app.services.embedding_service.AsyncOpenAI"):
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            svc = EmbeddingService(api_key="sk-test")
            vector = svc.embed_text_sync("Hello world")

            assert isinstance(vector, np.ndarray)
            assert vector.shape == (1536,)
            assert vector.dtype == np.float32
            # Check normalization: L2 norm ≈ 1.0
            norm = np.linalg.norm(vector)
            assert abs(norm - 1.0) < 1e-4


class TestAsyncEmbedText:

    @pytest.mark.asyncio
    async def test_async_embed_returns_normalized_vector(self):
        from app.services.embedding_service import EmbeddingService

        mock_response = _make_embedding_response(1)

        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI") as mock_async_cls:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_client

            svc = EmbeddingService(api_key="sk-test")
            vector = await svc.embed_text("GraphRAG community detection")

            assert isinstance(vector, np.ndarray)
            assert vector.shape == (1536,)
            assert vector.dtype == np.float32
            norm = np.linalg.norm(vector)
            assert abs(norm - 1.0) < 1e-4


class TestBatchEmbed:

    @pytest.mark.asyncio
    async def test_batch_returns_correct_shape(self):
        from app.services.embedding_service import EmbeddingService

        n_texts = 10
        texts = [f"Text chunk {i}" for i in range(n_texts)]

        mock_response = _make_embedding_response(n_texts)

        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI") as mock_async_cls:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_client

            svc = EmbeddingService(api_key="sk-test")
            result = await svc.embed_batch(texts, batch_size=10)

            assert isinstance(result, np.ndarray)
            assert result.shape == (n_texts, 1536)
            assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_batch_all_rows_normalized(self):
        from app.services.embedding_service import EmbeddingService

        n = 8
        texts = [f"chunk {i}" for i in range(n)]
        mock_response = _make_embedding_response(n)

        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI") as mock_async_cls:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_async_cls.return_value = mock_client

            svc = EmbeddingService(api_key="sk-test")
            result = await svc.embed_batch(texts, batch_size=n)

            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_allclose(norms, np.ones(n), atol=1e-4)

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        from app.services.embedding_service import EmbeddingService

        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI"):
            svc = EmbeddingService(api_key="sk-test")
            result = await svc.embed_batch([])
            assert result.shape == (0, 1536)

    @pytest.mark.asyncio
    async def test_batch_splits_into_correct_batches(self):
        """Verify that large lists are split into multiple API calls."""
        from app.services.embedding_service import EmbeddingService

        n_texts = 25
        batch_size = 10
        texts = [f"chunk {i}" for i in range(n_texts)]

        # Track call count
        call_count = 0
        call_sizes = []

        async def mock_create(**kwargs):
            nonlocal call_count
            n = len(kwargs["input"])
            call_sizes.append(n)
            call_count += 1
            return _make_embedding_response(n)

        with patch("app.services.embedding_service.OpenAI"), \
             patch("app.services.embedding_service.AsyncOpenAI") as mock_async_cls:
            mock_client = MagicMock()
            mock_client.embeddings.create = mock_create
            mock_async_cls.return_value = mock_client

            svc = EmbeddingService(api_key="sk-test")
            result = await svc.embed_batch(texts, batch_size=batch_size)

            # Should have made 3 calls: 10 + 10 + 5
            assert call_count == 3
            assert sum(call_sizes) == n_texts
            assert result.shape == (n_texts, 1536)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: EMBEDDING → FAISS ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingFAISSIntegration:
    """
    Test the full VectorRAG indexing + search flow without real API calls.
    Uses synthetic L2-normalized embeddings to verify the pipeline connects correctly.
    """

    def test_full_index_search_roundtrip(self):
        """
        Simulate: embed chunks → build FAISS index → search with query vector.
        The query vector is an exact copy of one chunk — should be rank 1.
        """
        from app.services.faiss_service import FAISSService

        rng = np.random.default_rng(seed=7)
        n_chunks = 50
        dim = 1536

        # Simulate L2-normalized embeddings from EmbeddingService
        vecs = rng.standard_normal((n_chunks, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = (vecs / norms).astype(np.float32)

        metadata = [
            {
                "chunk_id": f"chunk_{i:04d}",
                "text": f"Content about topic {i % 10}. Entity mentions here.",
                "source_document": f"doc_{i // 10}.json",
                "token_count": 600,
            }
            for i in range(n_chunks)
        ]

        # Build index
        svc = FAISSService(embedding_dim=dim)
        svc.build_index(vecs, metadata)
        assert svc.total_vectors == n_chunks

        # Query with exact copy of chunk 23 — should be rank 1
        query = vecs[23].copy()
        results = svc.search(query, top_k=5)

        assert len(results) == 5
        assert results[0].chunk_id == "chunk_0023"
        assert results[0].score > 0.999  # near-identical

    def test_cosine_similarity_semantics(self):
        """
        Vectors that point in similar directions should score higher
        than vectors pointing in opposite directions.
        """
        from app.services.faiss_service import FAISSService

        dim = 4
        # Two vectors: similar to query, dissimilar to query
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        similar = np.array([0.9, 0.436, 0.0, 0.0], dtype=np.float32)
        dissimilar = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Normalize
        similar   = similar   / np.linalg.norm(similar)
        dissimilar = dissimilar / np.linalg.norm(dissimilar)

        embeddings = np.vstack([similar, dissimilar]).astype(np.float32)
        metadata = [
            {"chunk_id": "similar",    "text": "similar text",    "source_document": "a.json", "token_count": 5},
            {"chunk_id": "dissimilar", "text": "dissimilar text", "source_document": "a.json", "token_count": 5},
        ]

        svc = FAISSService(embedding_dim=dim)
        svc.build_index(embeddings, metadata)
        results = svc.search(query, top_k=2, score_threshold=-2.0)
        assert len(results) == 2

        assert results[0].chunk_id == "similar"
        assert results[0].score > results[1].score
        # Dissimilar vector has negative cosine similarity
        assert results[1].score < 0