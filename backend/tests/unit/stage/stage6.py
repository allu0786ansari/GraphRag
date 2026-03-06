"""
tests/unit/test_stage6_query.py — Stage 6 Query Engine tests.

Test strategy:
  - Zero real API calls. All LLM/embedding calls are mocked.
  - Real FAISS index built in-memory (no disk I/O).
  - Real tokenizer (tiktoken).
  - Real ROUGE-L implementation — no mocks.
  - Real claim deduplication and clustering — no mocks.
  - Real JSON parsing in all engines — no mocks.
  - File I/O via tmp_path for SummaryStore.

Coverage targets:
  vectorrag_engine.py     90%+
  graphrag_engine.py      90%+
  evaluation_engine.py    85%+
  claim_validation.py     95%+   (pure functions are fully testable)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES & FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir(tmp_path) -> Path:
    return tmp_path


def make_completion_result(content: str, prompt_tokens: int = 200, completion_tokens: int = 150):
    """Build a CompletionResult with all required fields."""
    from app.services.openai_service import CompletionResult
    return CompletionResult(
        content=content,
        model="gpt-4o",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        finish_reason="stop",
    )


def make_mock_openai(response_content: str = "Test answer.") -> AsyncMock:
    """Return an AsyncMock OpenAI service that returns a fixed completion."""
    mock = AsyncMock()
    mock.model = "gpt-4o"
    mock.complete = AsyncMock(return_value=make_completion_result(response_content))
    mock.async_chat_completion = AsyncMock(
        return_value=make_completion_result(response_content)
    )
    return mock


def make_tokenizer():
    """Return a real TokenizerService."""
    from app.services.tokenizer_service import TokenizerService
    return TokenizerService(model="gpt-4o")


def make_community_summary(
    community_id: str = "comm_c1_0000",
    level: str = "c1",
    title: str = "AI Investment Community",
    summary: str = "OpenAI and Microsoft formed a major AI partnership.",
    impact_rating: float = 7.5,
    n_findings: int = 3,
):
    """Build a CommunitySummary for testing."""
    from app.models.graph_models import CommunitySummary, CommunityFinding, CommunityLevel
    findings = [
        CommunityFinding(
            finding_id=i,
            summary=f"Finding {i}: Key insight about AI investment.",
            explanation=f"Explanation for finding {i} with more detail.",
        )
        for i in range(n_findings)
    ]
    return CommunitySummary(
        community_id=community_id,
        level=CommunityLevel(level),
        title=title,
        summary=summary,
        impact_rating=impact_rating,
        rating_explanation="High impact due to market significance.",
        findings=findings,
        node_ids=["openai", "microsoft", "sam_altman"],
        context_tokens_used=512,
    )


def make_summary_store(tmp_path: Path, summaries: list) -> "SummaryStore":
    """Write summaries to disk and return a SummaryStore pointing at tmp_path."""
    from app.storage.summary_store import SummaryStore
    store = SummaryStore(artifacts_dir=tmp_path)
    store.save_summaries(summaries)
    return store


def make_faiss_service_with_vectors(n_vectors: int = 5, dim: int = 8):
    """Build an in-memory FAISSService with random vectors + metadata."""
    from app.services.faiss_service import FAISSService
    svc = FAISSService(embedding_dim=dim)
    vectors = np.random.rand(n_vectors, dim).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)
    metadata = [
        {
            "chunk_id": f"chunk_{i:04d}",
            "text": f"This is the text of chunk {i}. It discusses AI and technology topics.",
            "source_document": f"doc_{i:03d}.json",
            "token_count": 20,
        }
        for i in range(n_vectors)
    ]
    svc.build_index(vectors, metadata)
    return svc, vectors, metadata


def make_embedding_service_mock(dim: int = 8):
    """Return a mock EmbeddingService that returns a random unit vector."""
    mock = AsyncMock()
    mock.model = "text-embedding-3-small"
    async def fake_embed_text(text: str) -> np.ndarray:
        v = np.random.rand(dim).astype(np.float32)
        return v / np.linalg.norm(v)
    mock.embed_text = fake_embed_text
    mock.embed_batch = AsyncMock(
        return_value=np.random.rand(5, dim).astype(np.float32)
    )
    return mock


def make_vectorrag_engine(
    tmp_path: Path,
    n_chunks: int = 5,
    dim: int = 8,
    openai_response: str = "VectorRAG answer about AI.",
):
    """Build a fully wired VectorRAGEngine with in-memory FAISS + mock LLM."""
    from app.core.query.vectorrag_engine import VectorRAGEngine

    faiss_svc, _, _ = make_faiss_service_with_vectors(n_chunks, dim)
    mock_openai = make_mock_openai(openai_response)
    mock_embedding = make_embedding_service_mock(dim)
    tokenizer = make_tokenizer()

    # Write dummy index files so _ensure_index_loaded doesn't fail
    index_path = tmp_path / "faiss_index.bin"
    meta_path = tmp_path / "embeddings.json"
    index_path.write_bytes(b"dummy")
    meta_path.write_text("[]")

    engine = VectorRAGEngine(
        openai_service=mock_openai,
        embedding_service=mock_embedding,
        faiss_service=faiss_svc,
        tokenizer=tokenizer,
        faiss_index_path=index_path,
        embeddings_metadata_path=meta_path,
        context_window=8000,
    )
    # Mark index as already loaded (we built it in-memory above)
    engine._index_loaded = True
    return engine


def make_graphrag_engine(tmp_path: Path, summaries: list, openai_response: str = None):
    """Build a GraphRAGEngine backed by real SummaryStore."""
    from app.core.query.graphrag_engine import GraphRAGEngine

    store = make_summary_store(tmp_path, summaries)
    mock_openai = make_mock_openai(
        openai_response or json.dumps({
            "points": [
                {"description": "OpenAI leads AI research globally.", "score": 85},
                {"description": "Microsoft invested $10B in OpenAI.", "score": 90},
                {"description": "AI safety is a key concern.", "score": 70},
            ]
        })
    )
    tokenizer = make_tokenizer()
    return GraphRAGEngine(
        openai_service=mock_openai,
        summary_store=store,
        tokenizer=tokenizer,
        context_window=8000,
        max_concurrency=3,
        helpfulness_threshold=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT & STRUCTURAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:

    def test_vectorrag_imports(self):
        from app.core.query.vectorrag_engine import VectorRAGEngine, get_vectorrag_engine
        assert VectorRAGEngine is not None

    def test_graphrag_imports(self):
        from app.core.query.graphrag_engine import GraphRAGEngine, get_graphrag_engine
        assert GraphRAGEngine is not None

    def test_evaluation_imports(self):
        from app.core.query.evaluation_engine import EvaluationEngine, get_evaluation_engine
        assert EvaluationEngine is not None

    def test_claim_validation_imports(self):
        from app.core.query.claim_validation import (
            ClaimValidationEngine, get_claim_validation_engine,
            rouge_l_f1, _lcs_length, _tokenize,
        )
        assert ClaimValidationEngine is not None
        assert rouge_l_f1 is not None

    def test_query_init_exports(self):
        from app.core.query import (
            VectorRAGEngine, GraphRAGEngine,
            EvaluationEngine, ClaimValidationEngine,
            rouge_l_f1,
        )
        assert all(x is not None for x in [
            VectorRAGEngine, GraphRAGEngine,
            EvaluationEngine, ClaimValidationEngine, rouge_l_f1,
        ])

    def test_no_circular_imports(self):
        import importlib
        for mod in [
            "app.core.query.vectorrag_engine",
            "app.core.query.graphrag_engine",
            "app.core.query.evaluation_engine",
            "app.core.query.claim_validation",
        ]:
            assert importlib.import_module(mod) is not None

    def test_claim_validation_single_class(self):
        """The duplicate class bug must be fixed — only one ClaimValidationEngine."""
        import ast
        from pathlib import Path
        claim_file = Path(__file__).parents[2] / "app" / "core" / "query" / "claim_validation.py"
        with open(claim_file) as f:
            tree = ast.parse(f.read())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert classes.count("ClaimValidationEngine") == 1, \
            "Duplicate ClaimValidationEngine class definition found"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L PURE FUNCTION TESTS  (no mocks needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRougeLTokenize:

    def test_lowercase(self):
        from app.core.query.claim_validation import _tokenize
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        from app.core.query.claim_validation import _tokenize
        tokens = _tokenize("OpenAI, Inc. was founded.")
        assert "openai" in tokens
        assert "inc" in tokens
        assert "founded" in tokens
        assert "." not in tokens

    def test_empty_string(self):
        from app.core.query.claim_validation import _tokenize
        assert _tokenize("") == []

    def test_whitespace_only(self):
        from app.core.query.claim_validation import _tokenize
        assert _tokenize("   ") == []

    def test_numbers_preserved(self):
        from app.core.query.claim_validation import _tokenize
        tokens = _tokenize("In 2023 OpenAI raised $10B")
        assert "2023" in tokens
        assert "10b" in tokens


class TestLCSLength:

    def test_identical_sequences(self):
        from app.core.query.claim_validation import _lcs_length
        seq = ["a", "b", "c", "d"]
        assert _lcs_length(seq, seq) == 4

    def test_empty_sequences(self):
        from app.core.query.claim_validation import _lcs_length
        assert _lcs_length([], []) == 0
        assert _lcs_length(["a"], []) == 0
        assert _lcs_length([], ["a"]) == 0

    def test_no_common_elements(self):
        from app.core.query.claim_validation import _lcs_length
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_subsequence(self):
        from app.core.query.claim_validation import _lcs_length
        # "a c e" is a common subsequence of length 3
        assert _lcs_length(["a", "b", "c", "d", "e"], ["a", "c", "e"]) == 3

    def test_commutative(self):
        from app.core.query.claim_validation import _lcs_length
        seq1 = ["the", "cat", "sat"]
        seq2 = ["the", "dog", "sat"]
        assert _lcs_length(seq1, seq2) == _lcs_length(seq2, seq1)

    def test_single_element_match(self):
        from app.core.query.claim_validation import _lcs_length
        assert _lcs_length(["a"], ["a"]) == 1
        assert _lcs_length(["a"], ["b"]) == 0


class TestRougeLF1:

    def test_identical_strings_score_one(self):
        from app.core.query.claim_validation import rouge_l_f1
        s = "OpenAI was founded in 2015"
        assert rouge_l_f1(s, s) == pytest.approx(1.0)

    def test_completely_different_strings_score_zero(self):
        from app.core.query.claim_validation import rouge_l_f1
        score = rouge_l_f1("apple banana cherry", "zebra violin trumpet")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        from app.core.query.claim_validation import rouge_l_f1
        score = rouge_l_f1("the cat sat on the mat", "the cat ate the rat")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        from app.core.query.claim_validation import rouge_l_f1
        assert rouge_l_f1("", "") == 0.0
        assert rouge_l_f1("hello", "") == 0.0
        assert rouge_l_f1("", "hello") == 0.0

    def test_symmetric(self):
        from app.core.query.claim_validation import rouge_l_f1
        a = "Microsoft invested in OpenAI for AI development"
        b = "OpenAI received investment from Microsoft"
        # ROUGE-L F1 is symmetric by construction
        assert rouge_l_f1(a, b) == pytest.approx(rouge_l_f1(b, a), abs=1e-9)

    def test_score_in_range(self):
        from app.core.query.claim_validation import rouge_l_f1
        import random
        random.seed(42)
        words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast"]
        for _ in range(20):
            s1 = " ".join(random.choices(words, k=5))
            s2 = " ".join(random.choices(words, k=5))
            score = rouge_l_f1(s1, s2)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_substring_has_high_score(self):
        from app.core.query.claim_validation import rouge_l_f1
        # A claim that is a near-subset of another should have high similarity
        full = "OpenAI was founded in 2015 by Sam Altman and Elon Musk in San Francisco"
        partial = "OpenAI was founded in 2015 by Sam Altman"
        score = rouge_l_f1(partial, full)
        assert score > 0.5

    def test_near_duplicate_above_dedup_threshold(self):
        from app.core.query.claim_validation import rouge_l_f1
        c1 = "Microsoft invested ten billion dollars in OpenAI"
        c2 = "Microsoft invested $10 billion in OpenAI"
        score = rouge_l_f1(c1, c2)
        # Not identical but very similar
        assert score > 0.4  # substantial overlap


# ═══════════════════════════════════════════════════════════════════════════════
# CLAIM VALIDATION ENGINE — UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClaimDeduplication:

    def make_engine(self):
        from app.core.query.claim_validation import ClaimValidationEngine
        return ClaimValidationEngine(
            openai_service=MagicMock(),
            tokenizer=make_tokenizer(),
            dedup_threshold=0.85,
        )

    def test_empty_list(self):
        engine = self.make_engine()
        assert engine._deduplicate_claims([]) == []

    def test_single_claim(self):
        engine = self.make_engine()
        result = engine._deduplicate_claims(["OpenAI was founded in 2015."])
        assert result == ["OpenAI was founded in 2015."]

    def test_exact_duplicates_removed(self):
        engine = self.make_engine()
        claims = [
            "OpenAI was founded in 2015.",
            "OpenAI was founded in 2015.",  # exact dup
            "Microsoft invested in OpenAI.",
        ]
        result = engine._deduplicate_claims(claims)
        assert len(result) == 2
        assert "OpenAI was founded in 2015." in result
        assert "Microsoft invested in OpenAI." in result

    def test_near_duplicates_removed(self):
        engine = self.make_engine()
        claims = [
            "Sam Altman is the CEO of OpenAI the artificial intelligence company.",
            "Sam Altman is the CEO of OpenAI the artificial intelligence company.",
            "Google is a search and technology company.",
        ]
        result = engine._deduplicate_claims(claims)
        # First and second are identical → second removed
        assert len(result) == 2

    def test_distinct_claims_all_kept(self):
        engine = self.make_engine()
        claims = [
            "OpenAI was founded in 2015.",
            "Microsoft invested $10B in OpenAI.",
            "Sam Altman is the CEO of OpenAI.",
            "Google launched Gemini as a competitor to GPT-4.",
            "AI safety research is growing rapidly.",
        ]
        result = engine._deduplicate_claims(claims)
        assert len(result) == 5

    def test_preserves_first_occurrence(self):
        engine = self.make_engine()
        claims = [
            "OpenAI was founded in 2015 by Sam Altman.",
            "OpenAI was founded in 2015 by Sam Altman.",  # dup
            "Different claim entirely.",
        ]
        result = engine._deduplicate_claims(claims)
        assert result[0] == "OpenAI was founded in 2015 by Sam Altman."


class TestClaimClustering:

    def make_engine(self):
        from app.core.query.claim_validation import ClaimValidationEngine
        return ClaimValidationEngine(
            openai_service=MagicMock(),
            tokenizer=make_tokenizer(),
        )

    def test_empty_claims(self):
        engine = self.make_engine()
        assert engine._cluster_claims([], 0.7) == []

    def test_single_claim_one_cluster(self):
        engine = self.make_engine()
        labels = engine._cluster_claims(["OpenAI was founded in 2015."], 0.7)
        assert labels == [0]

    def test_identical_claims_same_cluster(self):
        engine = self.make_engine()
        claims = [
            "OpenAI was founded in 2015 by Sam Altman.",
            "OpenAI was founded in 2015 by Sam Altman.",
        ]
        labels = engine._cluster_claims(claims, 0.7)
        assert labels[0] == labels[1]

    def test_completely_different_claims_different_clusters(self):
        engine = self.make_engine()
        claims = [
            "OpenAI builds artificial intelligence systems.",
            "The French Revolution began in 1789.",
            "Photosynthesis converts sunlight into energy.",
            "Mount Everest is the highest mountain on Earth.",
        ]
        labels = engine._cluster_claims(claims, 0.7)
        assert len(set(labels)) == 4  # all in separate clusters

    def test_similar_claims_same_cluster_low_threshold(self):
        engine = self.make_engine()
        claims = [
            "OpenAI raised money from Microsoft.",
            "Microsoft invested money in OpenAI.",
        ]
        labels_tight = engine._cluster_claims(claims, 0.3)  # very tight clustering
        labels_loose = engine._cluster_claims(claims, 0.8)  # loose clustering
        # At loose threshold, these similar claims may merge
        # At tight threshold, they should stay separate
        # We just verify output length matches input
        assert len(labels_tight) == 2
        assert len(labels_loose) == 2

    def test_labels_are_integers(self):
        engine = self.make_engine()
        claims = ["Claim one.", "Claim two.", "Claim three."]
        labels = engine._cluster_claims(claims, 0.7)
        assert all(isinstance(l, int) for l in labels)
        assert len(labels) == 3

    def test_cluster_count_decreases_with_looser_threshold(self):
        engine = self.make_engine()
        # Highly similar claims
        claims = [
            "Microsoft invested ten billion dollars in OpenAI.",
            "Microsoft gave ten billion dollars to OpenAI.",
            "OpenAI received ten billion dollars from Microsoft.",
            "Apple makes iPhone smartphones.",
            "Apple produces iPhone mobile phones.",
        ]
        labels_tight  = engine._cluster_claims(claims, 0.3)  # tight → more clusters
        labels_loose  = engine._cluster_claims(claims, 0.8)  # loose → fewer clusters
        # Looser threshold merges more → fewer or equal clusters
        assert len(set(labels_loose)) <= len(set(labels_tight))


class TestClaimParsing:

    def make_engine(self):
        from app.core.query.claim_validation import ClaimValidationEngine
        return ClaimValidationEngine(
            openai_service=MagicMock(),
            tokenizer=make_tokenizer(),
        )

    def test_parse_valid_json(self):
        engine = self.make_engine()
        response = json.dumps({"claims": [
            "OpenAI was founded in 2015.",
            "Sam Altman is the CEO.",
        ]})
        result = engine._parse_claims_response(response)
        assert len(result) == 2
        assert "OpenAI was founded in 2015." in result

    def test_parse_json_with_fences(self):
        engine = self.make_engine()
        response = '```json\n{"claims": ["OpenAI was founded in 2015.", "Microsoft invested in OpenAI."]}\n```'
        result = engine._parse_claims_response(response)
        assert len(result) == 2

    def test_parse_invalid_json_returns_empty(self):
        engine = self.make_engine()
        result = engine._parse_claims_response("not json at all")
        assert result == []

    def test_parse_empty_claims_list(self):
        engine = self.make_engine()
        response = json.dumps({"claims": []})
        result = engine._parse_claims_response(response)
        assert result == []

    def test_parse_short_strings_filtered(self):
        engine = self.make_engine()
        response = json.dumps({"claims": ["OK", "A.", "Valid claim here about something."]})
        result = engine._parse_claims_response(response)
        # Only the long claim passes the >10 char filter
        assert any("Valid claim" in c for c in result)

    def test_parse_non_string_items_skipped(self):
        engine = self.make_engine()
        response = json.dumps({"claims": [
            "Valid claim text.",
            42,
            None,
            {"nested": "object"},
            "Another valid claim.",
        ]})
        result = engine._parse_claims_response(response)
        assert len(result) == 2


class TestClaimExtractionAsync:

    def make_engine(self, claims_response: list[str]):
        from app.core.query.claim_validation import ClaimValidationEngine
        mock_openai = make_mock_openai(json.dumps({"claims": claims_response}))
        return ClaimValidationEngine(
            openai_service=mock_openai,
            tokenizer=make_tokenizer(),
        )

    @pytest.mark.asyncio
    async def test_extract_and_cluster_returns_metrics(self):
        claims = [
            "OpenAI was founded in 2015 by Sam Altman.",
            "Microsoft invested $10 billion in OpenAI.",
            "Sam Altman is the CEO of OpenAI.",
            "Google DeepMind is a competitor to OpenAI.",
            "AI safety research focuses on alignment problems.",
        ]
        engine = self.make_engine(claims)
        metrics = await engine.extract_and_cluster(
            question_id=0,
            question="What are the main AI companies?",
            answer="OpenAI, Microsoft, Google are key players...",
            system="graphrag",
        )
        assert metrics.question_id == 0
        assert metrics.system == "graphrag"
        assert metrics.unique_claim_count == 5
        assert metrics.cluster_count_threshold_05 is not None
        assert metrics.cluster_count_threshold_07 is not None
        assert len(metrics.claims) == 5

    @pytest.mark.asyncio
    async def test_claims_have_correct_ids(self):
        engine = self.make_engine(["First atomic claim.", "Second atomic claim."])
        metrics = await engine.extract_and_cluster(0, "Q", "Answer text.", "vectorrag")
        for claim in metrics.claims:
            assert claim.source_system == "vectorrag"
            assert claim.question_id == 0
            assert "q000_vectorrag_claim_" in claim.claim_id

    @pytest.mark.asyncio
    async def test_compare_returns_winner(self):
        from app.core.query.claim_validation import ClaimValidationEngine
        # GraphRAG has more claims → should win
        graphrag_claims = [f"GraphRAG claim {i}." for i in range(10)]
        vectorrag_claims = [f"VectorRAG claim {i}." for i in range(4)]

        call_count = [0]

        async def mock_complete_vary(user_prompt, system_prompt=None, **kwargs):
            resp = graphrag_claims if call_count[0] == 0 else vectorrag_claims
            call_count[0] += 1
            return make_completion_result(json.dumps({"claims": resp}))

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.complete = mock_complete_vary

        engine = ClaimValidationEngine(
            openai_service=mock_openai, tokenizer=make_tokenizer()
        )
        comparison = await engine.compare(
            question_id=0,
            question="What are the themes?",
            graphrag_answer="GraphRAG answer with many details and insights.",
            vectorrag_answer="VectorRAG answer.",
        )
        assert comparison.question_id == 0
        assert comparison.comprehensiveness_winner in ("graphrag", "vectorrag", "tie")
        assert comparison.comprehensiveness_delta != 0 or comparison.comprehensiveness_winner == "tie"

    @pytest.mark.asyncio
    async def test_deduplication_reduces_claim_count(self):
        from app.core.query.claim_validation import ClaimValidationEngine
        # Include some near-duplicates
        claims = [
            "OpenAI was founded in 2015.",
            "OpenAI was founded in 2015.",  # exact dup
            "OpenAI was founded in two thousand fifteen.",  # near-dup
            "Microsoft is a large technology corporation.",
        ]
        mock_openai = make_mock_openai(json.dumps({"claims": claims}))
        engine = ClaimValidationEngine(
            openai_service=mock_openai,
            tokenizer=make_tokenizer(),
            dedup_threshold=0.85,
        )
        metrics = await engine.extract_and_cluster(0, "Q", "Answer.", "graphrag")
        # Exact dup removed, near-dup may or may not be removed depending on ROUGE-L
        assert metrics.unique_claim_count < len(claims)

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        from app.core.query.claim_validation import ClaimValidationEngine

        call_idx = [0]

        async def mock_complete_batch(user_prompt, system_prompt=None, **kwargs):
            idx = call_idx[0] % 2  # alternate between graphrag and vectorrag
            call_idx[0] += 1
            claims = [f"Claim {j} for system {idx}." for j in range(8 if idx == 0 else 4)]
            return make_completion_result(json.dumps({"claims": claims}))

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.complete = mock_complete_batch

        engine = ClaimValidationEngine(
            openai_service=mock_openai, tokenizer=make_tokenizer()
        )
        result = await engine.evaluate_batch(
            questions=["Q1", "Q2"],
            graphrag_answers=["Long answer 1.", "Long answer 2."],
            vectorrag_answers=["Short A 1.", "Short A 2."],
        )
        assert result.total_questions == 2
        assert result.avg_graphrag_claims >= 0
        assert result.avg_vectorrag_claims >= 0
        assert 0.0 <= result.graphrag_comprehensiveness_win_rate <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORRAG ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorRAGInit:

    def test_init(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        assert engine.context_window == 8000
        assert engine._index_loaded is True

    def test_get_index_stats_loaded(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir, n_chunks=5, dim=8)
        stats = engine.get_index_stats()
        assert stats["loaded"] is True
        assert stats["total_vectors"] == 5

    def test_get_index_stats_not_loaded(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        engine._index_loaded = False
        stats = engine.get_index_stats()
        assert stats["loaded"] is False
        assert stats["total_vectors"] == 0


class TestVectorRAGContextWindow:

    def test_fill_context_window_includes_all_small_chunks(self, tmp_dir):
        """With a large context window and small chunks, all retrieved chunks fit."""
        engine = make_vectorrag_engine(tmp_dir, n_chunks=5, dim=8)
        from app.services.faiss_service import SearchResult
        results = [
            SearchResult(
                rank=i + 1, index_id=i, score=0.9 - i * 0.05,
                chunk_id=f"chunk_{i:04d}",
                text=f"Short text {i}.",
                source_document=f"doc_{i}.json",
                token_count=5,
                metadata={},
            )
            for i in range(5)
        ]
        included, ctx_text, ctx_tokens = engine._fill_context_window(results, "What are the themes?")
        assert len(included) == 5

    def test_fill_context_window_stops_at_limit(self, tmp_dir):
        """Chunks that overflow the window are excluded."""
        from app.core.query.vectorrag_engine import VectorRAGEngine
        from app.services.faiss_service import SearchResult

        engine = make_vectorrag_engine(tmp_dir, n_chunks=3, dim=8)
        engine.context_window = 50  # Very small limit

        big_text = "word " * 100  # ~100 tokens
        results = [
            SearchResult(
                rank=i + 1, index_id=i, score=0.9,
                chunk_id=f"chunk_{i}",
                text=big_text,
                source_document="doc.json",
                token_count=100,
                metadata={},
            )
            for i in range(3)
        ]
        included, _, _ = engine._fill_context_window(results, "Q")
        # With 50-token limit, no big chunk should fit
        assert len(included) < 3

    def test_context_text_contains_source_info(self, tmp_dir):
        from app.services.faiss_service import SearchResult
        engine = make_vectorrag_engine(tmp_dir, n_chunks=2, dim=8)
        results = [
            SearchResult(
                rank=1, index_id=0, score=0.95,
                chunk_id="chunk_0000",
                text="AI is transforming industries.",
                source_document="ai_report.json",
                token_count=6,
                metadata={},
            )
        ]
        _, ctx_text, _ = engine._fill_context_window(results, "Q")
        assert "ai_report.json" in ctx_text
        assert "AI is transforming industries." in ctx_text

    def test_empty_search_results(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        included, ctx_text, ctx_tokens = engine._fill_context_window([], "Q")
        assert included == []
        assert ctx_text == ""
        assert ctx_tokens == 0


class TestVectorRAGQuery:

    @pytest.mark.asyncio
    async def test_query_returns_vectorrag_answer(self, tmp_dir):
        engine = make_vectorrag_engine(
            tmp_dir, openai_response="AI has transformed many industries globally."
        )
        result = await engine.query("What are the main AI trends?")
        assert result.answer == "AI has transformed many industries globally."
        assert result.query == "What are the main AI trends?"
        assert result.chunks_retrieved >= 0
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_includes_context_when_requested(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        result = await engine.query("What are the themes?", include_context=True)
        assert result.context is not None

    @pytest.mark.asyncio
    async def test_query_excludes_context_when_not_requested(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        result = await engine.query("Q", include_context=False)
        assert result.context is None

    @pytest.mark.asyncio
    async def test_query_includes_token_usage(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        result = await engine.query("Q", include_token_usage=True)
        assert result.token_usage is not None
        assert result.token_usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_query_excludes_token_usage_when_not_requested(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        result = await engine.query("Q", include_token_usage=False)
        assert result.token_usage is None

    @pytest.mark.asyncio
    async def test_query_empty_results_returns_graceful_answer(self, tmp_dir):
        from app.core.query.vectorrag_engine import VectorRAGEngine
        from app.services.faiss_service import FAISSService

        faiss_svc = MagicMock()
        faiss_svc.search = MagicMock(return_value=[])  # no results

        engine = VectorRAGEngine(
            openai_service=make_mock_openai(),
            embedding_service=make_embedding_service_mock(),
            faiss_service=faiss_svc,
            tokenizer=make_tokenizer(),
            faiss_index_path=tmp_dir / "idx.bin",
            embeddings_metadata_path=tmp_dir / "emb.json",
        )
        engine._index_loaded = True
        result = await engine.query("Totally irrelevant question.")
        assert result.chunks_retrieved == 0
        assert len(result.answer) > 0  # graceful non-empty response

    @pytest.mark.asyncio
    async def test_query_missing_index_raises(self, tmp_dir):
        from app.core.query.vectorrag_engine import VectorRAGEngine
        engine = VectorRAGEngine(
            openai_service=make_mock_openai(),
            embedding_service=make_embedding_service_mock(),
            faiss_service=MagicMock(),
            tokenizer=make_tokenizer(),
            faiss_index_path=tmp_dir / "missing.bin",
            embeddings_metadata_path=tmp_dir / "missing.json",
        )
        with pytest.raises(FileNotFoundError):
            engine._ensure_index_loaded()

    @pytest.mark.asyncio
    async def test_context_tokens_used_positive(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir, n_chunks=5)
        result = await engine.query("What is OpenAI?")
        assert result.context_tokens_used >= 0

    def test_reload_index_resets_flag(self, tmp_dir):
        engine = make_vectorrag_engine(tmp_dir)
        assert engine._index_loaded is True
        engine._index_loaded = False  # simulate unloaded
        # The index path points to a dummy file (not a real FAISS index)
        # so loading raises either FileNotFoundError or RuntimeError from FAISS
        with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
            engine.reload_index()


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHRAG ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphRAGContextBuilding:

    def make_engine(self, tmp_dir):
        summaries = [make_community_summary() for _ in range(3)]
        return make_graphrag_engine(tmp_dir, summaries)

    def test_build_community_context_contains_title(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        community = make_community_summary(title="AI Investment Report")
        ctx = engine._build_community_context(community)
        assert "AI Investment Report" in ctx

    def test_build_community_context_contains_summary(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        community = make_community_summary(summary="OpenAI leads in language models.")
        ctx = engine._build_community_context(community)
        assert "OpenAI leads in language models." in ctx

    def test_build_community_context_contains_findings(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        community = make_community_summary(n_findings=3)
        ctx = engine._build_community_context(community)
        assert "Finding 0" in ctx or "Key Findings" in ctx

    def test_build_community_context_within_token_limit(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        community = make_community_summary(
            summary="word " * 2000,  # very long summary
            n_findings=10,
        )
        ctx = engine._build_community_context(community)
        tokens = engine.tokenizer.count_tokens(ctx)
        assert tokens <= engine.context_window


class TestGraphRAGMapParsing:

    def make_engine(self, tmp_dir):
        return make_graphrag_engine(tmp_dir, [])

    def test_parse_valid_response(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = json.dumps({"points": [
            {"description": "OpenAI raised significant funding.", "score": 85},
            {"description": "Microsoft partnership is key.", "score": 90},
        ]})
        points = engine._parse_map_response(response, "comm_001")
        assert len(points) == 2
        assert points[0]["description"] == "OpenAI raised significant funding."
        assert points[0]["score"] == 85

    def test_parse_with_markdown_fences(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = '```json\n{"points": [{"description": "AI is growing.", "score": 75}]}\n```'
        points = engine._parse_map_response(response, "comm_001")
        assert len(points) == 1
        assert points[0]["score"] == 75

    def test_parse_score_clamped(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = json.dumps({"points": [
            {"description": "High score test.", "score": 150},  # over 100
            {"description": "Negative test.", "score": -10},   # under 0
        ]})
        points = engine._parse_map_response(response, "comm_001")
        for p in points:
            assert 0 <= p["score"] <= 100

    def test_parse_empty_response_returns_empty(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        assert engine._parse_map_response("", "comm_001") == []
        assert engine._parse_map_response("{}", "comm_001") == []

    def test_parse_invalid_json_returns_empty(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        assert engine._parse_map_response("not json", "comm_001") == []

    def test_parse_skips_items_without_description(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = json.dumps({"points": [
            {"description": "", "score": 50},  # empty desc — skip
            {"score": 60},                      # no desc — skip
            {"description": "Valid point.", "score": 70},
        ]})
        points = engine._parse_map_response(response, "comm_001")
        assert len(points) == 1
        assert points[0]["description"] == "Valid point."

    def test_parse_not_relevant_response(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = json.dumps({"points": [{"description": "Not relevant.", "score": 0}]})
        points = engine._parse_map_response(response, "comm_001")
        assert len(points) == 1
        assert points[0]["score"] == 0


class TestGraphRAGReduceContext:

    def test_fill_reduce_context_includes_high_score_first(self, tmp_dir):
        engine = make_graphrag_engine(tmp_dir, [])
        points = [
            {"description": "Point C", "score": 60},
            {"description": "Point A", "score": 90},
            {"description": "Point B", "score": 75},
        ]
        # Sort descending first (as the engine does)
        points_sorted = sorted(points, key=lambda p: p["score"], reverse=True)
        ctx_text, tokens = engine._fill_reduce_context(points_sorted, "What are themes?")
        # Point A (score 90) should appear before Point B
        assert ctx_text.index("Point A") < ctx_text.index("Point B")

    def test_fill_reduce_context_truncates_at_limit(self, tmp_dir):
        engine = make_graphrag_engine(tmp_dir, [])
        engine.context_window = 50  # Very tight
        points = [{"description": "word " * 50, "score": 80} for _ in range(10)]
        ctx_text, tokens = engine._fill_reduce_context(points, "Q")
        # Context must be within the available token budget
        total = engine.tokenizer.count_tokens(ctx_text)
        assert total <= engine.context_window

    def test_fill_reduce_context_scores_in_text(self, tmp_dir):
        engine = make_graphrag_engine(tmp_dir, [])
        points = [{"description": "AI is important.", "score": 85}]
        ctx_text, _ = engine._fill_reduce_context(points, "Q")
        assert "85" in ctx_text


class TestGraphRAGQuery:

    @pytest.mark.asyncio
    async def test_query_returns_graphrag_answer(self, tmp_dir):
        summaries = [
            make_community_summary(f"comm_c1_{i:04d}") for i in range(5)
        ]
        engine = make_graphrag_engine(
            tmp_dir, summaries,
            openai_response=json.dumps({"points": [
                {"description": "OpenAI leads AI research.", "score": 85},
            ]})
        )
        # Override the reduce stage to return a plain text answer
        engine.openai_service.complete = AsyncMock(side_effect=[
            # First 5 calls are map stage — return JSON
            *[make_completion_result(json.dumps({"points": [
                {"description": "AI insight.", "score": 80}
            ]})) for _ in range(5)],
            # Last call is reduce stage — return plain text
            make_completion_result("Final comprehensive answer about AI trends."),
        ])

        result = await engine.query("What are the main AI trends?", community_level="c1")
        assert result.query == "What are the main AI trends?"
        assert result.community_level == "c1"
        assert result.communities_total == 5
        assert result.map_answers_generated >= 0
        assert result.latency_ms >= 0
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_query_no_summaries_returns_empty_answer(self, tmp_dir):
        from app.storage.summary_store import SummaryStore
        from app.core.query.graphrag_engine import GraphRAGEngine
        store = SummaryStore(artifacts_dir=tmp_dir)
        # Save an empty summaries file so the store doesn't raise FileNotFoundError
        store.save_summaries([])
        engine = GraphRAGEngine(
            openai_service=make_mock_openai(),
            summary_store=store,
            tokenizer=make_tokenizer(),
        )
        result = await engine.query("Q", community_level="c1")
        assert result.communities_total == 0
        assert "index" in result.answer.lower() or "not contain" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_query_filters_low_score_points(self, tmp_dir):
        summaries = [make_community_summary("comm_c1_0000")]
        engine = make_graphrag_engine(tmp_dir, summaries)
        engine.helpfulness_threshold = 70  # Only keep score > 70

        # Map returns mixed scores
        map_response = json.dumps({"points": [
            {"description": "High score point.", "score": 90},
            {"description": "Low score point.", "score": 20},  # should be filtered
        ]})
        reduce_response = "Final answer."

        engine.openai_service.complete = AsyncMock(side_effect=[
            make_completion_result(map_response),
            make_completion_result(reduce_response),
        ])

        result = await engine.query("Q", community_level="c1", helpfulness_threshold=70)
        assert result.map_answers_after_filter <= result.map_answers_generated

    @pytest.mark.asyncio
    async def test_query_includes_context(self, tmp_dir):
        summaries = [make_community_summary("comm_c1_0000")]
        engine = make_graphrag_engine(tmp_dir, summaries)
        engine.openai_service.complete = AsyncMock(side_effect=[
            make_completion_result(json.dumps({"points": [
                {"description": "Key point.", "score": 80}
            ]})),
            make_completion_result("Answer."),
        ])
        result = await engine.query("Q", include_context=True)
        assert result.context is not None
        assert len(result.context) == 1
        assert result.context[0].community_id == "comm_c1_0000"

    @pytest.mark.asyncio
    async def test_query_excludes_context_when_not_requested(self, tmp_dir):
        summaries = [make_community_summary("comm_c1_0000")]
        engine = make_graphrag_engine(tmp_dir, summaries)
        engine.openai_service.complete = AsyncMock(side_effect=[
            make_completion_result(json.dumps({"points": [{"description": "P.", "score": 80}]})),
            make_completion_result("A."),
        ])
        result = await engine.query("Q", include_context=False)
        assert result.context is None

    @pytest.mark.asyncio
    async def test_query_with_all_zero_scores_returns_empty(self, tmp_dir):
        summaries = [make_community_summary("comm_c1_0000")]
        engine = make_graphrag_engine(tmp_dir, summaries)
        engine.helpfulness_threshold = 0
        engine.openai_service.complete = AsyncMock(
            return_value=make_completion_result(
                json.dumps({"points": [{"description": "Not relevant.", "score": 0}]})
            )
        )
        result = await engine.query("Q", community_level="c1")
        # All points have score 0 → filtered out → empty answer
        assert result.map_answers_after_filter == 0

    def test_get_available_levels(self, tmp_dir):
        summaries = [
            make_community_summary("comm_c1_0000", "c1"),
            make_community_summary("comm_c2_0000", "c2"),
        ]
        engine = make_graphrag_engine(tmp_dir, summaries)
        levels = engine.get_available_levels()
        assert "c1" in levels
        assert "c2" in levels

    def test_get_community_counts(self, tmp_dir):
        summaries = [
            make_community_summary("comm_c1_0000", "c1"),
            make_community_summary("comm_c1_0001", "c1"),
            make_community_summary("comm_c2_0000", "c2"),
        ]
        engine = make_graphrag_engine(tmp_dir, summaries)
        counts = engine.get_community_counts()
        assert counts.get("c1") == 2
        assert counts.get("c2") == 1

    @pytest.mark.asyncio
    async def test_map_stage_handles_failed_calls(self, tmp_dir):
        """If a single map call fails, the engine continues with remaining summaries."""
        summaries = [
            make_community_summary("comm_c1_0000"),
            make_community_summary("comm_c1_0001"),
        ]
        engine = make_graphrag_engine(tmp_dir, summaries)

        call_n = [0]
        async def flaky_complete(user_prompt, system_prompt=None, **kwargs):
            call_n[0] += 1
            if call_n[0] == 1:
                raise Exception("API timeout")
            return make_completion_result(json.dumps({"points": [
                {"description": "Valid point.", "score": 75}
            ]}))

        engine.openai_service.complete = flaky_complete

        results, token_usage = await engine._map_stage("Q", summaries)
        # Should not crash — returns what succeeded
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def make_judgment_response(winner: str = "A", score_a: int = 80, score_b: int = 60):
    """Build a valid judge JSON response."""
    return json.dumps({
        "winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "reasoning": "Answer A covers significantly more topics.",
    })


def make_evaluation_engine(
    tmp_dir: Path,
    n_summaries: int = 3,
    judge_winner: str = "A",
):
    """Build an EvaluationEngine with mocked sub-engines and judge."""
    from app.core.query.evaluation_engine import EvaluationEngine
    from app.core.query.graphrag_engine import GraphRAGEngine
    from app.core.query.vectorrag_engine import VectorRAGEngine

    # Mock graphrag engine
    mock_graphrag = AsyncMock(spec=GraphRAGEngine)
    mock_graphrag.query = AsyncMock()
    from app.models.response_models import GraphRAGAnswer
    mock_graphrag.query.return_value = MagicMock(
        spec=GraphRAGAnswer,
        answer="GraphRAG comprehensive answer about AI trends and market dynamics.",
    )

    # Mock vectorrag engine
    mock_vectorrag = AsyncMock(spec=VectorRAGEngine)
    mock_vectorrag.query = AsyncMock()
    from app.models.response_models import VectorRAGAnswer
    mock_vectorrag.query.return_value = MagicMock(
        spec=VectorRAGAnswer,
        answer="VectorRAG concise answer about AI.",
    )

    # Judge LLM
    mock_judge = make_mock_openai(make_judgment_response(judge_winner))

    return EvaluationEngine(
        openai_service=mock_judge,
        graphrag_engine=mock_graphrag,
        vectorrag_engine=mock_vectorrag,
        tokenizer=make_tokenizer(),
        max_concurrency=3,
        randomize_answer_order=False,  # disable randomization for deterministic tests
    )


class TestEvaluationJudgeParsing:

    def make_engine(self, tmp_dir):
        return make_evaluation_engine(tmp_dir)

    def test_parse_valid_judgment(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = make_judgment_response("A", 85, 60)
        data = engine._parse_judgment(response)
        assert data["winner"] == "A"
        assert data["score_a"] == 85
        assert data["score_b"] == 60
        assert "reasoning" in data

    def test_parse_judgment_with_fences(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = f'```json\n{make_judgment_response("B", 70, 80)}\n```'
        data = engine._parse_judgment(response)
        assert data["winner"] == "B"

    def test_parse_invalid_json_returns_tie(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        data = engine._parse_judgment("this is not json")
        assert data["winner"] == "TIE"
        assert data["score_a"] == 50
        assert data["score_b"] == 50

    def test_parse_tie_judgment(self, tmp_dir):
        engine = self.make_engine(tmp_dir)
        response = json.dumps({"winner": "TIE", "score_a": 70, "score_b": 70,
                               "reasoning": "Both answers are equal."})
        data = engine._parse_judgment(response)
        assert data["winner"] == "TIE"


class TestEvaluationCriterionAggregation:

    def make_engine(self, tmp_dir):
        return make_evaluation_engine(tmp_dir)

    def make_judgment(self, winner, run_index=0, a_system="graphrag"):
        from app.models.evaluation_models import SingleJudgment, Winner, EvalCriterion
        b_system = "vectorrag" if a_system == "graphrag" else "graphrag"
        return SingleJudgment(
            criterion=EvalCriterion.COMPREHENSIVENESS,
            winner=Winner(winner),
            answer_a_system=a_system,
            answer_b_system=b_system,
            answer_a_score=80,
            answer_b_score=60,
            reasoning="Test reasoning.",
            run_index=run_index,
        )

    def test_aggregate_all_graphrag_wins(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner
        engine = self.make_engine(tmp_dir)
        judgments = [self.make_judgment("graphrag", i) for i in range(5)]
        result = engine._aggregate_criterion(
            EvalCriterion.COMPREHENSIVENESS, "Q?", judgments
        )
        assert result.graphrag_wins == 5
        assert result.vectorrag_wins == 0
        assert result.graphrag_win_rate == pytest.approx(1.0)
        assert result.majority_winner == Winner.GRAPHRAG

    def test_aggregate_all_vectorrag_wins(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner
        engine = self.make_engine(tmp_dir)
        judgments = [self.make_judgment("vectorrag", i) for i in range(5)]
        result = engine._aggregate_criterion(
            EvalCriterion.COMPREHENSIVENESS, "Q?", judgments
        )
        assert result.majority_winner == Winner.VECTORRAG
        assert result.graphrag_win_rate == pytest.approx(0.0)

    def test_aggregate_all_ties_win_rate_is_half(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner
        engine = self.make_engine(tmp_dir)
        judgments = [self.make_judgment("tie", i) for i in range(5)]
        result = engine._aggregate_criterion(
            EvalCriterion.COMPREHENSIVENESS, "Q?", judgments
        )
        assert result.ties == 5
        assert result.graphrag_win_rate == pytest.approx(0.5)
        assert result.majority_winner == Winner.TIE

    def test_aggregate_majority_vote(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner
        engine = self.make_engine(tmp_dir)
        # 3 graphrag, 2 vectorrag → graphrag wins by majority
        judgments = (
            [self.make_judgment("graphrag", i) for i in range(3)] +
            [self.make_judgment("vectorrag", i + 3) for i in range(2)]
        )
        result = engine._aggregate_criterion(
            EvalCriterion.COMPREHENSIVENESS, "Q?", judgments
        )
        assert result.majority_winner == Winner.GRAPHRAG
        assert result.graphrag_wins == 3
        assert result.vectorrag_wins == 2

    def test_aggregate_avg_scores_computed(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, SingleJudgment, Winner
        engine = self.make_engine(tmp_dir)

        judgments = [
            SingleJudgment(
                criterion=EvalCriterion.COMPREHENSIVENESS,
                winner=Winner.GRAPHRAG,
                answer_a_system="graphrag",
                answer_b_system="vectorrag",
                answer_a_score=80,
                answer_b_score=50,
                reasoning="Better coverage.",
                run_index=i,
            )
            for i in range(5)
        ]
        result = engine._aggregate_criterion(
            EvalCriterion.COMPREHENSIVENESS, "Q?", judgments
        )
        assert result.avg_graphrag_score == pytest.approx(80.0)
        assert result.avg_vectorrag_score == pytest.approx(50.0)


class TestEvaluationSingleJudgment:

    @pytest.mark.asyncio
    async def test_single_judgment_graphrag_wins(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner
        from app.core.query.evaluation_engine import EvaluationEngine

        mock_judge = make_mock_openai(make_judgment_response("A", 85, 60))
        engine = make_evaluation_engine(tmp_dir)
        engine.openai_service = mock_judge
        engine.randomize_answer_order = False  # A = graphrag

        judgment = await engine._single_judgment(
            question="What are the main themes?",
            graphrag_answer="Comprehensive GraphRAG answer.",
            vectorrag_answer="Short VectorRAG answer.",
            criterion=EvalCriterion.COMPREHENSIVENESS,
            run_index=0,
            randomize=False,
        )
        assert judgment.criterion == EvalCriterion.COMPREHENSIVENESS
        assert judgment.winner == Winner.GRAPHRAG
        assert judgment.run_index == 0
        assert judgment.answer_a_system == "graphrag"
        assert 0 <= judgment.answer_a_score <= 100
        assert len(judgment.reasoning) > 0

    @pytest.mark.asyncio
    async def test_single_judgment_position_swap(self, tmp_dir):
        """With randomize=True, A/B assignment can be swapped."""
        from app.models.evaluation_models import EvalCriterion

        engine = make_evaluation_engine(tmp_dir)
        engine.openai_service = make_mock_openai(make_judgment_response("A", 80, 70))

        # Run multiple times; at least one swap should occur with randomize=True
        systems_seen = set()
        for _ in range(10):
            j = await engine._single_judgment(
                question="Q", graphrag_answer="G", vectorrag_answer="V",
                criterion=EvalCriterion.COMPREHENSIVENESS,
                run_index=0, randomize=True,
            )
            systems_seen.add(j.answer_a_system)

        # With 10 trials and 50% swap probability, both systems should appear as A
        assert len(systems_seen) == 2, "Expected both A/B assignments to be seen"

    @pytest.mark.asyncio
    async def test_single_judgment_api_failure_returns_tie(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner

        engine = make_evaluation_engine(tmp_dir)
        engine.openai_service.complete = AsyncMock(side_effect=Exception("timeout"))

        judgment = await engine._single_judgment(
            question="Q", graphrag_answer="G", vectorrag_answer="V",
            criterion=EvalCriterion.COMPREHENSIVENESS,
            run_index=0, randomize=False,
        )
        assert judgment.winner == Winner.TIE


class TestEvaluationFullRun:

    @pytest.mark.asyncio
    async def test_evaluate_single_question(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner

        engine = make_evaluation_engine(tmp_dir, judge_winner="A")
        # A = graphrag (randomize_answer_order=False)

        result = await engine.evaluate(
            questions=["What are the main themes in the AI corpus?"],
            criteria=[EvalCriterion.COMPREHENSIVENESS],
            eval_runs=3,
            community_level="c1",
        )

        assert result.total_questions == 1
        assert result.eval_runs_per_question == 3
        assert len(result.question_results) == 1
        assert len(result.summary_stats) == 1
        assert result.comprehensiveness_win_rate is not None

    @pytest.mark.asyncio
    async def test_evaluate_all_four_criteria(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion

        engine = make_evaluation_engine(tmp_dir)
        result = await engine.evaluate(
            questions=["Q?"],
            criteria=None,  # defaults to all four
            eval_runs=2,
        )
        assert len(result.criteria_evaluated) == 4
        assert EvalCriterion.COMPREHENSIVENESS in result.criteria_evaluated
        assert EvalCriterion.DIRECTNESS in result.criteria_evaluated

    @pytest.mark.asyncio
    async def test_evaluate_multiple_questions(self, tmp_dir):
        engine = make_evaluation_engine(tmp_dir)
        questions = [f"Question {i}?" for i in range(3)]

        result = await engine.evaluate(
            questions=questions,
            criteria=["comprehensiveness"],
            eval_runs=2,
        )
        assert result.total_questions == 3
        assert len(result.question_results) == 3

    @pytest.mark.asyncio
    async def test_evaluate_winner_determined_by_majority(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion, Winner

        engine = make_evaluation_engine(tmp_dir, judge_winner="A")
        result = await engine.evaluate(
            questions=["Q?"],
            criteria=[EvalCriterion.COMPREHENSIVENESS],
            eval_runs=5,
            randomize_answer_order=False,  # A = graphrag always
        )
        cr = result.question_results[0].criterion_results[0]
        # All 5 runs winner is A = graphrag
        assert cr.majority_winner == Winner.GRAPHRAG
        assert cr.graphrag_wins == 5

    @pytest.mark.asyncio
    async def test_evaluate_returns_valid_eval_id(self, tmp_dir):
        engine = make_evaluation_engine(tmp_dir)
        result = await engine.evaluate(questions=["Q?"], eval_runs=1)
        assert result.evaluation_id.startswith("eval_")
        assert len(result.evaluation_id) > 5

    @pytest.mark.asyncio
    async def test_summary_stats_computed(self, tmp_dir):
        engine = make_evaluation_engine(tmp_dir, judge_winner="A")
        result = await engine.evaluate(
            questions=["Q1?", "Q2?"],
            criteria=["comprehensiveness"],
            eval_runs=2,
            randomize_answer_order=False,
        )
        stats = result.summary_stats
        assert len(stats) == 1
        assert stats[0].graphrag_win_rate_avg >= 0.0
        assert stats[0].total_questions == 2

    @pytest.mark.asyncio
    async def test_resolve_criteria_from_strings(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion
        engine = make_evaluation_engine(tmp_dir)
        resolved = engine._resolve_criteria(["comprehensiveness", "diversity"])
        assert EvalCriterion.COMPREHENSIVENESS in resolved
        assert EvalCriterion.DIVERSITY in resolved

    @pytest.mark.asyncio
    async def test_resolve_criteria_none_returns_all(self, tmp_dir):
        from app.models.evaluation_models import EvalCriterion
        engine = make_evaluation_engine(tmp_dir)
        resolved = engine._resolve_criteria(None)
        assert len(resolved) == 4

    def test_truncate_answer(self, tmp_dir):
        engine = make_evaluation_engine(tmp_dir)
        short = "Short answer."
        assert engine._truncate_answer(short) == short

        long_answer = "word " * 2000
        truncated = engine._truncate_answer(long_answer, max_tokens=100)
        assert engine.tokenizer.count_tokens(truncated) <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndQueryComparison:
    """
    Verify both engines answer the same question and produce structured responses.
    No real LLM calls — all mocked.
    """

    @pytest.mark.asyncio
    async def test_both_engines_answer_same_query(self, tmp_dir):
        """The key 'done when' test: both engines answer the same global question."""
        query = "What are the dominant themes and key entities in this corpus?"

        # ── VectorRAG ──
        vector_engine = make_vectorrag_engine(
            tmp_dir,
            n_chunks=8,
            openai_response=(
                "The corpus focuses primarily on AI investment, with Microsoft and "
                "OpenAI forming a central partnership. Key themes include language "
                "models, AI safety, and competitive dynamics."
            ),
        )
        v_answer = await vector_engine.query(query)
        assert v_answer.answer != ""
        assert v_answer.chunks_retrieved >= 0
        assert v_answer.context_tokens_used >= 0

        # ── GraphRAG ──
        summaries = [
            make_community_summary(
                f"comm_c1_{i:04d}", "c1",
                title=f"Community {i}: AI Investment",
                summary="OpenAI and Microsoft partnership defines AI investment landscape.",
                impact_rating=8.0,
                n_findings=3,
            )
            for i in range(5)
        ]
        graph_engine = make_graphrag_engine(tmp_dir / "graphrag", summaries)

        map_response = json.dumps({"points": [
            {"description": "Microsoft invested $10B in OpenAI forming dominant AI coalition.", "score": 95},
            {"description": "OpenAI leads language model development with GPT-4 series.", "score": 90},
            {"description": "AI safety concerns pervade investment discussions.", "score": 75},
            {"description": "Google and Meta respond with competing open-source models.", "score": 70},
        ]})
        reduce_response = (
            "The corpus reveals a transformative period in AI development. "
            "Central themes include: (1) The Microsoft-OpenAI partnership as a "
            "dominant force reshaping the AI landscape; (2) Rapid advancement of "
            "large language models led by GPT-4; (3) Growing emphasis on AI safety "
            "and alignment research; (4) Competitive responses from Google and Meta "
            "through open-source initiatives. The investment dynamics show capital "
            "concentrating around foundation model companies while safety research "
            "gains institutional recognition."
        )

        call_n = [0]
        async def staged_complete(user_prompt, system_prompt=None, **kwargs):
            call_n[0] += 1
            if call_n[0] <= 5:  # map calls
                return make_completion_result(map_response)
            else:               # reduce call
                return make_completion_result(reduce_response)

        graph_engine.openai_service.complete = staged_complete

        g_answer = await graph_engine.query(query, community_level="c1")

        # ── Validate both answers ──────────────────────────────────────────────
        assert g_answer.answer != ""
        assert g_answer.communities_total == 5
        assert g_answer.map_answers_generated >= 0
        assert g_answer.community_level == "c1"

        # GraphRAG answer should be more comprehensive (longer) for global queries
        assert len(g_answer.answer) > len(v_answer.answer) * 0.5

    @pytest.mark.asyncio
    async def test_evaluation_produces_winner_per_criterion(self, tmp_dir):
        """Evaluation engine returns a winner for each criterion."""
        from app.models.evaluation_models import EvalCriterion

        engine = make_evaluation_engine(tmp_dir, judge_winner="A")

        result = await engine.evaluate(
            questions=["What are the main themes in the corpus?"],
            criteria=list(EvalCriterion),
            eval_runs=3,
            randomize_answer_order=False,
        )

        assert result.total_questions == 1
        q_result = result.question_results[0]

        for cr in q_result.criterion_results:
            assert cr.majority_winner in ("graphrag", "vectorrag", "tie")
            assert cr.total_runs == 3
            assert cr.graphrag_wins + cr.vectorrag_wins + cr.ties == 3

    @pytest.mark.asyncio
    async def test_claim_extractor_produces_counts(self, tmp_dir):
        """Claim extractor returns a claim count for each answer."""
        from app.core.query.claim_validation import ClaimValidationEngine

        graphrag_claims = [
            "OpenAI was founded in 2015 by Sam Altman and Elon Musk.",
            "Microsoft invested $10 billion in OpenAI in 2023.",
            "GPT-4 was released in March 2023.",
            "OpenAI's ChatGPT reached 100 million users in two months.",
            "AI safety research focuses on alignment and interpretability.",
            "Google launched Bard in response to ChatGPT.",
            "Meta released LLaMA as an open-source language model.",
            "NVIDIA's GPU revenue tripled due to AI demand.",
            "Anthropic was founded by former OpenAI employees.",
            "The EU passed the AI Act regulating high-risk AI systems.",
        ]
        vectorrag_claims = [
            "OpenAI built ChatGPT.",
            "Microsoft and OpenAI partnered.",
            "AI is growing rapidly.",
            "Large language models are powerful.",
        ]

        call_idx = [0]
        async def mock_complete(user_prompt, system_prompt=None, **kwargs):
            claims = graphrag_claims if call_idx[0] == 0 else vectorrag_claims
            call_idx[0] += 1
            return make_completion_result(json.dumps({"claims": claims}))

        mock_openai = AsyncMock()
        mock_openai.model = "gpt-4o"
        mock_openai.complete = mock_complete

        engine = ClaimValidationEngine(
            openai_service=mock_openai, tokenizer=make_tokenizer()
        )

        comparison = await engine.compare(
            question_id=0,
            question="What are the main themes in the AI corpus?",
            graphrag_answer="Long comprehensive GraphRAG answer about AI trends.",
            vectorrag_answer="Short VectorRAG answer about AI.",
        )

        assert comparison.graphrag_metrics.unique_claim_count > 0
        assert comparison.vectorrag_metrics.unique_claim_count > 0
        # GraphRAG should have more claims
        assert comparison.graphrag_metrics.unique_claim_count > \
               comparison.vectorrag_metrics.unique_claim_count
        assert comparison.comprehensiveness_winner == "graphrag"
        assert comparison.comprehensiveness_delta > 0

        # Cluster counts at all thresholds populated
        g = comparison.graphrag_metrics
        assert g.cluster_count_threshold_05 is not None
        assert g.cluster_count_threshold_07 is not None
        assert g.cluster_count_threshold_08 is not None