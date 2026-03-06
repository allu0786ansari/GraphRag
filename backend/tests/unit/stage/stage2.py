"""
tests/unit/test_stage2_models.py — Stage 2 data model tests.

Tests every model for:
  - Clean import (no circular imports)
  - .model_json_schema() produces correct shapes
  - Valid inputs are accepted
  - Invalid inputs raise ValidationError with clear messages
  - Validators enforce paper-exact constraints
  - __init__.py re-exports everything correctly
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS — verify no circular imports and __init__ re-exports work
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:

    def test_request_models_import(self):
        from app.models import (
            IndexRequest, QueryRequest, EvalRequest,
            QuestionGenRequest, QueryMode, RequestCommunityLevel,
        )
        assert IndexRequest is not None

    def test_response_models_import(self):
        from app.models import (
            IndexResponse, IndexStatusResponse, QueryResponse,
            GraphRAGAnswer, VectorRAGAnswer, ErrorResponse,
            TokenUsage, PipelineStage, PipelineStatus,
        )
        assert QueryResponse is not None

    def test_graph_models_import(self):
        from app.models import (
            ChunkSchema, ChunkExtraction, ExtractedEntity,
            ExtractedRelationship, NodeSchema, EdgeSchema,
            CommunitySchema, CommunitySummary, PipelineArtifacts,
        )
        assert NodeSchema is not None

    def test_evaluation_models_import(self):
        from app.models import (
            EvalResponse, CriterionResult, SingleJudgment,
            ClaimMetrics, ClaimEvalResponse, Winner, EvalCriterion,
        )
        assert EvalResponse is not None

    def test_no_circular_imports(self):
        """All four model files must import independently."""
        import importlib
        for module in [
            "app.models.request_models",
            "app.models.response_models",
            "app.models.graph_models",
            "app.models.evaluation_models",
        ]:
            mod = importlib.import_module(module)
            assert mod is not None

    def test_init_exports_all_symbols(self):
        import app.models as m
        from app.models import __all__
        for symbol in __all__:
            assert hasattr(m, symbol), f"__all__ lists '{symbol}' but it's not importable"


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndexRequest:

    def test_default_values_match_paper(self):
        from app.models import IndexRequest
        req = IndexRequest()
        assert req.chunk_size == 600
        assert req.chunk_overlap == 100
        assert req.gleaning_rounds == 2
        assert req.context_window_size == 8000
        assert req.max_community_levels == 3
        assert req.force_reindex is False
        assert req.skip_claims is False
        assert req.max_chunks is None

    def test_valid_custom_values(self):
        from app.models import IndexRequest
        req = IndexRequest(
            chunk_size=400,
            chunk_overlap=50,
            gleaning_rounds=1,
            context_window_size=4000,
            force_reindex=True,
            max_chunks=100,
        )
        assert req.chunk_size == 400
        assert req.max_chunks == 100

    def test_overlap_must_be_less_than_chunk_size(self):
        from app.models import IndexRequest
        with pytest.raises(ValidationError, match="chunk_overlap"):
            IndexRequest(chunk_size=200, chunk_overlap=200)

    def test_overlap_equal_to_chunk_size_fails(self):
        from app.models import IndexRequest
        with pytest.raises(ValidationError):
            IndexRequest(chunk_size=300, chunk_overlap=300)

    def test_chunk_size_bounds(self):
        from app.models import IndexRequest
        with pytest.raises(ValidationError):
            IndexRequest(chunk_size=50)   # below min 100
        with pytest.raises(ValidationError):
            IndexRequest(chunk_size=9999) # above max 2400

    def test_gleaning_rounds_bounds(self):
        from app.models import IndexRequest
        with pytest.raises(ValidationError):
            IndexRequest(gleaning_rounds=10)  # above max 5

    def test_json_schema_has_correct_keys(self):
        from app.models import IndexRequest
        schema = IndexRequest.model_json_schema()
        props = schema["properties"]
        assert "chunk_size" in props
        assert "chunk_overlap" in props
        assert "gleaning_rounds" in props
        assert "context_window_size" in props
        assert "force_reindex" in props
        assert "max_chunks" in props


class TestQueryRequest:

    def test_valid_minimal_request(self):
        from app.models import QueryRequest
        req = QueryRequest(query="What are the main themes in this dataset?")
        assert req.query == "What are the main themes in this dataset?"
        assert req.mode.value == "both"
        assert req.community_level.value == "c1"
        assert req.top_k == 10
        assert req.helpfulness_threshold == 0

    def test_all_modes_valid(self):
        from app.models import QueryRequest, QueryMode
        for mode in QueryMode:
            req = QueryRequest(query="test query here", mode=mode)
            assert req.mode == mode

    def test_all_community_levels_valid(self):
        from app.models import QueryRequest, RequestCommunityLevel
        for level in RequestCommunityLevel:
            req = QueryRequest(query="test query here", community_level=level)
            assert req.community_level == level

    def test_query_too_short_fails(self):
        from app.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="Hi")  # below min_length=3

    def test_query_too_long_fails(self):
        from app.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 2001)  # above max_length=2000

    def test_top_k_bounds(self):
        from app.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query text", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query text", top_k=200)

    def test_helpfulness_threshold_bounds(self):
        from app.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query text", helpfulness_threshold=101)

    def test_json_schema_shape(self):
        from app.models import QueryRequest
        schema = QueryRequest.model_json_schema()
        props = schema["properties"]
        for key in ["query", "mode", "community_level", "top_k", "include_context"]:
            assert key in props


class TestEvalRequest:

    def test_valid_minimal_request(self):
        from app.models import EvalRequest
        req = EvalRequest(questions=["What are the main themes in this news dataset?"])
        assert len(req.questions) == 1
        assert req.eval_runs == 5
        assert req.randomize_answer_order is True
        assert len(req.criteria) == 4

    def test_questions_are_stripped(self):
        from app.models import EvalRequest
        req = EvalRequest(questions=["  What are the themes?  "])
        assert req.questions[0] == "What are the themes?"

    def test_empty_question_fails(self):
        from app.models import EvalRequest
        with pytest.raises(ValidationError, match="empty"):
            EvalRequest(questions=["   "])

    def test_too_short_question_fails(self):
        from app.models import EvalRequest
        with pytest.raises(ValidationError, match="too short"):
            EvalRequest(questions=["Hi?"])

    def test_duplicate_criteria_fails(self):
        from app.models import EvalRequest, EvalCriterion
        with pytest.raises(ValidationError, match="Duplicate"):
            EvalRequest(
                questions=["What are the themes?"],
                criteria=[EvalCriterion.COMPREHENSIVENESS, EvalCriterion.COMPREHENSIVENESS],
            )

    def test_max_125_questions(self):
        from app.models import EvalRequest
        with pytest.raises(ValidationError):
            EvalRequest(questions=["valid question text here?"] * 126)

    def test_eval_runs_bounds(self):
        from app.models import EvalRequest
        with pytest.raises(ValidationError):
            EvalRequest(questions=["valid question?"], eval_runs=11)

    def test_all_four_criteria_default(self):
        from app.models import EvalRequest, EvalCriterion
        req = EvalRequest(questions=["What are the themes?"])
        criteria_values = {c.value for c in req.criteria}
        assert criteria_values == {
            "comprehensiveness", "diversity", "empowerment", "directness"
        }


class TestQuestionGenRequest:

    def test_default_values_match_paper(self):
        from app.models import QuestionGenRequest
        req = QuestionGenRequest(
            corpus_description="A collection of news articles covering technology and health."
        )
        assert req.num_personas == 5
        assert req.num_tasks_per_persona == 5
        assert req.num_questions_per_task == 5
        assert req.total_questions == 125

    def test_total_questions_computed_correctly(self):
        from app.models import QuestionGenRequest
        req = QuestionGenRequest(
            corpus_description="A collection of news articles.",
            num_personas=3,
            num_tasks_per_persona=4,
            num_questions_per_task=2,
        )
        assert req.total_questions == 24

    def test_corpus_description_too_short(self):
        from app.models import QuestionGenRequest
        with pytest.raises(ValidationError):
            QuestionGenRequest(corpus_description="short")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenUsage:

    def test_zero_factory(self):
        from app.models import TokenUsage
        t = TokenUsage.zero()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total_tokens == 0

    def test_addition(self):
        from app.models import TokenUsage
        a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        b = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        c = a + b
        assert c.prompt_tokens == 300
        assert c.completion_tokens == 150
        assert c.total_tokens == 450

    def test_addition_with_cost(self):
        from app.models import TokenUsage
        a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost_usd=0.01)
        b = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, estimated_cost_usd=0.02)
        c = a + b
        assert abs(c.estimated_cost_usd - 0.03) < 1e-9

    def test_json_schema(self):
        from app.models import TokenUsage
        schema = TokenUsage.model_json_schema()
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            assert key in schema["properties"]


class TestQueryResponse:

    def test_valid_both_mode_response(self):
        from app.models import (
            QueryResponse, GraphRAGAnswer, VectorRAGAnswer, TokenUsage
        )
        from datetime import datetime

        vr = VectorRAGAnswer(
            answer="VectorRAG answer text",
            query="What are the themes?",
            chunks_retrieved=10,
            context_tokens_used=4000,
            latency_ms=1200.0,
        )
        gr = GraphRAGAnswer(
            answer="GraphRAG answer text",
            query="What are the themes?",
            community_level="c1",
            communities_total=555,
            communities_used_in_map=555,
            map_answers_generated=28,
            map_answers_after_filter=25,
            context_tokens_used=352000,
            latency_ms=8500.0,
        )
        resp = QueryResponse(
            query="What are the themes?",
            mode="both",
            request_id="req_test_001",
            graphrag=gr,
            vectorrag=vr,
            total_latency_ms=9700.0,
        )
        assert resp.graphrag.answer == "GraphRAG answer text"
        assert resp.vectorrag.answer == "VectorRAG answer text"

    def test_graphrag_only_mode(self):
        from app.models import QueryResponse, GraphRAGAnswer
        gr = GraphRAGAnswer(
            answer="Answer",
            query="Q",
            community_level="c0",
            communities_total=55,
            communities_used_in_map=55,
            map_answers_generated=5,
            map_answers_after_filter=4,
            context_tokens_used=39000,
            latency_ms=3000.0,
        )
        resp = QueryResponse(
            query="Q", mode="graphrag", request_id="req_002",
            graphrag=gr, total_latency_ms=3100.0,
        )
        assert resp.vectorrag is None
        assert resp.graphrag is not None


class TestErrorResponse:

    def test_valid_error_response(self):
        from app.models import ErrorResponse
        err = ErrorResponse(
            error="not_indexed",
            message="Run POST /api/v1/index first.",
            request_id="req_123",
        )
        assert err.error == "not_indexed"
        assert err.details is None

    def test_json_schema(self):
        from app.models import ErrorResponse
        schema = ErrorResponse.model_json_schema()
        for key in ["error", "message", "request_id", "details"]:
            assert key in schema["properties"]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkSchema:

    def test_valid_chunk(self):
        from app.models import ChunkSchema
        chunk = ChunkSchema(
            chunk_id="news_001_0000",
            source_document="news_001.json",
            text="OpenAI was founded in 2015 by Sam Altman and Elon Musk.",
            token_count=12,
            start_char=0,
            end_char=54,
            chunk_index=0,
            total_chunks_in_doc=10,
        )
        assert chunk.chunk_id == "news_001_0000"
        assert chunk.metadata == {}

    def test_json_schema(self):
        from app.models import ChunkSchema
        schema = ChunkSchema.model_json_schema()
        for key in ["chunk_id", "source_document", "text", "token_count"]:
            assert key in schema["properties"]


class TestExtractedEntity:

    def test_valid_entity(self):
        from app.models import ExtractedEntity
        e = ExtractedEntity(
            name="OpenAI",
            entity_type="ORGANIZATION",
            description="An AI research company founded in 2015.",
            source_chunk_id="news_001_0000",
        )
        assert e.extraction_round == 0
        assert e.name == "OpenAI"

    def test_gleaning_round_tracked(self):
        from app.models import ExtractedEntity
        e = ExtractedEntity(
            name="Sam Altman",
            entity_type="PERSON",
            description="CEO of OpenAI.",
            source_chunk_id="news_001_0000",
            extraction_round=1,
        )
        assert e.extraction_round == 1


class TestExtractedRelationship:

    def test_valid_relationship(self):
        from app.models import ExtractedRelationship
        r = ExtractedRelationship(
            source_entity="Microsoft",
            target_entity="OpenAI",
            description="Microsoft invested $10B in OpenAI in 2023.",
            strength=9,
            source_chunk_id="news_001_0000",
        )
        assert r.strength == 9
        assert r.extraction_round == 0

    def test_strength_bounds(self):
        from app.models import ExtractedRelationship
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                source_entity="A", target_entity="B",
                description="test", strength=0,   # below min 1
                source_chunk_id="chunk_0"
            )
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                source_entity="A", target_entity="B",
                description="test", strength=11,  # above max 10
                source_chunk_id="chunk_0"
            )


class TestNodeSchema:

    def test_valid_node(self):
        from app.models import NodeSchema
        node = NodeSchema(
            node_id="openai",
            name="OpenAI",
            entity_type="ORGANIZATION",
            description="AI research company known for GPT-4.",
            degree=42,
        )
        assert node.community_ids == {}
        assert node.claims == []
        assert node.mention_count == 1

    def test_json_schema_has_required_fields(self):
        from app.models import NodeSchema
        schema = NodeSchema.model_json_schema()
        for key in ["node_id", "name", "entity_type", "description", "degree"]:
            assert key in schema["properties"]


class TestEdgeSchema:

    def test_valid_edge(self):
        from app.models import EdgeSchema
        edge = EdgeSchema(
            edge_id="microsoft__openai",
            source_node_id="microsoft",
            target_node_id="openai",
            description="Microsoft invested in OpenAI.",
            weight=7.0,
            combined_degree=84,
        )
        assert edge.weight == 7.0
        assert edge.source_chunk_ids == []

    def test_weight_non_negative(self):
        from app.models import EdgeSchema
        with pytest.raises(ValidationError):
            EdgeSchema(
                edge_id="a__b",
                source_node_id="a",
                target_node_id="b",
                description="test",
                weight=-1.0,
            )


class TestCommunitySchema:

    def test_valid_community(self):
        from app.models import CommunitySchema, GraphCommunityLevel
        comm = CommunitySchema(
            community_id="comm_c1_0045",
            level=GraphCommunityLevel.C1,
            level_index=45,
            node_ids=["openai", "microsoft", "sam_altman"],
        )
        assert comm.parent_community_id is None
        assert comm.child_community_ids == []

    def test_node_ids_must_not_be_empty(self):
        from app.models import CommunitySchema, GraphCommunityLevel
        with pytest.raises(ValidationError):
            CommunitySchema(
                community_id="comm_c1_0001",
                level=GraphCommunityLevel.C1,
                level_index=1,
                node_ids=[],  # empty — violates min_length=1
            )


class TestCommunitySummary:

    def test_valid_summary(self):
        from app.models import CommunitySummary, CommunityFinding, GraphCommunityLevel
        summary = CommunitySummary(
            community_id="comm_c1_0045",
            level=GraphCommunityLevel.C1,
            title="Microsoft-OpenAI AI Investment Ecosystem",
            summary="This community covers the AI investment landscape.",
            impact_rating=7.5,
            rating_explanation="Central to AI industry dynamics.",
            findings=[
                CommunityFinding(
                    finding_id=0,
                    summary="Microsoft invested $10B in OpenAI.",
                    explanation="This investment made Microsoft the primary commercial partner.",
                )
            ],
            node_ids=["openai", "microsoft"],
            context_tokens_used=6500,
        )
        assert summary.impact_rating == 7.5
        assert len(summary.findings) == 1

    def test_empty_findings_fails(self):
        from app.models import CommunitySummary, GraphCommunityLevel
        with pytest.raises(ValidationError):
            CommunitySummary(
                community_id="comm_c1_0001",
                level=GraphCommunityLevel.C1,
                title="Test",
                summary="Test summary.",
                impact_rating=5.0,
                rating_explanation="Test.",
                findings=[],  # empty — violates min_length=1
                node_ids=["node_a"],
                context_tokens_used=1000,
            )

    def test_impact_rating_bounds(self):
        from app.models import CommunitySummary, CommunityFinding, GraphCommunityLevel
        finding = CommunityFinding(
            finding_id=0, summary="test", explanation="test explanation"
        )
        with pytest.raises(ValidationError):
            CommunitySummary(
                community_id="c", level=GraphCommunityLevel.C1,
                title="T", summary="S", impact_rating=11.0,  # above 10
                rating_explanation="R", findings=[finding],
                node_ids=["a"], context_tokens_used=100,
            )


class TestPipelineArtifacts:

    def test_not_indexed_by_default(self):
        from app.models import PipelineArtifacts
        artifacts = PipelineArtifacts()
        assert artifacts.is_fully_indexed is False
        assert artifacts.graphrag_ready is False
        assert artifacts.vectorrag_ready is False

    def test_graphrag_ready_when_summaries_exist(self):
        from app.models import PipelineArtifacts
        artifacts = PipelineArtifacts(community_summaries_exists=True)
        assert artifacts.graphrag_ready is True
        assert artifacts.is_fully_indexed is False  # still missing other artifacts

    def test_fully_indexed(self):
        from app.models import PipelineArtifacts
        artifacts = PipelineArtifacts(
            chunks_exists=True,
            extractions_exists=True,
            graph_exists=True,
            community_map_exists=True,
            community_summaries_exists=True,
            faiss_index_exists=True,
            embeddings_exists=True,
        )
        assert artifacts.is_fully_indexed is True
        assert artifacts.graphrag_ready is True
        assert artifacts.vectorrag_ready is True


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleJudgment:

    def test_valid_judgment(self):
        from app.models import SingleJudgment, Winner, EvalCriterion
        j = SingleJudgment(
            criterion=EvalCriterion.COMPREHENSIVENESS,
            winner=Winner.GRAPHRAG,
            answer_a_system="graphrag",
            answer_b_system="vectorrag",
            answer_a_score=85,
            answer_b_score=62,
            reasoning="Answer A covers more aspects of the question.",
            run_index=0,
        )
        assert j.winner == Winner.GRAPHRAG
        assert j.answer_a_score == 85

    def test_score_bounds(self):
        from app.models import SingleJudgment, Winner, EvalCriterion
        with pytest.raises(ValidationError):
            SingleJudgment(
                criterion=EvalCriterion.DIVERSITY,
                winner=Winner.TIE,
                answer_a_system="graphrag",
                answer_b_system="vectorrag",
                answer_a_score=101,   # above 100
                answer_b_score=50,
                reasoning="test",
                run_index=0,
            )


class TestCriterionResult:

    def test_win_rate_computed_correctly(self):
        from app.models import CriterionResult, SingleJudgment, Winner, EvalCriterion

        def make_judgment(winner, run_index):
            return SingleJudgment(
                criterion=EvalCriterion.COMPREHENSIVENESS,
                winner=winner,
                answer_a_system="graphrag",
                answer_b_system="vectorrag",
                answer_a_score=80 if winner == Winner.GRAPHRAG else 60,
                answer_b_score=60 if winner == Winner.GRAPHRAG else 80,
                reasoning="test",
                run_index=run_index,
            )

        result = CriterionResult(
            criterion=EvalCriterion.COMPREHENSIVENESS,
            question="What are the main themes?",
            judgments=[
                make_judgment(Winner.GRAPHRAG, 0),
                make_judgment(Winner.GRAPHRAG, 1),
                make_judgment(Winner.GRAPHRAG, 2),
                make_judgment(Winner.VECTORRAG, 3),
                make_judgment(Winner.TIE, 4),
            ],
            graphrag_wins=3,
            vectorrag_wins=1,
            ties=1,
            total_runs=5,
            graphrag_win_rate=0.0,   # will be recomputed by validator
            avg_graphrag_score=80.0,
            avg_vectorrag_score=65.0,
            majority_winner=Winner.TIE,  # will be recomputed
        )
        # 3 graphrag wins / (3+1) decisive = 0.75
        assert result.graphrag_win_rate == 0.75
        assert result.majority_winner == Winner.GRAPHRAG

    def test_all_ties_gives_50pct(self):
        from app.models import CriterionResult, SingleJudgment, Winner, EvalCriterion
        result = CriterionResult(
            criterion=EvalCriterion.DIRECTNESS,
            question="Q?",
            judgments=[],
            graphrag_wins=0,
            vectorrag_wins=0,
            ties=5,
            total_runs=5,
            graphrag_win_rate=0.0,
            avg_graphrag_score=70.0,
            avg_vectorrag_score=70.0,
            majority_winner=Winner.TIE,
        )
        assert result.graphrag_win_rate == 0.5
        assert result.majority_winner == Winner.TIE


class TestClaimMetrics:

    def test_valid_claim_metrics(self):
        from app.models import ClaimMetrics, ExtractedClaim

        claim = ExtractedClaim(
            claim_id="q001_graphrag_claim_000",
            text="OpenAI was founded in 2015.",
            source_system="graphrag",
            question_id=0,
        )

        metrics = ClaimMetrics(
            question_id=0,
            question="What are the main themes?",
            system="graphrag",
            answer="GraphRAG answer here",
            claims=[claim],
            unique_claim_count=1,
            cluster_count_threshold_07=1,
        )

        assert metrics.unique_claim_count == 1
        assert metrics.cluster_count_threshold_05 is None  # not set


class TestWinnerEnum:

    def test_all_values(self):
        from app.models import Winner
        assert Winner.GRAPHRAG.value == "graphrag"
        assert Winner.VECTORRAG.value == "vectorrag"
        assert Winner.TIE.value == "tie"


class TestEvalCriterionEnum:

    def test_all_values_match_paper(self):
        from app.models import EvalCriterion
        values = {c.value for c in EvalCriterion}
        assert values == {"comprehensiveness", "diversity", "empowerment", "directness"}


# ═══════════════════════════════════════════════════════════════════════════════
# JSON SCHEMA SHAPE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonSchemas:
    """
    Verify .model_json_schema() produces sensible output for all major models.
    This catches misconfigured Field() definitions and missing type annotations.
    """

    @pytest.mark.parametrize("model_name,expected_props", [
        ("IndexRequest", ["chunk_size", "chunk_overlap", "gleaning_rounds"]),
        ("QueryRequest", ["query", "mode", "community_level", "top_k"]),
        ("EvalRequest", ["questions", "criteria", "eval_runs"]),
        ("NodeSchema", ["node_id", "name", "entity_type", "description", "degree"]),
        ("EdgeSchema", ["edge_id", "source_node_id", "target_node_id", "weight"]),
        ("CommunitySchema", ["community_id", "level", "node_ids"]),
        ("CommunitySummary", ["community_id", "title", "summary", "impact_rating", "findings"]),
        ("ChunkSchema", ["chunk_id", "source_document", "text", "token_count"]),
        ("EvalResponse", ["evaluation_id", "total_questions", "question_results"]),
        ("QueryResponse", ["query", "mode", "request_id", "graphrag", "vectorrag"]),
        ("ErrorResponse", ["error", "message", "request_id"]),
    ])
    def test_schema_has_expected_properties(self, model_name, expected_props):
        import app.models as m
        model_cls = getattr(m, model_name)
        schema = model_cls.model_json_schema()
        props = schema.get("properties", {})
        for prop in expected_props:
            assert prop in props, f"{model_name}.model_json_schema() missing property '{prop}'"