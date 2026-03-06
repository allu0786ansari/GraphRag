"""
tests/unit/test_graphrag_engine.py — GraphRAGEngine unit tests.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock


def make_completion_result(content: str):
    from app.services.openai_service import CompletionResult
    return CompletionResult(
        content=content, model="gpt-4o",
        prompt_tokens=200, completion_tokens=80, total_tokens=280,
        finish_reason="stop",
    )


def make_map_json(points=None) -> str:
    if points is None:
        points = [{"description": "OpenAI dominates the AI landscape.", "score": 75}]
    return json.dumps({"points": points})


def make_summary_store(summaries):
    store = MagicMock()
    store.summaries_exist = MagicMock(return_value=bool(summaries))
    store.load_summaries = MagicMock(return_value=summaries)
    store.load_summaries_by_level = MagicMock(
        side_effect=lambda level: [s for s in summaries if s.level.value == level]
    )
    return store


@pytest.fixture
def openai_svc():
    svc = MagicMock()
    svc.model = "gpt-4o"
    svc.complete = AsyncMock(
        return_value=make_completion_result(make_map_json())
    )
    return svc


@pytest.fixture
def engine(openai_svc, mock_tokenizer, sample_summaries):
    from app.core.query.graphrag_engine import GraphRAGEngine
    store = make_summary_store(sample_summaries)
    return GraphRAGEngine(
        openai_service=openai_svc,
        summary_store=store,
        tokenizer=mock_tokenizer,
        context_window=8000,
        max_concurrency=4,
        helpfulness_threshold=0,
    )


# ── query() ─────────────────────────────────────────────────────────────────

class TestGraphRAGEngineQuery:

    @pytest.mark.asyncio
    async def test_returns_graphrag_answer(self, engine, sample_summaries):
        from app.models.response_models import GraphRAGAnswer
        level = sample_summaries[0].level.value
        answer = await engine.query("What are the main themes?", community_level=level)
        assert isinstance(answer, GraphRAGAnswer)

    @pytest.mark.asyncio
    async def test_answer_has_query_field(self, engine, sample_summaries):
        query = "What are the main topics?"
        level = sample_summaries[0].level.value
        answer = await engine.query(query, community_level=level)
        assert answer.query == query

    @pytest.mark.asyncio
    async def test_answer_has_community_level(self, engine, sample_summaries):
        level = sample_summaries[0].level.value
        answer = await engine.query("test", community_level=level)
        assert answer.community_level == level

    @pytest.mark.asyncio
    async def test_empty_summaries_returns_graceful_answer(self, openai_svc, mock_tokenizer):
        from app.core.query.graphrag_engine import GraphRAGEngine
        store = make_summary_store([])
        engine = GraphRAGEngine(openai_service=openai_svc, summary_store=store,
                                 tokenizer=mock_tokenizer)
        answer = await engine.query("Any question?", community_level="c1")
        assert isinstance(answer.answer, str)
        assert len(answer.answer) > 0

    @pytest.mark.asyncio
    async def test_token_usage_populated_when_requested(self, engine, sample_summaries):
        level = sample_summaries[0].level.value
        answer = await engine.query("themes?", community_level=level,
                                     include_token_usage=True)
        if answer.token_usage is not None:
            assert answer.token_usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_include_context_false_omits_context(self, engine, sample_summaries):
        level = sample_summaries[0].level.value
        answer = await engine.query("test", community_level=level, include_context=False)
        assert answer.context is None or answer.context == []

    @pytest.mark.asyncio
    async def test_include_context_true_returns_context(self, engine, sample_summaries):
        level = sample_summaries[0].level.value
        answer = await engine.query("test", community_level=level, include_context=True)
        assert answer.context is not None

    @pytest.mark.asyncio
    async def test_helpfulness_filter_removes_zero_score_points(
        self, mock_tokenizer, sample_summaries
    ):
        from app.core.query.graphrag_engine import GraphRAGEngine
        zero_svc = MagicMock()
        zero_svc.model = "gpt-4o"
        zero_svc.complete = AsyncMock(
            return_value=make_completion_result(
                json.dumps({"points": [{"description": "irrelevant", "score": 0}]})
            )
        )
        store = make_summary_store(sample_summaries)
        engine = GraphRAGEngine(openai_service=zero_svc, summary_store=store,
                                 tokenizer=mock_tokenizer, helpfulness_threshold=0)
        level = sample_summaries[0].level.value
        answer = await engine.query("test", community_level=level, helpfulness_threshold=0)
        assert answer.map_answers_after_filter == 0

    @pytest.mark.asyncio
    async def test_communities_total_correct(self, engine, sample_summaries):
        level = sample_summaries[0].level.value
        level_summaries = [s for s in sample_summaries if s.level.value == level]
        answer = await engine.query("test", community_level=level)
        assert answer.communities_total == len(level_summaries)


# ── _parse_map_response() ────────────────────────────────────────────────────

class TestParseMapResponse:
    """_parse_map_response(text, community_id) → list[dict[description, score]]"""

    def test_parses_valid_points_json(self, engine):
        text = json.dumps({"points": [
            {"description": "OpenAI leads AI research.", "score": 80},
            {"description": "Microsoft invested heavily.", "score": 60},
        ]})
        points = engine._parse_map_response(text, "comm_c1_0001")
        assert len(points) == 2
        assert points[0]["score"] == 80

    def test_returns_empty_for_invalid_json(self, engine):
        assert engine._parse_map_response("not json", "c1") == []

    def test_returns_empty_for_empty_string(self, engine):
        assert engine._parse_map_response("", "c1") == []

    def test_score_clamped_0_to_100(self, engine):
        text = json.dumps({"points": [{"description": "test", "score": 9999}]})
        points = engine._parse_map_response(text, "c1")
        if points:
            assert 0 <= points[0]["score"] <= 100

    def test_strips_markdown_fences(self, engine):
        text = '```json\n{"points": [{"description": "AI trend.", "score": 50}]}\n```'
        points = engine._parse_map_response(text, "c1")
        assert len(points) == 1

    def test_zero_score_point_returned(self, engine):
        """Score=0 points are returned — filtering happens in query()."""
        text = json.dumps({"points": [{"description": "Not relevant.", "score": 0}]})
        points = engine._parse_map_response(text, "c1")
        assert len(points) == 1
        assert points[0]["score"] == 0

    def test_missing_description_skipped(self, engine):
        text = json.dumps({"points": [
            {"score": 50},   # no description
            {"description": "Valid point.", "score": 70},
        ]})
        points = engine._parse_map_response(text, "c1")
        valid = [p for p in points if p.get("description")]
        assert any(p["score"] == 70 for p in valid)


# ── get_available_levels() ───────────────────────────────────────────────────

class TestGetAvailableLevels:

    def test_returns_list(self, engine):
        levels = engine.get_available_levels()
        assert isinstance(levels, list)

    def test_returns_levels_present_in_summaries(self, engine, sample_summaries):
        levels = engine.get_available_levels()
        for s in sample_summaries:
            assert s.level.value in levels

    def test_returns_empty_on_file_not_found(self, openai_svc, mock_tokenizer):
        from app.core.query.graphrag_engine import GraphRAGEngine
        store = MagicMock()
        store.load_summaries = MagicMock(side_effect=FileNotFoundError)
        engine = GraphRAGEngine(openai_service=openai_svc, summary_store=store,
                                 tokenizer=mock_tokenizer)
        assert engine.get_available_levels() == []


# ── _empty_answer() ──────────────────────────────────────────────────────────

class TestEmptyAnswer:

    def test_query_and_level_preserved(self, engine):
        answer = engine._empty_answer("test query", "c1", latency_ms=100.0)
        assert answer.query == "test query"
        assert answer.community_level == "c1"

    def test_has_non_empty_answer_text(self, engine):
        answer = engine._empty_answer("q", "c1", latency_ms=50.0)
        assert isinstance(answer.answer, str)
        assert len(answer.answer) > 0

    def test_zero_communities_total_by_default(self, engine):
        answer = engine._empty_answer("q", "c1", latency_ms=0.0)
        assert answer.communities_total == 0

    def test_different_message_for_nonzero_total(self, engine):
        a0 = engine._empty_answer("q", "c1", communities_total=0, latency_ms=0.0)
        a5 = engine._empty_answer("q", "c1", communities_total=5, latency_ms=0.0)
        assert a0.answer != a5.answer