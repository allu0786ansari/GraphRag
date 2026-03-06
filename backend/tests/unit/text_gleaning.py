"""
tests/unit/test_gleaning.py — GleaningLoop unit tests.

Tests:
  - run() with max_rounds=0 returns initial extraction unchanged
  - run() stops when LLM answers NO to check prompt
  - run() calls parse_fn and merges new entities on YES
  - run() runs up to max_rounds
  - check_needs_gleaning() returns True on YES, False on NO
  - _merge_into() merges entities without duplicating
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


def make_extraction(chunk_id="chunk_001", n_entities=2, n_rels=1):
    from app.models.graph_models import (
        ChunkExtraction, ExtractedEntity, ExtractedRelationship
    )
    entities = [
        ExtractedEntity(
            name=f"Entity_{i}", entity_type="ORGANIZATION",
            description=f"Desc {i}", source_chunk_id=chunk_id,
        )
        for i in range(n_entities)
    ]
    rels = [
        ExtractedRelationship(
            source_entity=f"Entity_{i}", target_entity=f"Entity_{i+1}",
            description="related", strength=7, source_chunk_id=chunk_id,
        )
        for i in range(n_rels)
    ]
    return ChunkExtraction(
        chunk_id=chunk_id, entities=entities, relationships=rels,
        extraction_completed=True, gleaning_rounds_completed=0,
    )


def make_completion(content):
    from app.services.openai_service import CompletionResult
    return CompletionResult(
        content=content, model="gpt-4o",
        prompt_tokens=50, completion_tokens=10, total_tokens=60,
        finish_reason="stop",
    )


@pytest.fixture
def openai_svc():
    svc = MagicMock()
    svc.model = "gpt-4o"
    return svc


@pytest.fixture
def gleaner(openai_svc):
    from app.core.pipeline.gleaning import GleaningLoop
    return GleaningLoop(openai_service=openai_svc)


# ── run() ─────────────────────────────────────────────────────────────────────

class TestGleaningLoopRun:

    @pytest.mark.asyncio
    async def test_max_rounds_zero_returns_initial_unchanged(self, gleaner):
        initial = make_extraction(n_entities=3)
        history = [{"role": "user", "content": "extract"}, {"role": "assistant", "content": "result"}]
        result = await gleaner.run(
            chunk_text="some text", initial_extraction=initial,
            conversation_history=history, max_rounds=0,
        )
        assert result is initial
        assert len(result.entities) == 3

    @pytest.mark.asyncio
    async def test_stops_on_no_answer(self, gleaner, openai_svc):
        openai_svc.async_completion_with_logit_bias = AsyncMock(
            return_value=make_completion("NO")
        )
        initial = make_extraction(n_entities=2)
        history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        result = await gleaner.run(
            chunk_text="text", initial_extraction=initial,
            conversation_history=history, max_rounds=2,
        )
        # Should stop after round 1 (answer was NO)
        assert openai_svc.async_completion_with_logit_bias.call_count == 1
        # No continuation call needed
        assert not hasattr(openai_svc, '_continuation_called')

    @pytest.mark.asyncio
    async def test_yes_triggers_continuation_and_parse(self, gleaner, openai_svc):
        openai_svc.async_completion_with_logit_bias = AsyncMock(
            return_value=make_completion("YES")
        )
        continuation_text = '("entity"<|>NewEntity<|>PERSON<|>New person found)##<|COMPLETE|>'
        openai_svc.async_chat_completion = AsyncMock(
            return_value=make_completion(continuation_text)
        )
        initial = make_extraction(n_entities=2)
        history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]

        from app.core.pipeline.extraction import ExtractionPipeline, TUPLE_DELIM, RECORD_DELIM, COMPLETION_DELIM
        from app.services.tokenizer_service import TokenizerService
        tokenizer = TokenizerService(model="gpt-4o")
        pipeline = ExtractionPipeline(openai_service=openai_svc, tokenizer=tokenizer)

        result = await gleaner.run(
            chunk_text="text", initial_extraction=initial,
            conversation_history=history, max_rounds=1,
            parse_fn=pipeline._parse_extraction_output,
        )
        # Should have called async_chat_completion for continuation
        openai_svc.async_chat_completion.assert_called()
        assert result.gleaning_rounds_completed >= 1

    @pytest.mark.asyncio
    async def test_runs_up_to_max_rounds(self, gleaner, openai_svc):
        # Always answer YES so gleaning continues
        openai_svc.async_completion_with_logit_bias = AsyncMock(
            return_value=make_completion("YES")
        )
        openai_svc.async_chat_completion = AsyncMock(
            return_value=make_completion('("entity"<|>Extra<|>ORG<|>extra)##<|COMPLETE|>')
        )
        initial = make_extraction(n_entities=1)
        history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]

        from app.core.pipeline.extraction import ExtractionPipeline
        from app.services.tokenizer_service import TokenizerService
        tokenizer = TokenizerService(model="gpt-4o")
        pipeline = ExtractionPipeline(openai_service=openai_svc, tokenizer=tokenizer)

        result = await gleaner.run(
            chunk_text="text", initial_extraction=initial,
            conversation_history=history, max_rounds=3,
            parse_fn=pipeline._parse_extraction_output,
        )
        assert openai_svc.async_completion_with_logit_bias.call_count == 3
        assert result.gleaning_rounds_completed == 3


# ── check_needs_gleaning() ────────────────────────────────────────────────────

class TestCheckNeedsGleaning:

    @pytest.mark.asyncio
    async def test_returns_true_on_yes(self, gleaner, openai_svc):
        openai_svc.async_completion_with_logit_bias = AsyncMock(
            return_value=make_completion("YES")
        )
        history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        result = await gleaner.check_needs_gleaning(history)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_no(self, gleaner, openai_svc):
        openai_svc.async_completion_with_logit_bias = AsyncMock(
            return_value=make_completion("NO")
        )
        history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        result = await gleaner.check_needs_gleaning(history)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_ambiguous_answer(self, gleaner, openai_svc):
        openai_svc.async_completion_with_logit_bias = AsyncMock(
            return_value=make_completion("MAYBE")
        )
        history = []
        result = await gleaner.check_needs_gleaning(history)
        assert result is False


# ── _merge_into() ─────────────────────────────────────────────────────────────

class TestMergeInto:

    def test_merge_adds_new_entities(self):
        from app.core.pipeline.gleaning import _merge_into
        from app.models.graph_models import ChunkExtraction, ExtractedEntity

        base = make_extraction(n_entities=2)
        new_e = ExtractedEntity(
            name="NewCo", entity_type="ORGANIZATION",
            description="New company", source_chunk_id="c001",
        )
        new_ext = ChunkExtraction(
            chunk_id="c001", entities=[new_e], relationships=[],
            extraction_completed=True,
        )
        _merge_into(base, new_ext, round_num=1)
        names = {e.name for e in base.entities}
        assert "NewCo" in names

    def test_merge_does_not_duplicate_existing_entities(self):
        from app.core.pipeline.gleaning import _merge_into
        from app.models.graph_models import ChunkExtraction, ExtractedEntity

        base = make_extraction(n_entities=2)
        original_count = len(base.entities)
        duplicate = ExtractedEntity(
            name="Entity_0", entity_type="ORGANIZATION",
            description="Same entity", source_chunk_id="c001",
        )
        new_ext = ChunkExtraction(
            chunk_id="c001", entities=[duplicate], relationships=[],
            extraction_completed=True,
        )
        _merge_into(base, new_ext, round_num=1)
        # Should not add duplicate
        names = [e.name for e in base.entities]
        assert names.count("Entity_0") == 1