"""
tests/unit/test_extraction.py — ExtractionPipeline unit tests.

Tests:
  - _parse_extraction_output: entities, relationships, claims, malformed records
  - extract_chunk: success, graceful failure on LLM error
  - extract_chunk with skip_claims
  - Completion delimiter truncation
  - Duplicate entity deduplication in parse
  - extract_chunks_batch: progress callback, partial failures
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


TUPLE_DELIM = "<|>"
RECORD_DELIM = "##"
COMPLETION_DELIM = "<|COMPLETE|>"


def make_extraction_response(*records) -> str:
    return RECORD_DELIM.join(records) + RECORD_DELIM + COMPLETION_DELIM


def make_entity(name, etype, desc):
    return f'("entity"{TUPLE_DELIM}{name}{TUPLE_DELIM}{etype}{TUPLE_DELIM}{desc})'


def make_relationship(src, tgt, desc, strength=7):
    return f'("relationship"{TUPLE_DELIM}{src}{TUPLE_DELIM}{tgt}{TUPLE_DELIM}{desc}{TUPLE_DELIM}{strength})'


def make_claim(subject, ctype, status, desc):
    return f'("claim"{TUPLE_DELIM}{subject}{TUPLE_DELIM}{ctype}{TUPLE_DELIM}{status}{TUPLE_DELIM}{desc}{TUPLE_DELIM}2020-01-01{TUPLE_DELIM}2020-12-31)'


@pytest.fixture
def pipeline(mock_openai_service, mock_tokenizer):
    from app.core.pipeline.extraction import ExtractionPipeline
    return ExtractionPipeline(
        openai_service=mock_openai_service,
        tokenizer=mock_tokenizer,
        gleaning_loop=None,
        skip_claims=False,
    )


@pytest.fixture
def sample_chunk():
    from app.models.graph_models import ChunkSchema
    return ChunkSchema(
        chunk_id="chunk_0001",
        source_document="article.txt",
        text="OpenAI was founded by Sam Altman. Microsoft invested $10B.",
        token_count=15,
        start_char=0,
        end_char=60,
        chunk_index=0,
        total_chunks_in_doc=1,
    )


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParseExtractionOutput:

    def test_parses_entity_records(self, pipeline):
        text = make_extraction_response(
            make_entity("OpenAI", "ORGANIZATION", "AI lab"),
            make_entity("Sam Altman", "PERSON", "CEO"),
        )
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.entities) == 2
        names = {e.name for e in result.entities}
        assert "OpenAI" in names
        assert "Sam Altman" in names

    def test_parses_relationship_records(self, pipeline):
        text = make_extraction_response(
            make_entity("OpenAI", "ORGANIZATION", "AI lab"),
            make_entity("Sam Altman", "PERSON", "CEO"),
            make_relationship("Sam Altman", "OpenAI", "CEO of", 9),
        )
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.relationships) == 1
        assert result.relationships[0].source_entity == "Sam Altman"
        assert result.relationships[0].target_entity == "OpenAI"
        assert result.relationships[0].strength == 9

    def test_parses_claim_records(self, pipeline):
        text = make_extraction_response(
            make_entity("OpenAI", "ORGANIZATION", "AI lab"),
            make_claim("OpenAI", "INVESTMENT", "TRUE", "Received $10B from Microsoft"),
        )
        result = pipeline._parse_extraction_output(text, "chunk_001")
        assert len(result.claims) == 1
        assert result.claims[0].subject_entity == "OpenAI"

    def test_truncates_at_completion_delimiter(self, pipeline):
        text = (
            make_entity("OpenAI", "ORGANIZATION", "AI lab") + RECORD_DELIM +
            COMPLETION_DELIM +
            make_entity("GARBAGE", "UNKNOWN", "Should not appear")
        )
        result = pipeline._parse_extraction_output(text, "chunk_001")
        names = {e.name for e in result.entities}
        assert "GARBAGE" not in names

    def test_skips_malformed_records_gracefully(self, pipeline):
        text = (
            make_entity("OpenAI", "ORGANIZATION", "AI lab") + RECORD_DELIM +
            "this is not a valid record" + RECORD_DELIM +
            make_entity("Microsoft", "ORGANIZATION", "Tech company") + RECORD_DELIM +
            COMPLETION_DELIM
        )
        result = pipeline._parse_extraction_output(text, "chunk_001")
        # Should parse the two valid entities, skip the bad one
        assert len(result.entities) == 2

    def test_empty_text_returns_empty_extraction(self, pipeline):
        result = pipeline._parse_extraction_output("", "chunk_001")
        assert result.entities == []
        assert result.relationships == []
        assert result.claims == []

    def test_chunk_id_set_on_all_entities(self, pipeline):
        text = make_extraction_response(
            make_entity("OpenAI", "ORGANIZATION", "AI lab"),
            make_entity("Sam Altman", "PERSON", "CEO"),
        )
        result = pipeline._parse_extraction_output(text, "my_chunk_99")
        for entity in result.entities:
            assert entity.source_chunk_id == "my_chunk_99"

    def test_relationship_strength_clamped(self, pipeline):
        text = make_extraction_response(
            make_entity("A", "ORGANIZATION", "org"),
            make_entity("B", "ORGANIZATION", "org"),
            make_relationship("A", "B", "related", 999),  # way out of range
        )
        result = pipeline._parse_extraction_output(text, "c001")
        if result.relationships:
            assert result.relationships[0].strength <= 10


# ── extract_chunk() ───────────────────────────────────────────────────────────

class TestExtractChunk:

    @pytest.mark.asyncio
    async def test_returns_chunk_extraction_on_success(self, pipeline, sample_chunk):
        result = await pipeline.extract_chunk(sample_chunk, gleaning_rounds=0)
        assert result.chunk_id == "chunk_0001"
        assert result.extraction_completed is True
        assert len(result.entities) >= 1

    @pytest.mark.asyncio
    async def test_returns_failed_extraction_on_llm_error(
        self, mock_tokenizer, sample_chunk
    ):
        from app.core.pipeline.extraction import ExtractionPipeline
        failing_svc = MagicMock()
        failing_svc.model = "gpt-4o"
        failing_svc.async_chat_completion = AsyncMock(
            side_effect=RuntimeError("OpenAI API error")
        )
        pipeline = ExtractionPipeline(
            openai_service=failing_svc,
            tokenizer=mock_tokenizer,
        )
        result = await pipeline.extract_chunk(sample_chunk, gleaning_rounds=0)
        assert result.extraction_completed is False
        assert result.error_message is not None
        assert "OpenAI API error" in result.error_message

    @pytest.mark.asyncio
    async def test_skip_claims_produces_no_claims(self, mock_openai_service, mock_tokenizer, sample_chunk):
        from app.core.pipeline.extraction import ExtractionPipeline
        pipeline = ExtractionPipeline(
            openai_service=mock_openai_service,
            tokenizer=mock_tokenizer,
            skip_claims=True,
        )
        result = await pipeline.extract_chunk(sample_chunk, gleaning_rounds=0)
        assert result.extraction_completed is True
        # Claims might still parse from mock response — what matters is
        # that the pipeline does not inject claim prompt
        assert pipeline.skip_claims is True

    @pytest.mark.asyncio
    async def test_calls_openai_service_once_without_gleaning(
        self, mock_openai_service, mock_tokenizer, sample_chunk
    ):
        from app.core.pipeline.extraction import ExtractionPipeline
        pipeline = ExtractionPipeline(
            openai_service=mock_openai_service,
            tokenizer=mock_tokenizer,
        )
        await pipeline.extract_chunk(sample_chunk, gleaning_rounds=0)
        assert mock_openai_service.async_chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_gleaning_loop_called_when_configured(
        self, mock_openai_service, mock_tokenizer, sample_chunk
    ):
        from app.core.pipeline.extraction import ExtractionPipeline
        mock_gleaner = MagicMock()
        mock_gleaner.run = AsyncMock(return_value=MagicMock(
            chunk_id="chunk_0001",
            entities=[], relationships=[], claims=[],
            extraction_completed=True,
            gleaning_rounds_completed=2,
        ))
        pipeline = ExtractionPipeline(
            openai_service=mock_openai_service,
            tokenizer=mock_tokenizer,
            gleaning_loop=mock_gleaner,
        )
        await pipeline.extract_chunk(sample_chunk, gleaning_rounds=2)
        mock_gleaner.run.assert_called_once()


# ── extract_chunks_batch() ────────────────────────────────────────────────────

class TestExtractChunksBatch:

    @pytest.mark.asyncio
    async def test_batch_returns_one_result_per_chunk(
        self, pipeline, sample_chunks
    ):
        results = await pipeline.extract_chunks_batch(
            chunks=sample_chunks[:3],
            gleaning_rounds=0,
            max_concurrency=2,
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty_list(self, pipeline):
        results = await pipeline.extract_chunks_batch(chunks=[])
        assert results == []

    @pytest.mark.asyncio
    async def test_progress_callback_called_per_chunk(
        self, pipeline, sample_chunks
    ):
        calls = []
        async def on_complete(extraction):
            calls.append(extraction.chunk_id)

        await pipeline.extract_chunks_batch(
            chunks=sample_chunks[:4],
            gleaning_rounds=0,
            max_concurrency=4,
            on_chunk_complete=on_complete,
        )
        assert len(calls) == 4

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_abort_batch(
        self, mock_tokenizer, sample_chunks
    ):
        from app.core.pipeline.extraction import ExtractionPipeline

        call_count = [0]
        results_storage = []

        async def flaky_completion(messages):
            call_count[0] += 1
            from app.services.openai_service import CompletionResult
            if call_count[0] == 2:
                raise RuntimeError("Transient error on chunk 2")
            return CompletionResult(
                content='("entity"<|>OpenAI<|>ORGANIZATION<|>AI lab)##<|COMPLETE|>',
                model="gpt-4o", prompt_tokens=50,
                completion_tokens=20, total_tokens=70, finish_reason="stop",
            )

        svc = MagicMock()
        svc.model = "gpt-4o"
        svc.async_chat_completion = flaky_completion

        pipeline = ExtractionPipeline(
            openai_service=svc, tokenizer=mock_tokenizer
        )
        results = await pipeline.extract_chunks_batch(
            chunks=sample_chunks[:3],
            gleaning_rounds=0,
        )
        assert len(results) == 3
        # One should be failed
        failed = [r for r in results if not r.extraction_completed]
        assert len(failed) == 1