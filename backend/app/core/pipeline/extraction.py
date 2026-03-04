"""
core/pipeline/extraction.py — GPT-4o entity, relationship, and claim extraction.

Implements paper Section 3.1.2 and Appendix E.1 (extraction prompts).

For each chunk, the extraction pipeline:
  1. Sends an extraction prompt to GPT-4o requesting entities and relationships
     in a structured tuple format (paper Appendix E.1).
  2. Parses the LLM response to extract typed records.
  3. Passes the conversation history to the GleaningLoop for 0–2 additional
     self-reflection rounds to catch missed entities.
  4. Saves the final ChunkExtraction (primary + gleaning results merged).

Output format (paper Appendix E.1 tuple format):
  ("entity"{TUPLE_DELIM}<name>{TUPLE_DELIM}<type>{TUPLE_DELIM}<description>)
  ("relationship"{TUPLE_DELIM}<src>{TUPLE_DELIM}<tgt>{TUPLE_DELIM}<desc>{TUPLE_DELIM}<strength>)

Where TUPLE_DELIM = "<|>"  and  RECORD_DELIM = "##"

Claim extraction (optional, Section 3.1.2):
  ("claim"{TUPLE_DELIM}<subject>{TUPLE_DELIM}<type>{TUPLE_DELIM}<status>{TUPLE_DELIM}
          <description>{TUPLE_DELIM}<start_date>{TUPLE_DELIM}<end_date>)
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from app.models.graph_models import (
    ChunkExtraction,
    ChunkSchema,
    ExtractedClaim,
    ExtractedEntity,
    ExtractedRelationship,
)
from app.services.openai_service import OpenAIService, build_messages
from app.services.tokenizer_service import TokenizerService
from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger
from app.core.pipeline.gleaning import GleaningLoop

log = get_logger(__name__)

# ── Tuple format constants (paper Appendix E.1) ────────────────────────────────
TUPLE_DELIM  = "<|>"
RECORD_DELIM = "##"
COMPLETION_DELIM = "<|COMPLETE|>"

# ── Extraction system prompt (paper Appendix E.1) ─────────────────────────────
_EXTRACTION_SYSTEM_PROMPT = """\
You are a helpful assistant that performs information extraction from text.
Your goal is to identify all entities and relationships in the provided text.

-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, \
identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delim}<entity_name>{tuple_delim}<entity_type>{tuple_delim}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delim}<source_entity>{tuple_delim}<target_entity>{tuple_delim}<relationship_description>{tuple_delim}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. \
Use {record_delim} as the list delimiter.

4. When finished, output {completion_delim}

######################
-Examples-
######################
Example 1:
Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of the Anthropic office in San Francisco was palpable.
################
Output:
("entity"{tuple_delim}Anthropic{tuple_delim}organization{tuple_delim}Anthropic is an AI safety company.)##
("entity"{tuple_delim}Alex{tuple_delim}person{tuple_delim}Alex is an employee at Anthropic.)##
("entity"{tuple_delim}San Francisco{tuple_delim}location{tuple_delim}San Francisco is the location of the Anthropic office.)##
("relationship"{tuple_delim}Alex{tuple_delim}Anthropic{tuple_delim}Alex is employed by Anthropic.{tuple_delim}8)##
("relationship"{tuple_delim}Anthropic{tuple_delim}San Francisco{tuple_delim}Anthropic is headquartered in San Francisco.{tuple_delim}7)##
<|COMPLETE|>
######################
-Real Data-
######################
Entity_types: [{entity_types}]
Text: {input_text}
################
Output:"""

# ── Claim extraction prompt (optional) ────────────────────────────────────────
_CLAIM_EXTRACTION_PROMPT = """\
Additionally, identify any verifiable factual claims about the entities above.
For each claim, extract:
- subject_entity: the entity the claim is about
- claim_type: type of claim (e.g. founding, acquisition, investment, role, event)
- claim_status: TRUE, FALSE, SUSPECTED, INFERRED, or NOT EVALUATED
- claim_description: the full claim text
- start_date: ISO date if the claim has a start date (or NONE)
- end_date: ISO date if the claim has an end date (or NONE)
Format each claim as:
("claim"{tuple_delim}<subject_entity>{tuple_delim}<claim_type>{tuple_delim}<claim_status>{tuple_delim}<claim_description>{tuple_delim}<start_date>{tuple_delim}<end_date>)
"""

# ── Default entity types (paper uses these for news corpora) ──────────────────
DEFAULT_ENTITY_TYPES = [
    "organization", "person", "location", "event",
    "product", "technology", "concept",
]


class ExtractionPipeline:
    """
    GPT-4o entity, relationship, and claim extraction for document chunks.

    Processes one chunk at a time via extract_chunk(), or processes many
    chunks concurrently via extract_chunks_batch().

    Usage:
        pipeline = ExtractionPipeline(openai_service, tokenizer)

        # Single chunk
        extraction = await pipeline.extract_chunk(chunk)

        # Batch (with concurrency control)
        extractions = await pipeline.extract_chunks_batch(chunks, max_concurrency=20)
    """

    def __init__(
        self,
        openai_service: OpenAIService,
        tokenizer: TokenizerService,
        gleaning_loop: GleaningLoop | None = None,
        entity_types: list[str] | None = None,
        skip_claims: bool = False,
    ) -> None:
        """
        Args:
            openai_service: OpenAI service (all LLM calls go through here).
            tokenizer:      TokenizerService (for context window checks).
            gleaning_loop:  GleaningLoop instance. If None, gleaning is disabled.
            entity_types:   Entity type labels for the extraction prompt.
                            Defaults to DEFAULT_ENTITY_TYPES.
            skip_claims:    If True, skip claim extraction. Default: False.
        """
        self.openai_service = openai_service
        self.tokenizer      = tokenizer
        self.gleaning_loop  = gleaning_loop
        self.entity_types   = entity_types or DEFAULT_ENTITY_TYPES
        self.skip_claims    = skip_claims

    async def extract_chunk(
        self,
        chunk: ChunkSchema,
        gleaning_rounds: int = 2,
    ) -> ChunkExtraction:
        """
        Extract entities, relationships, and claims from a single chunk.

        Args:
            chunk:           The ChunkSchema to process.
            gleaning_rounds: Number of gleaning iterations. Default: 2.
                             Set to 0 to disable gleaning for this chunk.

        Returns:
            ChunkExtraction with all extracted data. On failure, returns a
            ChunkExtraction with extraction_completed=False and error_message set.
        """
        t0 = time.monotonic()

        try:
            extraction, history = await self._primary_extraction(chunk)

            # Run gleaning if a loop is configured
            if self.gleaning_loop and gleaning_rounds > 0:
                extraction = await self.gleaning_loop.run(
                    chunk_text=chunk.text,
                    initial_extraction=extraction,
                    conversation_history=history,
                    max_rounds=gleaning_rounds,
                    parse_fn=self._parse_extraction_output,
                )

            log.debug(
                "Chunk extracted",
                chunk_id=chunk.chunk_id,
                entities=len(extraction.entities),
                relationships=len(extraction.relationships),
                claims=len(extraction.claims),
                gleaning_rounds=extraction.gleaning_rounds_completed,
                elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            return extraction

        except Exception as e:
            log.error(
                "Extraction failed",
                chunk_id=chunk.chunk_id,
                error=str(e),
            )
            return ChunkExtraction(
                chunk_id=chunk.chunk_id,
                extraction_completed=False,
                error_message=str(e)[:500],
            )

    async def _primary_extraction(
        self,
        chunk: ChunkSchema,
    ) -> tuple[ChunkExtraction, list[dict]]:
        """
        Run the primary extraction prompt for a chunk.

        Returns:
            Tuple of (ChunkExtraction, conversation_history).
            The conversation history is passed to the gleaning loop.
        """
        entity_types_str = ", ".join(self.entity_types)

        prompt = _EXTRACTION_SYSTEM_PROMPT.format(
            entity_types=entity_types_str,
            tuple_delim=TUPLE_DELIM,
            record_delim=RECORD_DELIM,
            completion_delim=COMPLETION_DELIM,
            input_text=chunk.text,
        )

        if not self.skip_claims:
            prompt = prompt.rstrip() + "\n\n" + _CLAIM_EXTRACTION_PROMPT.format(
                tuple_delim=TUPLE_DELIM,
            )

        messages = [{"role": "user", "content": prompt}]

        result = await self.openai_service.async_chat_completion(messages=messages)

        extraction = self._parse_extraction_output(result.content, chunk.chunk_id)

        # Build the conversation history for the gleaning loop
        history = [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": result.content},
        ]

        return extraction, history

    def _parse_extraction_output(
        self,
        text: str,
        chunk_id: str,
    ) -> ChunkExtraction:
        """
        Parse the tuple-format LLM output into a ChunkExtraction.

        Handles the paper's format (Appendix E.1):
          ("entity"<|>name<|>type<|>description)##
          ("relationship"<|>src<|>tgt<|>desc<|>strength)##
          ("claim"<|>subject<|>type<|>status<|>desc<|>start<|>end)##

        Malformed records are logged and skipped — never raise.
        """
        # Strip everything after the completion delimiter
        if COMPLETION_DELIM in text:
            text = text[:text.index(COMPLETION_DELIM)]

        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelationship] = []
        claims: list[ExtractedClaim] = []

        # Split on RECORD_DELIM and process each record
        records = [r.strip() for r in text.split(RECORD_DELIM) if r.strip()]

        for record in records:
            # Strip outer parentheses if present
            record = record.strip()
            if record.startswith("(") and record.endswith(")"):
                record = record[1:-1]

            if not record:
                continue

            parts = [p.strip() for p in record.split(TUPLE_DELIM)]
            if len(parts) < 2:
                continue

            record_type = parts[0].strip('"').lower()

            try:
                if record_type == "entity" and len(parts) >= 4:
                    entities.append(ExtractedEntity(
                        name=parts[1],
                        entity_type=parts[2].upper(),
                        description=parts[3],
                        source_chunk_id=chunk_id,
                    ))

                elif record_type == "relationship" and len(parts) >= 5:
                    try:
                        strength = int(float(parts[4].strip()))
                        strength = max(1, min(10, strength))
                    except (ValueError, IndexError):
                        strength = 5

                    relationships.append(ExtractedRelationship(
                        source_entity=parts[1],
                        target_entity=parts[2],
                        description=parts[3],
                        strength=strength,
                        source_chunk_id=chunk_id,
                    ))

                elif record_type == "claim" and len(parts) >= 5:
                    claims.append(ExtractedClaim(
                        subject_entity=parts[1],
                        claim_type=parts[2] if len(parts) > 2 else "unknown",
                        status=parts[3] if len(parts) > 3 else "NOT EVALUATED",
                        claim_description=parts[4] if len(parts) > 4 else "",
                        start_date=_clean_date(parts[5]) if len(parts) > 5 else None,
                        end_date=_clean_date(parts[6])   if len(parts) > 6 else None,
                        source_chunk_id=chunk_id,
                    ))

            except Exception as e:
                log.debug(
                    "Skipping malformed extraction record",
                    record_type=record_type,
                    error=str(e),
                )

        return ChunkExtraction(
            chunk_id=chunk_id,
            entities=entities,
            relationships=relationships,
            claims=claims,
            gleaning_rounds_completed=0,
            extraction_completed=True,
        )

    async def extract_chunks_batch(
        self,
        chunks: list[ChunkSchema],
        gleaning_rounds: int = 2,
        max_concurrency: int = 20,
        on_chunk_complete: "callable | None" = None,
    ) -> list[ChunkExtraction]:
        """
        Extract all chunks concurrently with a concurrency cap.

        Args:
            chunks:            Chunks to process.
            gleaning_rounds:   Gleaning iterations per chunk.
            max_concurrency:   Max concurrent LLM calls. Default: 20.
            on_chunk_complete: Optional callback(ChunkExtraction) called after
                               each chunk completes. Use to save incrementally.

        Returns:
            List of ChunkExtraction in the same order as input chunks.
        """
        if not chunks:
            return []

        t0 = time.monotonic()
        log.info(
            "Batch extraction started",
            total_chunks=len(chunks),
            gleaning_rounds=gleaning_rounds,
            max_concurrency=max_concurrency,
        )

        async def _extract_one(chunk: ChunkSchema) -> ChunkExtraction:
            result = await self.extract_chunk(chunk, gleaning_rounds=gleaning_rounds)
            if on_chunk_complete:
                try:
                    if asyncio.iscoroutinefunction(on_chunk_complete):
                        await on_chunk_complete(result)
                    else:
                        on_chunk_complete(result)
                except Exception as e:
                    log.warning("on_chunk_complete callback failed", error=str(e))
            return result

        results = await gather_with_concurrency(
            coroutines=[_extract_one(chunk) for chunk in chunks],
            max_concurrency=max_concurrency,
            return_exceptions=False,
        )

        elapsed = time.monotonic() - t0
        successful = [r for r in results if r.extraction_completed]
        failed     = [r for r in results if not r.extraction_completed]

        total_entities = sum(len(r.entities) for r in successful)
        total_rels     = sum(len(r.relationships) for r in successful)

        log.info(
            "Batch extraction complete",
            total=len(chunks),
            successful=len(successful),
            failed=len(failed),
            total_entities=total_entities,
            total_relationships=total_rels,
            elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(len(chunks) / elapsed, 2) if elapsed > 0 else 0,
        )

        return results

    def get_extraction_stats(self, extractions: list[ChunkExtraction]) -> dict:
        """Return summary statistics about a list of extractions."""
        successful = [e for e in extractions if e.extraction_completed]
        failed     = [e for e in extractions if not e.extraction_completed]

        if not successful:
            return {"total": len(extractions), "successful": 0, "failed": len(failed)}

        all_entities = [e for ext in successful for e in ext.entities]
        all_rels     = [r for ext in successful for r in ext.relationships]
        all_claims   = [c for ext in successful for c in ext.claims]

        entity_types: dict[str, int] = {}
        for e in all_entities:
            entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1

        return {
            "total":                len(extractions),
            "successful":           len(successful),
            "failed":               len(failed),
            "total_entities":       len(all_entities),
            "total_relationships":  len(all_rels),
            "total_claims":         len(all_claims),
            "avg_entities_per_chunk":      round(len(all_entities) / len(successful), 1),
            "avg_relationships_per_chunk": round(len(all_rels) / len(successful), 1),
            "entity_type_distribution": dict(sorted(entity_types.items(), key=lambda x: -x[1])),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_date(val: str) -> str | None:
    """Return a date string if it looks valid, else None."""
    if not val or val.strip().upper() in ("NONE", "NULL", "", "N/A"):
        return None
    return val.strip()


# ── Factory function ───────────────────────────────────────────────────────────

def get_extraction_pipeline(
    skip_claims: bool = False,
    gleaning_rounds: int = 2,
) -> ExtractionPipeline:
    """Build an ExtractionPipeline from application settings."""
    from app.config import get_settings
    from app.services.openai_service import get_openai_service
    from app.services.tokenizer_service import get_tokenizer

    settings = get_settings()
    openai_svc = get_openai_service()
    tokenizer  = get_tokenizer()

    gleaning_loop = GleaningLoop(openai_svc) if gleaning_rounds > 0 else None

    return ExtractionPipeline(
        openai_service=openai_svc,
        tokenizer=tokenizer,
        gleaning_loop=gleaning_loop,
        skip_claims=skip_claims,
    )


__all__ = [
    "ExtractionPipeline",
    "get_extraction_pipeline",
    "TUPLE_DELIM",
    "RECORD_DELIM",
    "COMPLETION_DELIM",
    "DEFAULT_ENTITY_TYPES",
]