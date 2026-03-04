"""
core/pipeline/gleaning.py — Self-reflection gleaning loop.

Implements paper Section 3.1.2:
  "After the initial extraction, we prompt the LLM to identify any entities
   that may have been missed. This loop runs up to N times (default: 2)."

The gleaning loop works as follows:
  1. Build a gleaning prompt asking: "Were any entities missed?"
  2. Force a YES/NO response using logit_bias on those two token IDs.
  3. If YES: send a continuation prompt requesting the missed entities.
  4. Parse the additional extraction output and add to existing results.
  5. Repeat until NO or max_rounds is reached.

The logit_bias trick (setting bias=100 on YES/NO token IDs) forces the LLM
to output exactly one token, making the check cheap (~1 token output vs
a free-form response). This is the paper's exact technique.

Token IDs for cl100k_base and o200k_base:
  YES: varies by encoding, we detect at runtime
  NO:  varies by encoding, we detect at runtime
"""

from __future__ import annotations

import re
import time
from typing import Any

import tiktoken

from app.models.graph_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractedClaim,
)
from app.services.openai_service import OpenAIService, build_messages
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Gleaning prompts ───────────────────────────────────────────────────────────

_GLEANING_CHECK_PROMPT = """\
It appears some entities and relationships may have been missed in the last extraction. \
Answer YES | NO if there are still entities or relationships that need to be added.
"""

_GLEANING_CONTINUE_PROMPT = """\
MANY entities and relationships were missed in the last extraction. \
Remember, every named entity mentioned in the document should appear as a node. \
Add the missing entities and relationships below using the same format as before.\
"""


class GleaningLoop:
    """
    Self-reflection gleaning loop for improving extraction completeness.

    After the initial extraction, this loop:
      1. Asks the LLM whether any entities were missed (forced YES/NO).
      2. If YES, requests the missed entities/relationships.
      3. Parses and merges new extractions into the existing result.
      4. Repeats for up to max_rounds.

    Usage:
        gleaner = GleaningLoop(openai_service)
        updated_extraction = await gleaner.run(
            chunk_text=chunk.text,
            initial_extraction=extraction,
            conversation_history=history,
            max_rounds=2,
        )
    """

    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai_service = openai_service
        # Get YES/NO token IDs for the model's encoding
        self._yes_ids, self._no_ids = _get_yes_no_token_ids(openai_service.model)

        log.debug(
            "GleaningLoop initialized",
            model=openai_service.model,
            yes_token_ids=self._yes_ids,
            no_token_ids=self._no_ids,
        )

    async def run(
        self,
        chunk_text: str,
        initial_extraction: ChunkExtraction,
        conversation_history: list[dict[str, str]],
        max_rounds: int = 2,
        parse_fn: "callable | None" = None,
    ) -> ChunkExtraction:
        """
        Run the gleaning loop on an initial extraction.

        Args:
            chunk_text:           The original chunk text (for context).
            initial_extraction:   The ChunkExtraction from the primary extraction.
            conversation_history: The full conversation so far (system + user + assistant).
                                  The gleaning check/continue messages are appended here.
            max_rounds:           Maximum gleaning iterations. Default: 2 (paper-exact).
                                  Set to 0 to disable gleaning entirely.
            parse_fn:             Function to parse extraction output text.
                                  Signature: parse_fn(text, chunk_id) -> ChunkExtraction
                                  If None, uses the default tuple-format parser.

        Returns:
            Updated ChunkExtraction with all entities/relationships merged across rounds.
        """
        if max_rounds == 0:
            return initial_extraction

        extraction = initial_extraction
        history = list(conversation_history)  # defensive copy
        rounds_completed = 0

        for round_num in range(1, max_rounds + 1):
            t0 = time.monotonic()

            # Step 1: Ask if entities were missed (forced YES/NO)
            history.append({"role": "user", "content": _GLEANING_CHECK_PROMPT})

            check_result = await self.openai_service.async_completion_with_logit_bias(
                messages=history,
                yes_token_ids=self._yes_ids,
                no_token_ids=self._no_ids,
                bias=100,
            )

            answer = check_result.content.strip().upper()
            history.append({"role": "assistant", "content": answer})

            log.debug(
                "Gleaning check",
                chunk_id=extraction.chunk_id,
                round=round_num,
                answer=answer,
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )

            # If NO (or unrecognized) — stop gleaning
            if not answer.startswith("Y"):
                log.debug(
                    "Gleaning loop stopped — no missed entities",
                    chunk_id=extraction.chunk_id,
                    rounds_completed=round_num - 1,
                )
                break

            # Step 2: Request the missed entities
            history.append({"role": "user", "content": _GLEANING_CONTINUE_PROMPT})

            continue_result = await self.openai_service.async_chat_completion(
                messages=history,
            )

            continuation_text = continue_result.content
            history.append({"role": "assistant", "content": continuation_text})

            # Step 3: Parse and merge new extractions
            if parse_fn:
                try:
                    new_extraction = parse_fn(continuation_text, extraction.chunk_id)
                    _merge_into(extraction, new_extraction, round_num)

                    log.debug(
                        "Gleaning round complete",
                        chunk_id=extraction.chunk_id,
                        round=round_num,
                        new_entities=len(new_extraction.entities),
                        new_relationships=len(new_extraction.relationships),
                    )
                except Exception as e:
                    log.warning(
                        "Failed to parse gleaning output",
                        chunk_id=extraction.chunk_id,
                        round=round_num,
                        error=str(e),
                    )
            else:
                # Without a parse function, just note the round completed
                log.debug(
                    "Gleaning round completed (no parse_fn provided)",
                    chunk_id=extraction.chunk_id,
                    round=round_num,
                )

            rounds_completed = round_num

        extraction.gleaning_rounds_completed = rounds_completed
        return extraction

    async def check_needs_gleaning(
        self,
        history: list[dict[str, str]],
    ) -> bool:
        """
        Ask the LLM whether any entities were missed (forced YES/NO).

        Standalone method for use in custom extraction loops.

        Returns:
            True if the LLM answers YES (more entities to extract).
        """
        check_history = history + [{"role": "user", "content": _GLEANING_CHECK_PROMPT}]
        result = await self.openai_service.async_completion_with_logit_bias(
            messages=check_history,
            yes_token_ids=self._yes_ids,
            no_token_ids=self._no_ids,
            bias=100,
        )
        return result.content.strip().upper().startswith("Y")


# ── Token ID helpers ───────────────────────────────────────────────────────────

def _get_yes_no_token_ids(model: str) -> tuple[list[int], list[int]]:
    """
    Return tiktoken token IDs for YES and NO variants for the given model.

    We check both uppercase and lowercase forms. The exact IDs depend on
    the encoding (cl100k_base for older GPT-4 vs o200k_base for gpt-4o).
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    yes_variants = ["YES", "Yes", "yes", "Y", "y"]
    no_variants  = ["NO",  "No",  "no",  "N", "n"]

    yes_ids: list[int] = []
    no_ids:  list[int] = []

    for v in yes_variants:
        ids = enc.encode(v)
        if len(ids) == 1:  # only single-token forms work for logit_bias
            yes_ids.append(ids[0])

    for v in no_variants:
        ids = enc.encode(v)
        if len(ids) == 1:
            no_ids.append(ids[0])

    # Deduplicate while preserving order
    yes_ids = list(dict.fromkeys(yes_ids))
    no_ids  = list(dict.fromkeys(no_ids))

    return yes_ids, no_ids


def _merge_into(
    base: ChunkExtraction,
    new: ChunkExtraction,
    round_num: int,
) -> None:
    """
    Merge entities/relationships from a gleaning round into the base extraction.

    Deduplication: entities/relationships with the same name/pair that were
    already extracted are skipped. We use name normalization for comparison.

    Modifies base in-place.
    """
    # Build sets of existing (normalized) names for dedup
    existing_entity_names: set[str] = {
        _norm(e.name) for e in base.entities
    }
    existing_rel_pairs: set[tuple[str, str]] = {
        (_norm(r.source_entity), _norm(r.target_entity))
        for r in base.relationships
    }

    added_entities = 0
    for entity in new.entities:
        if _norm(entity.name) not in existing_entity_names:
            entity.extraction_round = round_num
            base.entities.append(entity)
            existing_entity_names.add(_norm(entity.name))
            added_entities += 1

    added_rels = 0
    for rel in new.relationships:
        pair = (_norm(rel.source_entity), _norm(rel.target_entity))
        if pair not in existing_rel_pairs:
            rel.extraction_round = round_num
            base.relationships.append(rel)
            existing_rel_pairs.add(pair)
            added_rels += 1

    # Always append new claims (they are unique by nature)
    for claim in new.claims:
        base.claims.append(claim)


def _norm(name: str) -> str:
    """Normalize an entity name for deduplication comparison."""
    return name.strip().lower()


__all__ = [
    "GleaningLoop",
    "get_yes_no_token_ids",
]

# Re-export helper for tests
get_yes_no_token_ids = _get_yes_no_token_ids