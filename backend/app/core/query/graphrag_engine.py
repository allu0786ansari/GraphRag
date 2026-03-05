"""
core/query/graphrag_engine.py — GraphRAG map-reduce query engine.

Implements the full GraphRAG query pipeline from the paper (Section 3.2):

  Map stage (parallel):
    For each community summary at the selected level:
      - Build a context window: community title + summary + key findings
      - Ask the LLM: "Given this community data, generate a partial answer
        to the question. Rate how helpful this context was (0–100)."
      - The LLM returns a partial answer + helpfulness score as JSON.
    All map calls run concurrently (up to max_concurrency at once).

  Filter stage:
    Discard partial answers with helpfulness_score <= threshold (paper: 0).
    Sort remaining answers by helpfulness_score descending.

  Reduce stage (single call):
    Combine all surviving partial answers into one final answer.
    Partial answers are concatenated and fed to the LLM, which synthesises
    a comprehensive global response. If partial answers exceed the 8k window,
    the top-scoring ones fill the window (priority by helpfulness score).

Community level selection:
  C0 = broadest, fewest communities, cheapest (~2.3% of full-text cost)
  C1 = default (best comprehensiveness/cost trade-off per paper Table 2)
  C2 = more granular, ~57.4% cost
  C3 = most granular (leaf), ~66.8% cost

Paper reference Section 3.2:
  "We generate intermediate answers from each community report context,
   with a helpfulness score, then reduce all helpful intermediate answers
   to a single final response."

Prompt design:
  Map prompt asks for JSON: {"points": [{"description": str, "score": int}]}
  This exact JSON structure is from the paper's Appendix B.
  Reduce prompt uses the collected partial answers as input.

Usage:
    engine = GraphRAGEngine.from_settings()
    answer = await engine.query(
        query="What are the main themes across the corpus?",
        community_level="c1",
    )
    print(answer.answer)
    print(f"Used {answer.map_answers_after_filter} partial answers")
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from app.models.graph_models import CommunitySummary
from app.models.response_models import CommunityContext, GraphRAGAnswer, TokenUsage
from app.services.openai_service import OpenAIService
from app.services.tokenizer_service import TokenizerService
from app.storage.summary_store import SummaryStore
from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Prompt templates (paper Appendix B) ───────────────────────────────────────

# Map stage: one call per community summary
_MAP_SYSTEM_PROMPT = (
    "You are a helpful assistant analyzing a knowledge graph community report "
    "to answer a global question about a document corpus. "
    "Respond ONLY with valid JSON — no preamble, no markdown fences."
)

_MAP_USER_PROMPT = """\
Given the following community report from a knowledge graph, generate a list of \
key points that are relevant to answering the question. Each point should have a \
description and a score (0–100) indicating how helpful that point is for \
answering the question. Score 0 means completely irrelevant.

Community Report:
{community_context}

Question: {query}

Respond with this exact JSON format:
{{
  "points": [
    {{"description": "...", "score": <integer 0-100>}},
    ...
  ]
}}

If this community report is not relevant to the question, return:
{{"points": [{{"description": "Not relevant.", "score": 0}}]}}"""

# Reduce stage: single call combining all partial answers
_REDUCE_SYSTEM_PROMPT = (
    "You are a helpful assistant synthesizing research findings into a "
    "comprehensive, well-structured answer. "
    "Draw on all provided points to construct a complete response."
)

_REDUCE_USER_PROMPT = """\
You are generating a comprehensive answer to a question using a collection of \
relevant analytical points drawn from a knowledge graph analysis of a large \
document corpus.

Analytical Points (ranked by relevance):
{partial_answers}

Question: {query}

Instructions:
- Synthesize ALL provided points into a single comprehensive answer.
- Be thorough — cover multiple aspects and themes.
- Organise your response clearly with appropriate structure.
- Do not simply list the points; integrate them into a coherent narrative.
- If points are contradictory, note the different perspectives.

Answer:"""


class GraphRAGEngine:
    """
    GraphRAG map-reduce query engine.

    Implements the full paper pipeline: parallel map over community summaries,
    helpfulness filtering, then a single reduce call to synthesise the final answer.

    The engine is stateful: community summaries are loaded from disk on first
    query and cached in memory for subsequent queries.

    Usage:
        engine = GraphRAGEngine.from_settings()

        answer = await engine.query(
            query="What are the main themes?",
            community_level="c1",
            helpfulness_threshold=0,
        )
    """

    def __init__(
        self,
        openai_service: OpenAIService,
        summary_store: SummaryStore,
        tokenizer: TokenizerService,
        context_window: int = 8000,
        max_concurrency: int = 20,
        helpfulness_threshold: int = 0,
    ) -> None:
        """
        Args:
            openai_service:        For all LLM calls (map + reduce).
            summary_store:         Loads community summaries from disk.
            tokenizer:             Token counting for context window management.
            context_window:        Token limit for each LLM call. Paper: 8000.
            max_concurrency:       Max concurrent map-stage calls. Default: 20.
            helpfulness_threshold: Discard partial answers at or below this score.
                                   Paper uses 0 (only discard score=0 answers).
        """
        self.openai_service        = openai_service
        self.summary_store         = summary_store
        self.tokenizer             = tokenizer
        self.context_window        = context_window
        self.max_concurrency       = max_concurrency
        self.helpfulness_threshold = helpfulness_threshold

    # ── Public query interface ─────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        community_level: str = "c1",
        helpfulness_threshold: int | None = None,
        include_context: bool = True,
        include_token_usage: bool = True,
    ) -> GraphRAGAnswer:
        """
        Answer a question using GraphRAG map-reduce.

        Args:
            query:                The global sensemaking question.
            community_level:      Which community hierarchy level to query.
                                  "c0"=broadest/cheapest, "c1"=default,
                                  "c2"/"c3"=more granular/expensive.
            helpfulness_threshold: Override the engine's default threshold.
                                   Partial answers at or below this score are discarded.
            include_context:      If True, include community contexts in response.
            include_token_usage:  If True, include token usage breakdown.

        Returns:
            GraphRAGAnswer with final answer, map/reduce statistics, and
            optionally the community contexts used.
        """
        t0 = time.monotonic()
        query = query.strip()
        threshold = helpfulness_threshold if helpfulness_threshold is not None \
            else self.helpfulness_threshold

        log.info(
            "GraphRAG query started",
            query_preview=query[:80],
            community_level=community_level,
            threshold=threshold,
        )

        # 1. Load community summaries for the selected level
        summaries = self.summary_store.load_summaries_by_level(community_level)
        if not summaries:
            log.warning(
                "No community summaries at level",
                level=community_level,
            )
            return self._empty_answer(
                query=query,
                community_level=community_level,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        log.info(
            "Community summaries loaded for map stage",
            count=len(summaries),
            level=community_level,
        )

        # 2. Map stage — generate partial answers in parallel
        map_results, map_token_usage = await self._map_stage(
            query=query,
            summaries=summaries,
        )

        # 3. Filter: discard low-score partial answers
        all_points: list[dict] = []
        community_contexts: list[CommunityContext] = []
        total_context_tokens = 0

        for community, points, ctx_tokens in map_results:
            best_score = max((p["score"] for p in points), default=0)
            helpful_points = [p for p in points if p["score"] > threshold]

            if include_context:
                partial_text = "\n".join(
                    f"- {p['description']}" for p in helpful_points[:3]
                ) if helpful_points else None

                community_contexts.append(CommunityContext(
                    community_id=community.community_id,
                    level=community.level.value,
                    title=community.title,
                    summary=community.summary,
                    helpfulness_score=best_score,
                    partial_answer=partial_text,
                    token_count=ctx_tokens,
                ))

            all_points.extend(helpful_points)
            if helpful_points:
                total_context_tokens += ctx_tokens

        # Sort by score descending (best partial answers first for reduce stage)
        all_points.sort(key=lambda p: p["score"], reverse=True)

        map_answers_generated   = sum(len(pts) for _, pts, _ in map_results)
        map_answers_after_filter = len(all_points)

        log.info(
            "Map stage complete",
            total_communities=len(summaries),
            total_points=map_answers_generated,
            points_after_filter=map_answers_after_filter,
            threshold=threshold,
        )

        if not all_points:
            log.warning("No helpful partial answers after filtering", query=query)
            return self._empty_answer(
                query=query,
                community_level=community_level,
                communities_total=len(summaries),
                communities_used=len([r for r in map_results if r[1]]),
                map_answers_generated=map_answers_generated,
                map_answers_after_filter=0,
                context_tokens=total_context_tokens,
                context=community_contexts if include_context else None,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # 4. Reduce stage — synthesise final answer from partial answers
        final_answer, reduce_token_usage = await self._reduce_stage(
            query=query,
            points=all_points,
        )

        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        # 5. Aggregate token usage
        token_usage = None
        if include_token_usage:
            total_prompt     = map_token_usage["prompt"]     + reduce_token_usage["prompt"]
            total_completion = map_token_usage["completion"] + reduce_token_usage["completion"]
            token_usage = TokenUsage(
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_prompt + total_completion,
            )

        log.info(
            "GraphRAG query complete",
            community_level=community_level,
            communities_total=len(summaries),
            map_answers_generated=map_answers_generated,
            map_answers_after_filter=map_answers_after_filter,
            answer_length=len(final_answer),
            latency_ms=latency_ms,
        )

        return GraphRAGAnswer(
            answer=final_answer,
            query=query,
            community_level=community_level,
            communities_total=len(summaries),
            communities_used_in_map=len(summaries),
            map_answers_generated=map_answers_generated,
            map_answers_after_filter=map_answers_after_filter,
            context_tokens_used=total_context_tokens,
            context=community_contexts if include_context else None,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    # ── Map stage ──────────────────────────────────────────────────────────────

    async def _map_stage(
        self,
        query: str,
        summaries: list[CommunitySummary],
    ) -> tuple[list[tuple[CommunitySummary, list[dict], int]], dict]:
        """
        Run the map stage: one LLM call per community summary, concurrently.

        Each call receives the community's title + summary + top findings
        and returns a list of key points with helpfulness scores.

        Returns:
            Tuple of:
              - List of (community, points, context_tokens) triples.
                'points' is the parsed list from the LLM's JSON response.
              - Aggregate token usage dict: {"prompt": int, "completion": int}
        """
        async def _map_one(
            community: CommunitySummary,
        ) -> tuple[CommunitySummary, list[dict], int]:
            community_ctx = self._build_community_context(community)
            ctx_tokens = self.tokenizer.count_tokens(community_ctx)

            prompt = _MAP_USER_PROMPT.format(
                community_context=community_ctx,
                query=query,
            )

            try:
                result = await self.openai_service.complete(
                    user_prompt=prompt,
                    system_prompt=_MAP_SYSTEM_PROMPT,
                    json_mode=True,
                )
                points = self._parse_map_response(result.content, community.community_id)
                # Attach token usage to the points list for aggregation
                points = [dict(p, _prompt_tokens=result.prompt_tokens,
                               _completion_tokens=result.completion_tokens)
                          for p in points]
            except Exception as e:
                log.warning(
                    "Map stage call failed for community",
                    community_id=community.community_id,
                    error=str(e),
                )
                points = []

            return community, points, ctx_tokens

        # Run all communities concurrently
        raw_results = await gather_with_concurrency(
            coroutines=[_map_one(s) for s in summaries],
            max_concurrency=self.max_concurrency,
            return_exceptions=True,
        )

        # Separate successes from exceptions
        results: list[tuple[CommunitySummary, list[dict], int]] = []
        total_prompt = 0
        total_completion = 0

        for item in raw_results:
            if isinstance(item, BaseException):
                log.warning("Map stage exception", error=str(item))
                continue
            community, points, ctx_tokens = item
            # Extract and strip the private token-usage keys
            for p in points:
                total_prompt     += p.pop("_prompt_tokens", 0)
                total_completion += p.pop("_completion_tokens", 0)
            results.append((community, points, ctx_tokens))

        return results, {"prompt": total_prompt, "completion": total_completion}

    # ── Reduce stage ───────────────────────────────────────────────────────────

    async def _reduce_stage(
        self,
        query: str,
        points: list[dict],
    ) -> tuple[str, dict]:
        """
        Reduce all partial answer points into a single final answer.

        Fills the 8k context window with points in score order (highest first).
        Sends them to the LLM to synthesise a single comprehensive answer.

        Returns:
            Tuple of (final_answer_text, {"prompt": int, "completion": int})
        """
        # Build the partial-answers text, filling up to context_window tokens
        partial_answers_text, token_count = self._fill_reduce_context(points, query)

        prompt = _REDUCE_USER_PROMPT.format(
            partial_answers=partial_answers_text,
            query=query,
        )

        result = await self.openai_service.complete(
            user_prompt=prompt,
            system_prompt=_REDUCE_SYSTEM_PROMPT,
        )

        log.debug(
            "Reduce stage complete",
            points_used=len(points),
            context_tokens=token_count,
            answer_tokens=result.completion_tokens,
        )

        return result.content, {
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
        }

    # ── Context building ───────────────────────────────────────────────────────

    def _build_community_context(self, community: CommunitySummary) -> str:
        """
        Build the text representation of a community for the map prompt.

        Includes: title, summary, impact rating, top findings.
        Findings are included up to the context_window limit.
        """
        lines: list[str] = [
            f"Community: {community.title}",
            f"Impact Rating: {community.impact_rating:.1f}/10",
            "",
            "Summary:",
            community.summary,
            "",
            "Key Findings:",
        ]

        for finding in community.findings:
            finding_text = f"- {finding.summary}"
            if hasattr(finding, "explanation") and finding.explanation:
                finding_text += f" ({finding.explanation})"
            lines.append(finding_text)

        full_text = "\n".join(lines)

        # Truncate to fit within context window, leaving room for the prompt template
        max_ctx_tokens = self.context_window - 500  # 500 tokens reserved for prompt overhead
        tokens = self.tokenizer.count_tokens(full_text)
        if tokens > max_ctx_tokens:
            full_text = self.tokenizer.truncate_to_limit(full_text, max_ctx_tokens)

        return full_text

    def _fill_reduce_context(
        self,
        points: list[dict],
        query: str,
    ) -> tuple[str, int]:
        """
        Build the reduce-stage input, filling the context window with points.

        Points are already sorted by score descending. We add them until
        the next point would exceed the context_window limit.

        Returns:
            Tuple of (partial_answers_text, total_token_count)
        """
        overhead = self.tokenizer.count_tokens(
            _REDUCE_USER_PROMPT.format(partial_answers="", query=query)
        )
        available = self.context_window - overhead

        included_lines: list[str] = []
        total_tokens = 0

        for i, point in enumerate(points, 1):
            line = f"{i}. [Score: {point['score']}] {point['description']}"
            line_tokens = self.tokenizer.count_tokens(line)
            if total_tokens + line_tokens > available:
                break
            included_lines.append(line)
            total_tokens += line_tokens

        return "\n".join(included_lines), total_tokens

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_map_response(
        self,
        response_text: str,
        community_id: str,
    ) -> list[dict]:
        """
        Parse the map-stage JSON response from the LLM.

        Expected format:
            {"points": [{"description": "...", "score": 75}, ...]}

        Falls back gracefully on parse errors.
        """
        # Strip markdown fences if present
        text = response_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object from the text
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                except json.JSONDecodeError:
                    log.debug(
                        "Map response JSON parse failed",
                        community_id=community_id,
                        preview=text[:200],
                    )
                    return []
            else:
                return []

        if not isinstance(data, dict):
            return []

        raw_points = data.get("points", [])
        if not isinstance(raw_points, list):
            return []

        parsed: list[dict] = []
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            try:
                score = int(item.get("score", 0))
                score = max(0, min(100, score))
            except (TypeError, ValueError):
                score = 0
            parsed.append({"description": description, "score": score})

        return parsed

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _empty_answer(
        self,
        query: str,
        community_level: str,
        communities_total: int = 0,
        communities_used: int = 0,
        map_answers_generated: int = 0,
        map_answers_after_filter: int = 0,
        context_tokens: int = 0,
        context: list[CommunityContext] | None = None,
        latency_ms: float = 0.0,
    ) -> GraphRAGAnswer:
        """Return a graceful empty answer when no community summaries are available."""
        return GraphRAGAnswer(
            answer=(
                "The knowledge graph does not contain community summaries at this level. "
                "Ensure the indexing pipeline has completed successfully."
                if communities_total == 0 else
                "No community summaries provided sufficient information to answer this question."
            ),
            query=query,
            community_level=community_level,
            communities_total=communities_total,
            communities_used_in_map=communities_used,
            map_answers_generated=map_answers_generated,
            map_answers_after_filter=map_answers_after_filter,
            context_tokens_used=context_tokens,
            context=context,
            token_usage=None,
            latency_ms=latency_ms,
        )

    def get_available_levels(self) -> list[str]:
        """Return community levels that have summaries available."""
        try:
            summaries = self.summary_store.load_summaries()
            levels = sorted({s.level.value for s in summaries})
            return levels
        except FileNotFoundError:
            return []

    def get_community_counts(self) -> dict[str, int]:
        """Return community count per level."""
        try:
            summaries = self.summary_store.load_summaries()
            counts: dict[str, int] = {}
            for s in summaries:
                lv = s.level.value
                counts[lv] = counts.get(lv, 0) + 1
            return counts
        except FileNotFoundError:
            return {}

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> "GraphRAGEngine":
        """Build a GraphRAGEngine from application settings."""
        from app.config import get_settings
        settings = get_settings()

        openai_svc = OpenAIService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=0.0,
        )
        store = SummaryStore(artifacts_dir=settings.artifacts_dir)
        tokenizer = TokenizerService(model=settings.openai_model)

        return cls(
            openai_service=openai_svc,
            summary_store=store,
            tokenizer=tokenizer,
            context_window=settings.context_window_size,
            helpfulness_threshold=settings.helpfulness_score_threshold,
        )


def get_graphrag_engine() -> GraphRAGEngine:
    """FastAPI dependency: return a GraphRAGEngine built from settings."""
    return GraphRAGEngine.from_settings()


__all__ = [
    "GraphRAGEngine",
    "get_graphrag_engine",
]