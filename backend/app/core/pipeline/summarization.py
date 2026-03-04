"""
core/pipeline/summarization.py — Community summary generation.

Implements paper Section 3.1.4:
  "For each community, we generate a report-style summary using the LLM.
   The prompt includes the community's entities, relationships, and claims
   ordered by combined_degree. If the content exceeds the 8k context window,
   we substitute sub-community summaries for raw element data."

Two strategies:
  1. Leaf communities (C3 / single level): build context from entity
     descriptions + edge descriptions sorted by combined_degree descending.
     Truncate to fit the 8k window using TokenizerService.

  2. Higher-level communities (C0, C1, C2): first attempt the leaf strategy.
     If the raw element content exceeds 8k, substitute the child community
     summaries (already generated for C3/leaf level) — these are shorter
     and always fit.

The LLM is prompted to respond in a specific JSON format (paper Appendix E.3):
  {
    "title": "...",
    "summary": "...",
    "rating": 7.5,
    "rating_explanation": "...",
    "findings": [
      {"summary": "...", "explanation": "..."},
      ...
    ]
  }

Generation order: leaf → higher levels, so sub-community summaries are
always available when needed by higher-level communities.

All summaries are generated concurrently within each level (map stage),
with max_concurrency capping the OpenAI call count.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.models.graph_models import (
    CommunityFinding,
    CommunityLevel,
    CommunitySchema,
    CommunitySummary,
)
from app.services.openai_service import OpenAIService
from app.services.tokenizer_service import TokenizerService
from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Summarization prompt (paper Appendix E.3) ──────────────────────────────────
_SUMMARIZATION_SYSTEM_PROMPT = """\
You are an AI assistant that helps a human analyst to perform general information discovery. \
Information discovery is the process of identifying and assessing relevant information associated \
with certain entities (e.g., organizations and individuals) within a network.
"""

_SUMMARIZATION_USER_PROMPT = """\
You are given a community of entities and relationships extracted from a document corpus.
Your task is to generate a comprehensive summary report of this community.

Use the data provided below about community entities and relationships.
Do not make anything up. Provide a factual, grounded summary.

# Community Data:
{community_context}

# Output Format:
Respond ONLY with a valid JSON object matching this exact structure.
Do not include markdown code fences or any text outside the JSON.
{{
  "title": "<A concise title for this community, naming 2-3 key entities>",
  "summary": "<An executive paragraph summarizing the community's main themes and key entities>",
  "rating": <A float 0.0-10.0 rating the community's overall importance/impact>,
  "rating_explanation": "<One sentence explaining the rating>",
  "findings": [
    {{
      "summary": "<One sentence key insight>",
      "explanation": "<Paragraph with supporting detail from the data>"
    }}
  ]
}}

Generate 3-7 findings. Each finding should be a distinct, specific insight supported by the data.
"""

# ── Context window budget ──────────────────────────────────────────────────────
_CONTEXT_WINDOW_TOKENS = 8000
_MAX_RESPONSE_TOKENS   = 2000


class SummarizationPipeline:
    """
    Generates structured community report summaries using GPT-4o.

    Processes communities level by level (leaf first), building context
    windows from graph data and generating JSON reports via the LLM.

    Usage:
        pipeline = SummarizationPipeline(openai_service, tokenizer)

        # Summarize all communities
        summaries = await pipeline.summarize_all(
            communities=communities,
            graph=nx_graph,
            max_concurrency=20,
        )

        # Or summarize a single community
        summary = await pipeline.summarize_community(community, graph)
    """

    def __init__(
        self,
        openai_service: OpenAIService,
        tokenizer: TokenizerService,
        context_window: int = _CONTEXT_WINDOW_TOKENS,
        max_response_tokens: int = _MAX_RESPONSE_TOKENS,
    ) -> None:
        self.openai_service      = openai_service
        self.tokenizer           = tokenizer
        self.context_window      = context_window
        self.max_response_tokens = max_response_tokens

    async def summarize_all(
        self,
        communities: list[CommunitySchema],
        graph: Any,
        max_concurrency: int = 20,
        on_summary_complete: "callable | None" = None,
    ) -> list[CommunitySummary]:
        """
        Generate summaries for all communities, level by level.

        Processing order: leaf level first (C3 or highest available),
        then progressively coarser levels (C2 → C1 → C0). This ensures
        sub-community summaries are available when needed for higher levels.

        Args:
            communities:         All CommunitySchema from community detection.
            graph:               The NetworkX knowledge graph.
            max_concurrency:     Max concurrent LLM calls per level.
            on_summary_complete: Optional callback(CommunitySummary) per summary.
                                 Use to persist summaries incrementally.

        Returns:
            List of CommunitySummary for all communities.
        """
        if not communities:
            log.warning("No communities to summarize")
            return []

        t0 = time.monotonic()
        total = len(communities)

        # Group by level — we need to sort in leaf-first order
        by_level: dict[str, list[CommunitySchema]] = {}
        for comm in communities:
            lv = comm.level.value if hasattr(comm.level, "value") else str(comm.level)
            by_level.setdefault(lv, []).append(comm)

        # Process order: most granular first (c3 → c2 → c1 → c0)
        level_order = sorted(by_level.keys(), reverse=True)

        log.info(
            "Summarization started",
            total_communities=total,
            levels=level_order,
            max_concurrency=max_concurrency,
        )

        all_summaries: list[CommunitySummary] = []
        # Map community_id → summary (for sub-community substitution)
        summaries_by_id: dict[str, CommunitySummary] = {}

        for level_name in level_order:
            level_communities = by_level[level_name]
            log.info(
                "Summarizing level",
                level=level_name,
                count=len(level_communities),
            )

            async def _summarize_one(comm: CommunitySchema) -> CommunitySummary:
                return await self.summarize_community(
                    community=comm,
                    graph=graph,
                    child_summaries=summaries_by_id,
                )

            level_summaries = await gather_with_concurrency(
                coroutines=[_summarize_one(c) for c in level_communities],
                max_concurrency=max_concurrency,
                return_exceptions=False,
            )

            for summary in level_summaries:
                summaries_by_id[summary.community_id] = summary
                all_summaries.append(summary)
                if on_summary_complete:
                    try:
                        import asyncio
                        if asyncio.iscoroutinefunction(on_summary_complete):
                            await on_summary_complete(summary)
                        else:
                            on_summary_complete(summary)
                    except Exception as e:
                        log.warning("on_summary_complete callback failed", error=str(e))

        elapsed = time.monotonic() - t0
        log.info(
            "Summarization complete",
            total=len(all_summaries),
            elapsed_seconds=round(elapsed, 2),
            summaries_per_second=round(len(all_summaries) / elapsed, 2) if elapsed else 0,
        )

        return all_summaries

    async def summarize_community(
        self,
        community: CommunitySchema,
        graph: Any,
        child_summaries: dict[str, CommunitySummary] | None = None,
    ) -> CommunitySummary:
        """
        Generate a summary for a single community.

        Builds a context window from the community's graph data, then
        calls the LLM to generate a structured JSON report.

        Args:
            community:       The CommunitySchema to summarize.
            graph:           The knowledge graph for context building.
            child_summaries: Already-generated summaries for child communities.
                             Used for sub-community substitution when needed.

        Returns:
            CommunitySummary with LLM-generated content.
            On failure, returns a minimal valid summary with error noted.
        """
        t0 = time.monotonic()

        try:
            context, was_truncated, used_sub = self._build_community_context(
                community=community,
                graph=graph,
                child_summaries=child_summaries or {},
            )

            prompt = _SUMMARIZATION_USER_PROMPT.format(
                community_context=context
            )

            result = await self.openai_service.complete(
                user_prompt=prompt,
                system_prompt=_SUMMARIZATION_SYSTEM_PROMPT,
                max_tokens=self.max_response_tokens,
                temperature=0.0,
            )

            summary = self._parse_summary_response(
                response_text=result.content,
                community=community,
                context_tokens=self.tokenizer.count_tokens(context),
                was_truncated=was_truncated,
                used_sub_community=used_sub,
                token_usage={
                    "prompt": result.prompt_tokens,
                    "completion": result.completion_tokens,
                },
            )

            log.debug(
                "Community summarized",
                community_id=community.community_id,
                level=community.level.value,
                context_tokens=summary.context_tokens_used,
                findings=len(summary.findings),
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            return summary

        except Exception as e:
            log.warning(
                "Community summarization failed — using fallback",
                community_id=community.community_id,
                error=str(e),
            )
            return self._make_fallback_summary(community, str(e))

    # ── Context building ───────────────────────────────────────────────────────

    def _build_community_context(
        self,
        community: CommunitySchema,
        graph: Any,
        child_summaries: dict[str, CommunitySummary],
    ) -> tuple[str, bool, bool]:
        """
        Build the context string for a community's LLM prompt.

        Returns:
            (context_text, was_truncated, used_sub_community_substitution)

        Strategy (paper Section 3.1.4):
          1. Collect entity descriptions + edge descriptions sorted by combined_degree.
          2. If all content fits in 8k tokens → use it directly.
          3. If too large AND we have child summaries → substitute child summaries.
          4. If still too large → truncate.
        """
        # Always try the raw element approach first
        raw_context = self._build_raw_context(community, graph)
        raw_tokens  = self.tokenizer.count_tokens(raw_context)

        if raw_tokens <= self.context_window:
            return raw_context, False, False

        # Raw context too large — try sub-community substitution if we have children
        if community.child_community_ids and child_summaries:
            sub_context = self._build_sub_community_context(community, child_summaries)
            sub_tokens  = self.tokenizer.count_tokens(sub_context)

            if sub_tokens <= self.context_window:
                return sub_context, False, True

            # Sub-community context also too large — truncate it
            truncated = self.tokenizer.truncate_to_limit(
                sub_context,
                max_tokens=self.context_window,
                truncation_marker="",
            )
            return truncated, True, True

        # No children or no child summaries — truncate the raw context
        truncated = self.tokenizer.truncate_to_limit(
            raw_context,
            max_tokens=self.context_window,
            truncation_marker="",
        )
        return truncated, True, False

    def _build_raw_context(
        self,
        community: CommunitySchema,
        graph: Any,
    ) -> str:
        """
        Build context from entity descriptions + edges sorted by combined_degree.

        Paper Section 3.1.4:
          "We add edges to the context window in order of combined_degree
           (sum of source and target node degrees) until the 8k window is full."
        """
        parts: list[str] = []

        # Entity summaries
        node_sections: list[str] = []
        for node_id in community.node_ids:
            if not graph.has_node(node_id):
                continue
            node_data = graph.nodes[node_id]
            name  = node_data.get("name", node_id)
            etype = node_data.get("entity_type", "UNKNOWN")
            desc  = node_data.get("description", "")
            claims = node_data.get("claims", [])
            claim_text = " ".join(claims[:3]) if claims else ""

            entry = f"Entity: {name}\nType: {etype}\nDescription: {desc}"
            if claim_text:
                entry += f"\nClaims: {claim_text}"
            node_sections.append(entry)

        if node_sections:
            parts.append("## Entities\n" + "\n\n".join(node_sections))

        # Relationship sections — sorted by combined_degree descending
        node_set = set(community.node_ids)
        edge_entries: list[tuple[int, str]] = []

        for u, v, data in graph.edges(data=True):
            if u not in node_set or v not in node_set:
                continue
            combined_deg = data.get("combined_degree", 0)
            u_name = graph.nodes[u].get("name", u)
            v_name = graph.nodes[v].get("name", v)
            desc   = data.get("description", "")
            weight = data.get("weight", 1)
            entry = f"Relationship: {u_name} ↔ {v_name}\nStrength: {weight}\nDescription: {desc}"
            edge_entries.append((combined_deg, entry))

        edge_entries.sort(key=lambda x: -x[0])

        if edge_entries:
            rel_texts = [e for _, e in edge_entries]
            parts.append("## Relationships\n" + "\n\n".join(rel_texts))

        return "\n\n".join(parts) if parts else f"Community: {community.community_id}\nNodes: {len(community.node_ids)}"

    def _build_sub_community_context(
        self,
        community: CommunitySchema,
        child_summaries: dict[str, CommunitySummary],
    ) -> str:
        """
        Build context from child community summaries (sub-community substitution).

        Used when raw element context exceeds the 8k window.
        Child summaries are much shorter than raw entity/edge data.
        """
        parts: list[str] = [f"## Sub-community Summaries for {community.community_id}"]

        for child_id in community.child_community_ids:
            child = child_summaries.get(child_id)
            if not child:
                continue
            entry = (
                f"### {child.title}\n"
                f"Level: {child.level.value}\n"
                f"Summary: {child.summary}\n"
                f"Impact Rating: {child.impact_rating}"
            )
            parts.append(entry)

        return "\n\n".join(parts)

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_summary_response(
        self,
        response_text: str,
        community: CommunitySchema,
        context_tokens: int,
        was_truncated: bool,
        used_sub_community: bool,
        token_usage: dict[str, int] | None = None,
    ) -> CommunitySummary:
        """
        Parse the LLM's JSON response into a CommunitySummary.

        Robust to: markdown code fences, extra whitespace, partial JSON.
        Falls back to a default summary if parsing fails entirely.
        """
        from datetime import datetime, timezone

        # Strip markdown code fences if present
        text = response_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object using regex
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return self._make_fallback_summary(community, "JSON parse failed")
            else:
                return self._make_fallback_summary(community, "No JSON found in response")

        # Parse findings
        findings: list[CommunityFinding] = []
        raw_findings = data.get("findings", [])
        for i, f in enumerate(raw_findings[:10]):  # cap at 10 per model constraint
            if isinstance(f, dict):
                findings.append(CommunityFinding(
                    finding_id=i,
                    summary=str(f.get("summary", "")),
                    explanation=str(f.get("explanation", "")),
                ))

        # Ensure at least 1 finding
        if not findings:
            findings = [CommunityFinding(
                finding_id=0,
                summary="Community identified",
                explanation=f"Community {community.community_id} contains {community.node_count} entities.",
            )]

        # Parse rating — clamp to [0, 10]
        try:
            rating = float(data.get("rating", 5.0))
            rating = max(0.0, min(10.0, rating))
        except (TypeError, ValueError):
            rating = 5.0

        return CommunitySummary(
            community_id=community.community_id,
            level=community.level,
            title=str(data.get("title", f"Community {community.community_id}")),
            summary=str(data.get("summary", "")),
            impact_rating=rating,
            rating_explanation=str(data.get("rating_explanation", "")),
            findings=findings,
            node_ids=community.node_ids,
            context_tokens_used=context_tokens,
            was_truncated=was_truncated,
            used_sub_community_substitution=used_sub_community,
            generated_at=datetime.now(tz=timezone.utc),
            token_usage=token_usage,
        )

    def _make_fallback_summary(
        self,
        community: CommunitySchema,
        error: str = "",
    ) -> CommunitySummary:
        """
        Return a minimal valid CommunitySummary when LLM generation fails.

        This ensures a failed community doesn't break the pipeline — it
        gets a placeholder summary that can be regenerated later.
        """
        from datetime import datetime, timezone

        return CommunitySummary(
            community_id=community.community_id,
            level=community.level,
            title=f"Community {community.community_id} (generation failed)",
            summary=f"Summary generation failed for community {community.community_id}. Error: {error[:200]}",
            impact_rating=0.0,
            rating_explanation="Summary could not be generated.",
            findings=[CommunityFinding(
                finding_id=0,
                summary="Generation failed",
                explanation=f"Failed to generate summary: {error[:500]}",
            )],
            node_ids=community.node_ids,
            context_tokens_used=0,
            generated_at=datetime.now(tz=timezone.utc),
        )

    def get_stats(self, summaries: list[CommunitySummary]) -> dict:
        """Return summary statistics for a list of CommunitySummary objects."""
        if not summaries:
            return {"total": 0}

        by_level: dict[str, int] = {}
        total_tokens = 0
        total_findings = 0
        truncated = 0

        for s in summaries:
            lv = s.level.value if hasattr(s.level, "value") else str(s.level)
            by_level[lv] = by_level.get(lv, 0) + 1
            total_tokens += s.context_tokens_used
            total_findings += len(s.findings)
            if s.was_truncated:
                truncated += 1

        return {
            "total":                len(summaries),
            "by_level":             by_level,
            "avg_context_tokens":   round(total_tokens / len(summaries), 0),
            "avg_findings":         round(total_findings / len(summaries), 1),
            "truncated_count":      truncated,
            "truncation_rate":      round(truncated / len(summaries), 3),
        }


# ── Factory function ───────────────────────────────────────────────────────────

def get_summarization_pipeline() -> SummarizationPipeline:
    """Build a SummarizationPipeline from application settings."""
    from app.config import get_settings
    from app.services.openai_service import get_openai_service
    from app.services.tokenizer_service import get_tokenizer

    settings = get_settings()
    return SummarizationPipeline(
        openai_service=get_openai_service(),
        tokenizer=get_tokenizer(),
        context_window=settings.context_window_size,
    )


__all__ = [
    "SummarizationPipeline",
    "get_summarization_pipeline",
]