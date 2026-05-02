"""
core/query/claim_validation.py — Claimify claim extraction + ROUGE-L diversity clustering.

Implements Experiment 2 from the paper (Section 4.2):

  Step 1 — Claim extraction (Claimify):
    Use an LLM to decompose each answer into a list of atomic, verifiable
    factual claims. Atomic = cannot be split further. Verifiable = could in
    principle be checked against a source.

    Based on: Metropolitansky & Larson (2025), "Claimify: Extracting
    Atomic Claims from LLM Outputs."

  Step 2 — Deduplication:
    Remove duplicate claims (exact or near-exact matches) using ROUGE-L
    similarity. Claims with ROUGE-L similarity > 0.85 are considered
    duplicates — keep only one.

  Step 3 — ROUGE-L clustering for diversity:
    Apply agglomerative clustering using ROUGE-L distance as the metric.
    Number of clusters = diversity score for that answer.
    Thresholds 0.5, 0.6, 0.7, 0.8 are tested (paper Figure 5 shows
    consistent results across all thresholds).

  Comprehensiveness metric = unique_claim_count (after deduplication).
  Diversity metric = cluster_count at a given ROUGE-L threshold.

Paper reference (Section 4.2):
  "We use Claimify to extract a set of atomic claims from each answer.
   Comprehensiveness is measured as the average number of unique claims.
   Diversity is measured as the number of distinct claim clusters using
   agglomerative clustering with ROUGE-L distance."

ROUGE-L implementation:
  We implement a pure-Python ROUGE-L scorer without requiring the `rouge`
  package (which has C extension issues on some platforms). This matches
  the token-level F1 score used in the paper.

Usage:
    engine = ClaimValidationEngine.from_settings()

    # Extract and cluster claims for one answer
    metrics = await engine.extract_and_cluster(
        question_id=0,
        question="What are the main themes?",
        answer=graphrag_answer,
        system="graphrag",
    )
    print(f"Unique claims: {metrics.unique_claim_count}")
    print(f"Clusters (threshold 0.7): {metrics.cluster_count_threshold_07}")

    # Full side-by-side comparison
    comparison = await engine.compare(
        question_id=0,
        question="What are the main themes?",
        graphrag_answer=graphrag_ans,
        vectorrag_answer=vectorrag_ans,
    )
    print(f"Winner: {comparison.comprehensiveness_winner}")
    print(f"Delta: {comparison.comprehensiveness_delta}")
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from app.models.evaluation_models import (
    ClaimComparisonResult,
    ClaimEvalResponse,
    ClaimMetrics,
    ExtractedClaim,
)
from app.services.llm_service import LLMService
from app.services.tokenizer_service import TokenizerService
from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Prompt templates ───────────────────────────────────────────────────────────

_CLAIM_SYSTEM_PROMPT = (
    "You are an expert at extracting atomic, verifiable factual claims "
    "from text. "
    "An atomic claim is a single, indivisible factual statement that can "
    "be evaluated as true or false. "
    "Respond ONLY with valid JSON — no preamble, no markdown fences."
)

_CLAIM_USER_PROMPT = """\
Extract ALL atomic factual claims from the following answer.

Rules for atomic claims:
1. Each claim must be a SINGLE factual statement (one subject, one predicate).
2. Each claim must be self-contained — no pronouns without referents.
3. Each claim must be factual and verifiable (not an opinion or summary).
4. Do NOT split a claim into sub-claims if it is already atomic.
5. Include ALL factual claims, even if they seem minor.
6. Exclude meta-statements like "The answer covers..." or "Based on the corpus..."

Question context: {question}

Answer:
{answer}

Respond with this exact JSON format:
{{
  "claims": [
    "Atomic claim text 1.",
    "Atomic claim text 2.",
    ...
  ]
}}

Extract every verifiable factual statement. Aim for 10-30 claims for a \
comprehensive answer."""

# Deduplication threshold: claims with ROUGE-L > this are considered duplicates
_DEDUP_THRESHOLD = 0.85

# Clustering thresholds to test (paper Section 4.2)
_CLUSTER_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


class ClaimValidationEngine:
    """
    Claim extraction and diversity clustering engine (paper Experiment 2).

    Extracts atomic claims from answers using an LLM, deduplicates them
    using ROUGE-L similarity, then clusters them at multiple thresholds
    to produce comprehensiveness and diversity metrics.

    Usage:
        engine = ClaimValidationEngine.from_settings()
        comparison = await engine.compare(
            question_id=0,
            question="What are the main themes?",
            graphrag_answer="...",
            vectorrag_answer="...",
        )
    """

    def __init__(
        self,
        openai_service: LLMService,
        tokenizer: TokenizerService,
        max_concurrency: int = 10,
        dedup_threshold: float = _DEDUP_THRESHOLD,
    ) -> None:
        """
        Args:
            openai_service:    For claim extraction LLM calls.
            tokenizer:         For answer truncation before extraction.
            max_concurrency:   Max concurrent LLM extraction calls.
            dedup_threshold:   ROUGE-L threshold above which claims are
                               considered duplicates. Default: 0.85.
        """
        self.openai_service    = openai_service
        self.tokenizer         = tokenizer
        self.max_concurrency   = max_concurrency
        self.dedup_threshold   = dedup_threshold

    # ── Public interface ───────────────────────────────────────────────────────

    async def extract_and_cluster(
        self,
        question_id: int,
        question: str,
        answer: str,
        system: str,
    ) -> ClaimMetrics:
        """
        Extract atomic claims from a single answer and compute metrics.

        Args:
            question_id: Index of the question.
            question:    The question text (used as context for extraction).
            answer:      The RAG answer to analyse.
            system:      Which system produced this answer ("graphrag"/"vectorrag").

        Returns:
            ClaimMetrics with unique claim count and cluster counts at
            thresholds 0.5, 0.6, 0.7, 0.8.
        """
        # 1. Extract raw claims via LLM
        raw_claims = await self._extract_claims_llm(question=question, answer=answer)

        # 2. Deduplicate using ROUGE-L
        unique_claim_texts = self._deduplicate_claims(raw_claims)

        # 3. Build ExtractedClaim objects
        claims = [
            ExtractedClaim(
                claim_id=f"q{question_id:03d}_{system}_claim_{i:03d}",
                text=text,
                source_system=system,
                question_id=question_id,
                cluster_id=None,  # assigned below
            )
            for i, text in enumerate(unique_claim_texts)
        ]

        # 4. Cluster at each threshold and report counts
        cluster_counts: dict[float, int] = {}
        cluster_assignments: dict[float, list[int]] = {}

        for threshold in _CLUSTER_THRESHOLDS:
            labels = self._cluster_claims(unique_claim_texts, threshold)
            n_clusters = len(set(labels)) if labels else 0
            cluster_counts[threshold] = n_clusters
            cluster_assignments[threshold] = labels

        # Assign cluster_id from the 0.7 threshold (representative middle value)
        if cluster_assignments.get(0.7):
            for i, claim in enumerate(claims):
                claim.cluster_id = cluster_assignments[0.7][i] if i < len(cluster_assignments[0.7]) else 0

        log.info(
            "Claim extraction complete",
            system=system,
            question_id=question_id,
            raw_claims=len(raw_claims),
            unique_claims=len(unique_claim_texts),
            clusters_07=cluster_counts.get(0.7, 0),
        )

        return ClaimMetrics(
            question_id=question_id,
            question=question,
            system=system,
            answer=answer,
            claims=claims,
            unique_claim_count=len(unique_claim_texts),
            cluster_count_threshold_05=cluster_counts.get(0.5),
            cluster_count_threshold_06=cluster_counts.get(0.6),
            cluster_count_threshold_07=cluster_counts.get(0.7),
            cluster_count_threshold_08=cluster_counts.get(0.8),
        )

    async def compare(
        self,
        question_id: int,
        question: str,
        graphrag_answer: str,
        vectorrag_answer: str,
    ) -> ClaimComparisonResult:
        """
        Side-by-side claim comparison for GraphRAG vs VectorRAG on one question.

        Runs extraction for both systems concurrently.

        Returns:
            ClaimComparisonResult with per-system metrics and winner declaration.
        """
        graphrag_metrics, vectorrag_metrics = await gather_with_concurrency(
            coroutines=[
                self.extract_and_cluster(
                    question_id=question_id,
                    question=question,
                    answer=graphrag_answer,
                    system="graphrag",
                ),
                self.extract_and_cluster(
                    question_id=question_id,
                    question=question,
                    answer=vectorrag_answer,
                    system="vectorrag",
                ),
            ],
            max_concurrency=2,
        )

        delta = graphrag_metrics.unique_claim_count - vectorrag_metrics.unique_claim_count
        comp_winner = (
            "graphrag" if delta > 0
            else "vectorrag" if delta < 0
            else "tie"
        )

        # Diversity winner at threshold 0.7 (primary threshold)
        div_winner_07 = None
        if (graphrag_metrics.cluster_count_threshold_07 is not None
                and vectorrag_metrics.cluster_count_threshold_07 is not None):
            g_clusters = graphrag_metrics.cluster_count_threshold_07
            v_clusters = vectorrag_metrics.cluster_count_threshold_07
            if g_clusters > v_clusters:
                div_winner_07 = "graphrag"
            elif v_clusters > g_clusters:
                div_winner_07 = "vectorrag"
            else:
                div_winner_07 = "tie"

        log.info(
            "Claim comparison complete",
            question_id=question_id,
            graphrag_claims=graphrag_metrics.unique_claim_count,
            vectorrag_claims=vectorrag_metrics.unique_claim_count,
            delta=delta,
            comprehensiveness_winner=comp_winner,
        )

        return ClaimComparisonResult(
            question_id=question_id,
            question=question,
            graphrag_metrics=graphrag_metrics,
            vectorrag_metrics=vectorrag_metrics,
            comprehensiveness_winner=comp_winner,
            comprehensiveness_delta=delta,
            diversity_winner_threshold_07=div_winner_07,
        )

    async def evaluate_batch(
        self,
        questions: list[str],
        graphrag_answers: list[str],
        vectorrag_answers: list[str],
    ) -> ClaimEvalResponse:
        """
        Run full claim evaluation on a batch of question/answer pairs.

        Args:
            questions:         List of question texts.
            graphrag_answers:  GraphRAG answers, same order as questions.
            vectorrag_answers: VectorRAG answers, same order as questions.

        Returns:
            ClaimEvalResponse with per-question comparisons and aggregate metrics.
        """
        import time
        started_at = datetime.now(timezone.utc)
        t0 = time.monotonic()

        comparisons = await gather_with_concurrency(
            coroutines=[
                self.compare(
                    question_id=i,
                    question=q,
                    graphrag_answer=ga,
                    vectorrag_answer=va,
                )
                for i, (q, ga, va) in enumerate(zip(questions, graphrag_answers, vectorrag_answers))
            ],
            max_concurrency=max(1, self.max_concurrency // 2),
            return_exceptions=False,
        )

        # Aggregate
        avg_graphrag = (
            sum(c.graphrag_metrics.unique_claim_count for c in comparisons)
            / len(comparisons)
        )
        avg_vectorrag = (
            sum(c.vectorrag_metrics.unique_claim_count for c in comparisons)
            / len(comparisons)
        )

        graphrag_comp_wins = sum(
            1 for c in comparisons if c.comprehensiveness_winner == "graphrag"
        )
        graphrag_comp_win_rate = graphrag_comp_wins / len(comparisons)

        # Diversity win rate at threshold 0.7
        div_comparisons = [c for c in comparisons if c.diversity_winner_threshold_07 is not None]
        graphrag_div_wins = sum(
            1 for c in div_comparisons if c.diversity_winner_threshold_07 == "graphrag"
        )
        avg_graphrag_clusters_07 = (
            sum(c.graphrag_metrics.cluster_count_threshold_07 or 0 for c in comparisons)
            / len(comparisons)
        )
        avg_vectorrag_clusters_07 = (
            sum(c.vectorrag_metrics.cluster_count_threshold_07 or 0 for c in comparisons)
            / len(comparisons)
        )
        graphrag_div_win_rate = (
            graphrag_div_wins / len(div_comparisons) if div_comparisons else None
        )

        # Optional Wilcoxon test for statistical significance
        p_comp, p_div = None, None
        try:
            from scipy.stats import wilcoxon  # type: ignore
            graphrag_claim_counts  = [c.graphrag_metrics.unique_claim_count  for c in comparisons]
            vectorrag_claim_counts = [c.vectorrag_metrics.unique_claim_count for c in comparisons]
            diffs = [g - v for g, v in zip(graphrag_claim_counts, vectorrag_claim_counts)]
            if any(d != 0 for d in diffs):
                stat, p_comp = wilcoxon(diffs, alternative="greater")
                p_comp = float(p_comp)
        except ImportError:
            pass
        except Exception:
            pass

        elapsed = round(time.monotonic() - t0, 2)

        log.info(
            "Claim evaluation batch complete",
            total_questions=len(questions),
            avg_graphrag_claims=round(avg_graphrag, 1),
            avg_vectorrag_claims=round(avg_vectorrag, 1),
            comprehensiveness_win_rate=f"{graphrag_comp_win_rate:.1%}",
            elapsed_seconds=elapsed,
        )

        return ClaimEvalResponse(
            evaluation_id=f"claim_eval_{uuid.uuid4().hex[:8]}",
            total_questions=len(questions),
            question_results=comparisons,
            avg_graphrag_claims=round(avg_graphrag, 2),
            avg_vectorrag_claims=round(avg_vectorrag, 2),
            avg_graphrag_clusters_07=round(avg_graphrag_clusters_07, 2),
            avg_vectorrag_clusters_07=round(avg_vectorrag_clusters_07, 2),
            graphrag_comprehensiveness_win_rate=round(graphrag_comp_win_rate, 4),
            graphrag_diversity_win_rate_07=graphrag_div_win_rate,
            p_value_comprehensiveness=p_comp,
            p_value_diversity=None,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=elapsed,
        )

    # ── LLM claim extraction ───────────────────────────────────────────────────

    async def _extract_claims_llm(
        self,
        question: str,
        answer: str,
        max_answer_tokens: int = 3000,
    ) -> list[str]:
        """
        Call the LLM to extract atomic claims from an answer.

        Truncates the answer if it exceeds max_answer_tokens to keep the
        extraction prompt within the 8k context window.

        Returns:
            List of raw claim strings (before deduplication).
        """
        truncated = answer
        if self.tokenizer.count_tokens(answer) > max_answer_tokens:
            truncated = self.tokenizer.truncate_to_limit(answer, max_answer_tokens)
            log.debug(
                "Answer truncated for claim extraction",
                original_tokens=self.tokenizer.count_tokens(answer),
                truncated_tokens=max_answer_tokens,
            )

        prompt = _CLAIM_USER_PROMPT.format(question=question, answer=truncated)

        try:
            result = await self.openai_service.complete(
                user_prompt=prompt,
                system_prompt=_CLAIM_SYSTEM_PROMPT,
                json_mode=True,
            )
            return self._parse_claims_response(result.content)
        except Exception as e:
            log.warning("Claim extraction LLM call failed", error=str(e))
            return []

    def _parse_claims_response(self, response_text: str) -> list[str]:
        """
        Parse the claim extraction LLM response.

        Expected: {"claims": ["Claim 1.", "Claim 2.", ...]}
        Returns a list of non-empty claim strings.
        """
        text = response_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, dict):
            return []

        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list):
            return []

        claims: list[str] = []
        for item in raw_claims:
            if isinstance(item, str):
                claim = item.strip()
                if len(claim) > 10:  # skip trivially short strings
                    claims.append(claim)

        return claims

    # ── Deduplication ──────────────────────────────────────────────────────────

    def _deduplicate_claims(self, claims: list[str]) -> list[str]:
        """
        Remove near-duplicate claims using ROUGE-L similarity.

        For each claim, compute ROUGE-L against all already-accepted claims.
        If any accepted claim has similarity > dedup_threshold, skip this claim.

        Time complexity: O(n²) where n = number of claims.
        For typical answer sizes (10-50 claims), this is fast (<10ms).

        Returns:
            List of unique claim strings (preserving first occurrence).
        """
        if not claims:
            return []

        unique: list[str] = [claims[0]]

        for candidate in claims[1:]:
            is_duplicate = False
            for accepted in unique:
                if rouge_l_f1(candidate, accepted) > self.dedup_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(candidate)

        log.debug(
            "Claim deduplication",
            before=len(claims),
            after=len(unique),
            removed=len(claims) - len(unique),
        )

        return unique

    # ── ROUGE-L clustering ─────────────────────────────────────────────────────

    def _cluster_claims(
        self,
        claims: list[str],
        distance_threshold: float,
    ) -> list[int]:
        """
        Cluster claims using agglomerative clustering with ROUGE-L distance.

        ROUGE-L distance = 1 - ROUGE-L F1 similarity.
        distance_threshold is the maximum inter-cluster distance.

        Uses a simple greedy single-linkage approach:
          - Start: each claim is its own cluster.
          - Merge: if two claims have ROUGE-L similarity > (1 - distance_threshold),
            assign the candidate to the cluster of its most similar existing member.
        This matches the paper's description without requiring scipy clustering
        (which requires a full pairwise distance matrix for small n, acceptable here).

        Args:
            claims:             List of deduplicated claim texts.
            distance_threshold: Maximum ROUGE-L distance for merging.
                                0.5 means merge if similarity > 0.5.

        Returns:
            List of cluster label integers, one per claim.
            Labels are 0-indexed. Unique label count = diversity score.
        """
        if not claims:
            return []
        if len(claims) == 1:
            return [0]

        similarity_threshold = 1.0 - distance_threshold  # convert distance to similarity

        # Greedy single-linkage: each claim checked against cluster representatives
        labels: list[int] = [0]  # first claim gets cluster 0
        cluster_representatives: dict[int, list[str]] = {0: [claims[0]]}
        next_cluster_id = 1

        for i, candidate in enumerate(claims[1:], 1):
            best_cluster = None
            best_sim = -1.0

            for cluster_id, members in cluster_representatives.items():
                # Check similarity against all members (complete linkage variant)
                for member in members:
                    sim = rouge_l_f1(candidate, member)
                    if sim > best_sim:
                        best_sim = sim
                        if sim >= similarity_threshold:
                            best_cluster = cluster_id

            if best_cluster is not None:
                labels.append(best_cluster)
                cluster_representatives[best_cluster].append(candidate)
            else:
                # New cluster
                labels.append(next_cluster_id)
                cluster_representatives[next_cluster_id] = [candidate]
                next_cluster_id += 1

        return labels

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> "ClaimValidationEngine":
        """Build a ClaimValidationEngine from application settings."""
        from app.config import get_settings
        settings = get_settings()
        from app.services.llm_service import LLMService
        from app.services.tokenizer_service import TokenizerService

        llm_svc = LLMService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            max_tokens=settings.openai_max_tokens,
            temperature=0.0,
        )
        tokenizer = TokenizerService(model=settings.gemini_model)
        return cls(openai_service=llm_svc, tokenizer=tokenizer)


# ── ROUGE-L implementation (pure Python, no external dependencies) ────────────

def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between two strings.

    ROUGE-L measures the longest common subsequence (LCS) at the word level.
    F1 = 2 × Precision × Recall / (Precision + Recall)
    where:
      Precision = LCS_length / len(hypothesis_tokens)
      Recall    = LCS_length / len(reference_tokens)

    This is a pure-Python implementation of the standard token-level ROUGE-L,
    matching the metric used in the paper's diversity evaluation.

    Args:
        hypothesis: The claim or text being evaluated.
        reference:  The reference text.

    Returns:
        ROUGE-L F1 score in [0.0, 1.0]. 1.0 = identical.
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(hyp_tokens, ref_tokens)

    precision = lcs_len / len(hyp_tokens)
    recall    = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer for ROUGE-L.

    Lowercases and strips punctuation from word boundaries.
    """
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _lcs_length(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    """
    Compute the length of the Longest Common Subsequence (LCS) of two sequences.

    Uses the standard dynamic programming algorithm.
    Time: O(n × m), Space: O(min(n, m)).

    For typical claim lengths (10-30 words), this is fast (<0.1ms).
    """
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    # Space-optimized: only keep two rows
    # Ensure seq2 is the shorter sequence for space efficiency
    if m < n:
        seq1, seq2 = seq2, seq1
        m, n = n, m

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def get_claim_validation_engine() -> ClaimValidationEngine:
    """FastAPI dependency: return a ClaimValidationEngine built from settings."""
    return ClaimValidationEngine.from_settings()


__all__ = [
    "ClaimValidationEngine",
    "get_claim_validation_engine",
    "rouge_l_f1",
    "_lcs_length",
    "_tokenize",
]