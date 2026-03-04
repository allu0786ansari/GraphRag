"""
models/evaluation_models.py — Evaluation pipeline data models.

Covers both evaluation methodologies from the paper:

Experiment 1: LLM-as-a-Judge (Section 4.1)
  - Present both answers (GraphRAG + VectorRAG) to a judge LLM
  - Score across 4 criteria: comprehensiveness, diversity, empowerment, directness
  - Run 5 times per question pair for stochasticity control
  - Report win rates as head-to-head percentages

Experiment 2: Claim-Based Validation (Section 4.2)
  - Use Claimify to extract atomic factual claims from each answer
  - Comprehensiveness = average number of unique claims per answer
  - Diversity = number of clusters (agglomerative + ROUGE-L, thresholds 0.5–0.8)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class Winner(str, Enum):
    """
    Head-to-head judgment result for a single criterion.

    The paper uses pairwise comparison: which answer is better?
    Tie is allowed when neither answer is clearly superior.
    """
    GRAPHRAG  = "graphrag"
    VECTORRAG = "vectorrag"
    TIE       = "tie"


class EvalCriterion(str, Enum):
    """
    The four evaluation criteria from the paper Section 4.1.

    Comprehensiveness: How much detail? Covers all aspects?
    Diversity:         How varied? Multiple perspectives?
    Empowerment:       Helps reader make informed judgments?
    Directness:        Concise and specific? (control — expected to favor VectorRAG)
    """
    COMPREHENSIVENESS = "comprehensiveness"
    DIVERSITY         = "diversity"
    EMPOWERMENT       = "empowerment"
    DIRECTNESS        = "directness"


# ── Experiment 1: LLM-as-a-Judge models ───────────────────────────────────────

class SingleJudgment(BaseModel):
    """
    Result of a single LLM judge call for one criterion on one question.

    The judge receives both answers (randomly assigned A/B to prevent
    position bias) and selects which is better for the given criterion.
    """

    criterion: EvalCriterion = Field(
        description="Which evaluation criterion was scored in this judgment.",
    )
    winner: Winner = Field(
        description="Which system the judge selected as better for this criterion.",
    )
    answer_a_system: str = Field(
        description="Which system was assigned to 'Answer A' in this run (graphrag or vectorrag).",
        examples=["graphrag"],
    )
    answer_b_system: str = Field(
        description="Which system was assigned to 'Answer B' in this run.",
        examples=["vectorrag"],
    )
    answer_a_score: int = Field(
        ge=0, le=100,
        description="Judge's score for Answer A on this criterion (0–100).",
        examples=[85],
    )
    answer_b_score: int = Field(
        ge=0, le=100,
        description="Judge's score for Answer B on this criterion (0–100).",
        examples=[62],
    )
    reasoning: str = Field(
        description="Judge's brief explanation for this judgment.",
        examples=["Answer A covers significantly more aspects of the question including cross-topic patterns..."],
    )
    run_index: int = Field(
        ge=0,
        description="Which evaluation run this judgment belongs to (0-indexed, max eval_runs-1).",
        examples=[0],
    )


class CriterionResult(BaseModel):
    """
    Aggregated result for a single criterion across all eval_runs for one question.

    Win rate is computed as: (runs won by graphrag) / (total decisive runs).
    Ties are excluded from win rate calculation (following paper methodology).
    """

    criterion: EvalCriterion = Field(
        description="The evaluated criterion.",
    )
    question: str = Field(
        description="The question this result is for.",
    )

    # ── Per-run results ───────────────────────────────────────────────────────
    judgments: list[SingleJudgment] = Field(
        description="All individual judgment calls for this criterion × question.",
    )

    # ── Aggregated stats ──────────────────────────────────────────────────────
    graphrag_wins: int = Field(
        ge=0,
        description="Number of runs where GraphRAG was judged better.",
    )
    vectorrag_wins: int = Field(
        ge=0,
        description="Number of runs where VectorRAG was judged better.",
    )
    ties: int = Field(
        ge=0,
        description="Number of runs that ended in a tie.",
    )
    total_runs: int = Field(
        ge=1,
        description="Total number of evaluation runs for this criterion × question.",
    )

    graphrag_win_rate: float = Field(
        ge=0.0, le=1.0,
        description=(
            "GraphRAG win rate: graphrag_wins / (graphrag_wins + vectorrag_wins). "
            "Ties are excluded. 1.0 = GraphRAG won all decisive runs."
        ),
        examples=[0.75],
    )
    avg_graphrag_score: float = Field(
        ge=0.0, le=100.0,
        description="Average judge score for GraphRAG answers across all runs.",
    )
    avg_vectorrag_score: float = Field(
        ge=0.0, le=100.0,
        description="Average judge score for VectorRAG answers across all runs.",
    )
    majority_winner: Winner = Field(
        description="The winner by majority vote across eval_runs.",
    )

    @model_validator(mode="after")
    def compute_win_rate(self) -> CriterionResult:
        decisive = self.graphrag_wins + self.vectorrag_wins
        if decisive == 0:
            self.graphrag_win_rate = 0.5  # all ties → 50%
        else:
            self.graphrag_win_rate = self.graphrag_wins / decisive
        return self

    @model_validator(mode="after")
    def compute_majority_winner(self) -> CriterionResult:
        if self.graphrag_wins > self.vectorrag_wins:
            self.majority_winner = Winner.GRAPHRAG
        elif self.vectorrag_wins > self.graphrag_wins:
            self.majority_winner = Winner.VECTORRAG
        else:
            self.majority_winner = Winner.TIE
        return self


class QuestionEvalResult(BaseModel):
    """
    Full evaluation result for a single question across all criteria.

    Contains both the answers produced by each system and the
    criterion-by-criterion judgment results.
    """

    question_id: int = Field(
        ge=0,
        description="Zero-based index of this question in the evaluation set.",
    )
    question: str = Field(
        description="The global sensemaking question.",
    )
    graphrag_answer: str = Field(
        description="Answer produced by GraphRAG for this question.",
    )
    vectorrag_answer: str = Field(
        description="Answer produced by VectorRAG for this question.",
    )
    community_level: str = Field(
        description="Community level used for the GraphRAG answer.",
        examples=["c1"],
    )

    criterion_results: list[CriterionResult] = Field(
        description="Judgment results for each evaluated criterion.",
    )

    @property
    def summary(self) -> dict[str, Winner]:
        """Quick-access map of criterion → majority_winner."""
        return {r.criterion.value: r.majority_winner for r in self.criterion_results}


class EvalSummaryStats(BaseModel):
    """
    Aggregate statistics across all questions for a single criterion.

    These are the headline numbers that match the paper's reported results
    (e.g. 'GraphRAG achieves 72–83% comprehensiveness win rate over VectorRAG').
    """

    criterion: EvalCriterion = Field(description="The aggregated criterion.")
    total_questions: int = Field(description="Number of questions evaluated.")

    graphrag_win_rate_avg: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Average GraphRAG win rate across all questions. "
            "Paper target: >0.72 for comprehensiveness, >0.62 for diversity."
        ),
        examples=[0.75],
    )
    graphrag_win_rate_std: float = Field(
        ge=0.0,
        description="Standard deviation of GraphRAG win rate across questions.",
    )
    graphrag_total_wins: int = Field(
        description="Total GraphRAG wins across all questions and runs.",
    )
    vectorrag_total_wins: int = Field(
        description="Total VectorRAG wins across all questions and runs.",
    )
    total_ties: int = Field(
        description="Total tied judgments across all questions and runs.",
    )
    p_value: float | None = Field(
        default=None,
        description=(
            "p-value from a one-sided binomial test: H0 = win rate ≤ 0.5. "
            "Paper reports p<0.001 for all significant GraphRAG wins."
        ),
        examples=[0.0001],
    )
    is_significant: bool | None = Field(
        default=None,
        description="True if p_value < 0.05.",
    )

    # ── Paper target benchmarks ────────────────────────────────────────────────
    meets_paper_target: bool | None = Field(
        default=None,
        description=(
            "True if GraphRAG win rate meets the paper's reported targets: "
            ">72% for comprehensiveness, >62% for diversity. "
            "None for empowerment and directness (no clear expected winner)."
        ),
    )


class EvalResponse(BaseModel):
    """
    POST /api/v1/evaluate

    Full LLM-as-a-Judge evaluation results.
    Contains per-question results and aggregate statistics.
    """

    evaluation_id: str = Field(
        description="Unique identifier for this evaluation run.",
        examples=["eval_a1b2c3d4"],
    )
    community_level: str = Field(
        description="Community level used for all GraphRAG answers.",
    )
    total_questions: int = Field(
        description="Number of questions evaluated.",
        examples=[125],
    )
    eval_runs_per_question: int = Field(
        description="LLM judge runs per question per criterion.",
        examples=[5],
    )
    criteria_evaluated: list[EvalCriterion] = Field(
        description="Which criteria were evaluated.",
    )

    question_results: list[QuestionEvalResult] = Field(
        description="Per-question evaluation results.",
    )
    summary_stats: list[EvalSummaryStats] = Field(
        description="Aggregate statistics per criterion across all questions.",
    )

    # ── Headline results (for dashboard display) ───────────────────────────────
    comprehensiveness_win_rate: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="GraphRAG comprehensiveness win rate. Target: >0.72.",
    )
    diversity_win_rate: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="GraphRAG diversity win rate. Target: >0.62.",
    )
    empowerment_win_rate: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="GraphRAG empowerment win rate (expected: mixed).",
    )
    directness_win_rate: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="GraphRAG directness win rate (expected: <0.5, VectorRAG wins).",
    )

    started_at: datetime = Field(description="UTC timestamp when evaluation started.")
    completed_at: datetime | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)
    total_token_usage: dict[str, int] | None = Field(
        default=None,
        description="Total LLM tokens consumed for the entire evaluation.",
    )


# ── Experiment 2: Claim-based validation models ────────────────────────────────

class ExtractedClaim(BaseModel):
    """
    A single atomic factual claim extracted from an answer by Claimify
    (Metropolitansky & Larson, 2025).

    Used in Experiment 2 for claim-based comprehensiveness and diversity metrics.
    """

    claim_id: str = Field(
        description="Unique identifier for this claim within its answer.",
        examples=["q001_graphrag_claim_003"],
    )
    text: str = Field(
        description="The atomic claim text.",
        examples=["OpenAI was founded in 2015 by Sam Altman and Elon Musk."],
    )
    source_system: str = Field(
        description="Which RAG system's answer this claim came from.",
        examples=["graphrag"],
    )
    question_id: int = Field(
        description="Which question's answer this claim was extracted from.",
    )
    cluster_id: int | None = Field(
        default=None,
        description=(
            "Cluster assignment from agglomerative clustering. "
            "Used to compute diversity as number of distinct clusters."
        ),
    )


class ClaimMetrics(BaseModel):
    """
    Claim-based comprehensiveness and diversity metrics for one system on one question.

    Comprehensiveness = number of unique claims (after deduplication).
    Diversity = number of distinct claim clusters at a given ROUGE-L threshold.

    Paper (Section 4.2) uses thresholds 0.5–0.8 and reports consistent results.
    """

    question_id: int = Field(description="The question these metrics are for.")
    question: str = Field(description="The question text.")
    system: str = Field(
        description="Which RAG system these metrics are for.",
        examples=["graphrag"],
    )
    answer: str = Field(description="The answer from which claims were extracted.")

    claims: list[ExtractedClaim] = Field(
        description="All atomic claims extracted from this answer.",
    )
    unique_claim_count: int = Field(
        ge=0,
        description=(
            "Number of unique claims after deduplication. "
            "This IS the comprehensiveness metric."
        ),
        examples=[23],
    )

    # ── Diversity metrics at multiple ROUGE-L thresholds ──────────────────────
    cluster_count_threshold_05: int | None = Field(
        default=None,
        description="Number of claim clusters at ROUGE-L distance threshold 0.5.",
        examples=[12],
    )
    cluster_count_threshold_06: int | None = Field(
        default=None,
        description="Number of claim clusters at ROUGE-L distance threshold 0.6.",
    )
    cluster_count_threshold_07: int | None = Field(
        default=None,
        description="Number of claim clusters at ROUGE-L distance threshold 0.7.",
    )
    cluster_count_threshold_08: int | None = Field(
        default=None,
        description="Number of claim clusters at ROUGE-L distance threshold 0.8.",
    )


class ClaimComparisonResult(BaseModel):
    """
    Side-by-side claim metrics for GraphRAG vs VectorRAG on a single question.
    """

    question_id: int = Field(description="The question being compared.")
    question: str = Field(description="The question text.")
    graphrag_metrics: ClaimMetrics = Field(description="Claim metrics for GraphRAG.")
    vectorrag_metrics: ClaimMetrics = Field(description="Claim metrics for VectorRAG.")

    comprehensiveness_winner: str = Field(
        description="Which system has more unique claims.",
        examples=["graphrag"],
    )
    comprehensiveness_delta: int = Field(
        description=(
            "Difference in unique claim count: "
            "graphrag_claims - vectorrag_claims. "
            "Positive means GraphRAG is more comprehensive."
        ),
        examples=[8],
    )
    diversity_winner_threshold_07: str | None = Field(
        default=None,
        description="Which system has more claim clusters at threshold 0.7.",
    )


class ClaimEvalResponse(BaseModel):
    """
    POST /api/v1/evaluate/claims

    Full claim-based evaluation results (Experiment 2 from the paper).
    """

    evaluation_id: str = Field(description="Unique identifier for this evaluation.")
    total_questions: int = Field(description="Number of questions evaluated.")

    question_results: list[ClaimComparisonResult] = Field(
        description="Per-question claim comparison results.",
    )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    avg_graphrag_claims: float = Field(
        ge=0.0,
        description=(
            "Average unique claims per answer for GraphRAG across all questions. "
            "Higher = more comprehensive per paper Experiment 2."
        ),
        examples=[21.4],
    )
    avg_vectorrag_claims: float = Field(
        ge=0.0,
        description="Average unique claims per answer for VectorRAG.",
        examples=[13.2],
    )

    avg_graphrag_clusters_07: float | None = Field(
        default=None,
        description="Average claim clusters for GraphRAG at ROUGE-L threshold 0.7.",
    )
    avg_vectorrag_clusters_07: float | None = Field(
        default=None,
        description="Average claim clusters for VectorRAG at ROUGE-L threshold 0.7.",
    )

    graphrag_comprehensiveness_win_rate: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Fraction of questions where GraphRAG had more unique claims than VectorRAG. "
            "Paper expects >0.9 (all GraphRAG conditions significantly beat SS)."
        ),
        examples=[0.92],
    )
    graphrag_diversity_win_rate_07: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Fraction of questions where GraphRAG had more clusters at threshold 0.7.",
    )

    p_value_comprehensiveness: float | None = Field(
        default=None,
        description="p-value for comprehensiveness difference (Wilcoxon signed-rank test).",
    )
    p_value_diversity: float | None = Field(
        default=None,
        description="p-value for diversity difference.",
    )

    started_at: datetime = Field(description="UTC timestamp when evaluation started.")
    completed_at: datetime | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)


# ── Persona and question generation models ────────────────────────────────────

class UserPersona(BaseModel):
    """
    A generated user persona for question generation (Algorithm 1, paper Section 3.3).
    """

    persona_id: int = Field(ge=0, description="Zero-based index.")
    role: str = Field(
        description="Professional role or identity of this persona.",
        examples=["Technology journalist", "Public health researcher"],
    )
    background: str = Field(
        description="Brief background of this persona.",
        examples=["Covers AI industry developments for a major tech publication."],
    )
    goals: list[str] = Field(
        description="What this persona wants to understand from the corpus.",
        min_length=1,
    )
    tasks: list[str] = Field(
        description="Specific tasks this persona would perform using the RAG system.",
        min_length=1,
    )
    questions: list[str] = Field(
        description=(
            "Global sensemaking questions generated for this persona. "
            "Each question requires whole-corpus understanding."
        ),
        default_factory=list,
    )

    @field_validator("questions")
    @classmethod
    def questions_must_be_global(cls, v: list[str]) -> list[str]:
        """
        Basic validation that questions are not trivially specific.
        Full validation (requiring whole-corpus understanding) is
        enforced by the LLM generation prompt.
        """
        for q in v:
            if len(q.strip()) < 10:
                raise ValueError(f"Question too short: '{q}'")
        return v