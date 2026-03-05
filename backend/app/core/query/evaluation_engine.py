"""
core/query/evaluation_engine.py — LLM-as-a-judge evaluation (paper Section 4.1).

Implements Experiment 1 from the paper:

  For each question:
    1. Generate both GraphRAG and VectorRAG answers.
    2. Run the judge LLM eval_runs times (paper: 5) on each pair.
    3. In each run: randomly assign A/B positions to prevent position bias.
    4. Ask the judge to score BOTH answers on ONE criterion and declare a winner.
    5. Aggregate across runs by majority vote.
    6. Report per-criterion win rates: graphrag_wins / (graphrag_wins + vectorrag_wins).
       Ties are excluded from win-rate computation (paper methodology).

Four evaluation criteria (paper Section 4.1):
  Comprehensiveness: Does the answer cover all aspects of the question in detail?
  Diversity:         Does the answer present multiple perspectives or categories?
  Empowerment:       Does the answer help the reader make informed judgments?
  Directness:        Is the answer specific and concise? (control — favors VectorRAG)

Position bias prevention:
  The judge LLM may prefer 'Answer A' regardless of quality (position bias).
  We randomly swap A/B assignment between runs and correct for it in scoring.
  This is the paper's exact methodology.

Judge prompt design:
  The judge receives both answers without knowing which system produced them.
  It returns JSON: {"winner": "A"|"B"|"TIE", "score_a": int, "score_b": int, "reasoning": str}
  This structure is from the paper's evaluation protocol (Appendix C).

Statistical reporting:
  Win rates are computed as graphrag_wins / (total_decisive_runs * total_questions).
  The paper targets: comprehensiveness >72%, diversity >62%.

Usage:
    engine = EvaluationEngine.from_settings()
    result = await engine.evaluate(
        questions=["What are the main themes?"],
        criteria=["comprehensiveness", "diversity", "empowerment", "directness"],
        community_level="c1",
        eval_runs=5,
    )
    for criterion_result in result.summary_stats:
        print(f"{criterion_result.criterion}: {criterion_result.graphrag_win_rate_avg:.0%}")
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from datetime import datetime, timezone

from app.core.query.graphrag_engine import GraphRAGEngine
from app.core.query.vectorrag_engine import VectorRAGEngine
from app.models.evaluation_models import (
    CriterionResult,
    EvalCriterion,
    EvalResponse,
    EvalSummaryStats,
    QuestionEvalResult,
    SingleJudgment,
    Winner,
)
from app.services.openai_service import OpenAIService
from app.services.tokenizer_service import TokenizerService
from app.utils.async_utils import gather_with_concurrency
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Judge prompt templates (paper Appendix C) ──────────────────────────────────

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial expert evaluating two AI-generated answers to a question "
    "about a large document corpus. "
    "Your task is to compare the two answers on a specific criterion and "
    "determine which is better. "
    "Respond ONLY with valid JSON — no preamble, no markdown fences."
)

_JUDGE_USER_PROMPT = """\
Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluation Criterion: {criterion_name}
Criterion Definition: {criterion_definition}

For this criterion, evaluate BOTH answers and respond with JSON in this exact format:
{{
  "winner": "A" or "B" or "TIE",
  "score_a": <integer 0-100>,
  "score_b": <integer 0-100>,
  "reasoning": "<brief explanation of your judgment, 1-3 sentences>"
}}

Scoring guidance:
- winner: "A" if Answer A is clearly better, "B" if Answer B is clearly better,
          "TIE" only if they are genuinely equal on this criterion.
- score_a / score_b: 0-100 quality scores for each answer on this criterion.
  A higher-scoring answer should be the winner.
- reasoning: Explain what makes the winner better (or why they are tied).

Important: Focus ONLY on the {criterion_name} criterion. Ignore other qualities."""

# Criterion definitions (paper Section 4.1)
_CRITERION_DEFINITIONS = {
    EvalCriterion.COMPREHENSIVENESS: (
        "How thoroughly does the answer address all aspects of the question? "
        "A comprehensive answer covers multiple dimensions, provides details, "
        "and does not leave out important aspects of the topic."
    ),
    EvalCriterion.DIVERSITY: (
        "How well does the answer represent multiple different perspectives, "
        "categories, or viewpoints? A diverse answer goes beyond a single angle "
        "and acknowledges the variety of relevant entities, themes, or opinions."
    ),
    EvalCriterion.EMPOWERMENT: (
        "How well does the answer help the reader understand the topic deeply "
        "enough to make informed decisions or judgments? An empowering answer "
        "provides context, explains significance, and enables the reader to reason "
        "further about the topic independently."
    ),
    EvalCriterion.DIRECTNESS: (
        "How directly and concisely does the answer address the question? "
        "A direct answer gets to the point quickly, avoids unnecessary padding, "
        "and provides specific information rather than vague generalities."
    ),
}


class EvaluationEngine:
    """
    LLM-as-a-judge evaluation engine (paper Section 4.1, Experiment 1).

    Orchestrates:
      1. Answer generation from both GraphRAG and VectorRAG.
      2. Repeated judge LLM calls per criterion × question pair.
      3. Aggregation into win rates, majority votes, and summary statistics.

    Usage:
        engine = EvaluationEngine.from_settings()
        result = await engine.evaluate(
            questions=["What are the main themes?"],
            eval_runs=5,
        )
    """

    def __init__(
        self,
        openai_service: OpenAIService,
        graphrag_engine: GraphRAGEngine,
        vectorrag_engine: VectorRAGEngine,
        tokenizer: TokenizerService,
        max_concurrency: int = 10,
        randomize_answer_order: bool = True,
    ) -> None:
        """
        Args:
            openai_service:         For judge LLM calls.
            graphrag_engine:        For generating GraphRAG answers.
            vectorrag_engine:       For generating VectorRAG answers.
            tokenizer:              For token counting (answer truncation).
            max_concurrency:        Max concurrent judge calls across all questions/runs.
            randomize_answer_order: If True, randomly swap A/B to prevent position bias.
        """
        self.openai_service        = openai_service
        self.graphrag_engine       = graphrag_engine
        self.vectorrag_engine      = vectorrag_engine
        self.tokenizer             = tokenizer
        self.max_concurrency       = max_concurrency
        self.randomize_answer_order = randomize_answer_order

    # ── Public interface ───────────────────────────────────────────────────────

    async def evaluate(
        self,
        questions: list[str],
        criteria: list[str | EvalCriterion] | None = None,
        community_level: str = "c1",
        eval_runs: int = 5,
        randomize_answer_order: bool | None = None,
    ) -> EvalResponse:
        """
        Run the full evaluation pipeline on a list of questions.

        For each question:
          - Generate GraphRAG + VectorRAG answers.
          - For each criterion × each run: call the judge LLM.
          - Aggregate into win rates and majority votes.

        Args:
            questions:             List of global sensemaking questions.
            criteria:              Criteria to evaluate. Defaults to all four.
            community_level:       GraphRAG community level for answer generation.
            eval_runs:             Judge runs per question × criterion. Paper: 5.
            randomize_answer_order: Override the engine's default A/B randomization.

        Returns:
            EvalResponse with per-question results and aggregate statistics.
        """
        started_at = datetime.now(timezone.utc)
        t0 = time.monotonic()

        # Resolve criteria
        resolved_criteria = self._resolve_criteria(criteria)

        do_randomize = (
            randomize_answer_order
            if randomize_answer_order is not None
            else self.randomize_answer_order
        )

        log.info(
            "Evaluation started",
            total_questions=len(questions),
            criteria=[c.value for c in resolved_criteria],
            eval_runs=eval_runs,
            community_level=community_level,
        )

        # 1. Generate both answers for all questions (concurrently)
        answer_pairs = await self._generate_answer_pairs(
            questions=questions,
            community_level=community_level,
        )

        # 2. For each question: run judge on all criteria × all runs
        question_results: list[QuestionEvalResult] = []

        for q_idx, (question, (graphrag_ans, vectorrag_ans)) in enumerate(
            zip(questions, answer_pairs)
        ):
            q_result = await self._evaluate_question(
                question_id=q_idx,
                question=question,
                graphrag_answer=graphrag_ans,
                vectorrag_answer=vectorrag_ans,
                criteria=resolved_criteria,
                eval_runs=eval_runs,
                community_level=community_level,
                randomize=do_randomize,
            )
            question_results.append(q_result)
            log.info(
                "Question evaluated",
                question_id=q_idx,
                summary={c: w.value for c, w in q_result.summary.items()},
            )

        # 3. Compute aggregate summary statistics per criterion
        summary_stats = self._compute_summary_stats(
            question_results=question_results,
            criteria=resolved_criteria,
            eval_runs=eval_runs,
        )

        elapsed = round(time.monotonic() - t0, 2)
        completed_at = datetime.now(timezone.utc)

        # 4. Extract headline win rates
        headline = {s.criterion: s.graphrag_win_rate_avg for s in summary_stats}

        log.info(
            "Evaluation complete",
            total_questions=len(questions),
            elapsed_seconds=elapsed,
            comprehensiveness=f"{headline.get(EvalCriterion.COMPREHENSIVENESS, 0):.1%}",
            diversity=f"{headline.get(EvalCriterion.DIVERSITY, 0):.1%}",
        )

        return EvalResponse(
            evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
            community_level=community_level,
            total_questions=len(questions),
            eval_runs_per_question=eval_runs,
            criteria_evaluated=resolved_criteria,
            question_results=question_results,
            summary_stats=summary_stats,
            comprehensiveness_win_rate=headline.get(EvalCriterion.COMPREHENSIVENESS),
            diversity_win_rate=headline.get(EvalCriterion.DIVERSITY),
            empowerment_win_rate=headline.get(EvalCriterion.EMPOWERMENT),
            directness_win_rate=headline.get(EvalCriterion.DIRECTNESS),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=elapsed,
        )

    # ── Answer generation ──────────────────────────────────────────────────────

    async def _generate_answer_pairs(
        self,
        questions: list[str],
        community_level: str,
    ) -> list[tuple[str, str]]:
        """
        Generate GraphRAG + VectorRAG answers for all questions concurrently.

        Returns list of (graphrag_answer, vectorrag_answer) tuples,
        in the same order as `questions`.
        """
        async def _both_answers(question: str) -> tuple[str, str]:
            graphrag_result, vectorrag_result = await gather_with_concurrency(
                coroutines=[
                    self.graphrag_engine.query(
                        query=question,
                        community_level=community_level,
                        include_context=False,
                        include_token_usage=False,
                    ),
                    self.vectorrag_engine.query(
                        query=question,
                        include_context=False,
                        include_token_usage=False,
                    ),
                ],
                max_concurrency=2,
            )
            return graphrag_result.answer, vectorrag_result.answer

        pairs = await gather_with_concurrency(
            coroutines=[_both_answers(q) for q in questions],
            max_concurrency=max(1, self.max_concurrency // 4),
            return_exceptions=False,
        )

        return list(pairs)

    # ── Per-question evaluation ────────────────────────────────────────────────

    async def _evaluate_question(
        self,
        question_id: int,
        question: str,
        graphrag_answer: str,
        vectorrag_answer: str,
        criteria: list[EvalCriterion],
        eval_runs: int,
        community_level: str,
        randomize: bool,
    ) -> QuestionEvalResult:
        """
        Run all judge calls for a single question across all criteria and runs.
        """
        # Build all (criterion, run_index) pairs for this question
        judgment_tasks = [
            self._single_judgment(
                question=question,
                graphrag_answer=graphrag_answer,
                vectorrag_answer=vectorrag_answer,
                criterion=criterion,
                run_index=run_idx,
                randomize=randomize,
            )
            for criterion in criteria
            for run_idx in range(eval_runs)
        ]

        all_judgments = await gather_with_concurrency(
            coroutines=judgment_tasks,
            max_concurrency=self.max_concurrency,
            return_exceptions=False,
        )

        # Group judgments by criterion
        by_criterion: dict[EvalCriterion, list[SingleJudgment]] = {
            c: [] for c in criteria
        }
        for j in all_judgments:
            by_criterion[j.criterion].append(j)

        # Compute per-criterion aggregated results
        criterion_results: list[CriterionResult] = []
        for criterion in criteria:
            judgments = by_criterion[criterion]
            cr = self._aggregate_criterion(
                criterion=criterion,
                question=question,
                judgments=judgments,
            )
            criterion_results.append(cr)

        return QuestionEvalResult(
            question_id=question_id,
            question=question,
            graphrag_answer=graphrag_answer,
            vectorrag_answer=vectorrag_answer,
            community_level=community_level,
            criterion_results=criterion_results,
        )

    # ── Single judge call ──────────────────────────────────────────────────────

    async def _single_judgment(
        self,
        question: str,
        graphrag_answer: str,
        vectorrag_answer: str,
        criterion: EvalCriterion,
        run_index: int,
        randomize: bool,
    ) -> SingleJudgment:
        """
        Make a single judge LLM call for one criterion × one run.

        Randomly assigns GraphRAG/VectorRAG to A/B positions if randomize=True.
        Corrects the winner assignment after parsing to always report in terms
        of graphrag/vectorrag regardless of A/B assignment.
        """
        # Decide A/B assignment
        if randomize and random.random() < 0.5:
            answer_a, answer_b = vectorrag_answer, graphrag_answer
            answer_a_system, answer_b_system = "vectorrag", "graphrag"
        else:
            answer_a, answer_b = graphrag_answer, vectorrag_answer
            answer_a_system, answer_b_system = "graphrag", "vectorrag"

        criterion_def = _CRITERION_DEFINITIONS[criterion]

        prompt = _JUDGE_USER_PROMPT.format(
            question=question,
            answer_a=self._truncate_answer(answer_a),
            answer_b=self._truncate_answer(answer_b),
            criterion_name=criterion.value.title(),
            criterion_definition=criterion_def,
        )

        try:
            result = await self.openai_service.complete(
                user_prompt=prompt,
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.0,
            )
            judgment_data = self._parse_judgment(result.content)
        except Exception as e:
            log.warning(
                "Judge call failed",
                criterion=criterion.value,
                run_index=run_index,
                error=str(e),
            )
            # Return a tie on failure — conservative fallback
            judgment_data = {"winner": "TIE", "score_a": 50, "score_b": 50,
                             "reasoning": f"Judge call failed: {e}"}

        # Map A/B winner back to graphrag/vectorrag
        raw_winner = judgment_data.get("winner", "TIE").strip().upper()
        if raw_winner == "A":
            winner = Winner(answer_a_system)
        elif raw_winner == "B":
            winner = Winner(answer_b_system)
        else:
            winner = Winner.TIE

        # Map scores back: if a_system=vectorrag, then score_a=vectorrag_score
        score_a = int(judgment_data.get("score_a", 50))
        score_b = int(judgment_data.get("score_b", 50))
        if answer_a_system == "graphrag":
            graphrag_score, vectorrag_score = score_a, score_b
        else:
            graphrag_score, vectorrag_score = score_b, score_a

        return SingleJudgment(
            criterion=criterion,
            winner=winner,
            answer_a_system=answer_a_system,
            answer_b_system=answer_b_system,
            answer_a_score=max(0, min(100, score_a)),
            answer_b_score=max(0, min(100, score_b)),
            reasoning=str(judgment_data.get("reasoning", ""))[:500],
            run_index=run_index,
        )

    # ── Aggregation ────────────────────────────────────────────────────────────

    def _aggregate_criterion(
        self,
        criterion: EvalCriterion,
        question: str,
        judgments: list[SingleJudgment],
    ) -> CriterionResult:
        """
        Aggregate multiple judge runs for a criterion into a CriterionResult.

        Majority vote determines the winner.
        Win rate = graphrag_wins / (graphrag_wins + vectorrag_wins) [ties excluded].
        Average scores computed across all runs.
        """
        graphrag_wins  = sum(1 for j in judgments if j.winner == Winner.GRAPHRAG)
        vectorrag_wins = sum(1 for j in judgments if j.winner == Winner.VECTORRAG)
        ties           = sum(1 for j in judgments if j.winner == Winner.TIE)

        # Compute per-system average scores (A/B might be swapped — use the
        # graphrag/vectorrag mapping stored in the judgment objects)
        graphrag_scores  = []
        vectorrag_scores = []
        for j in judgments:
            if j.answer_a_system == "graphrag":
                graphrag_scores.append(j.answer_a_score)
                vectorrag_scores.append(j.answer_b_score)
            else:
                graphrag_scores.append(j.answer_b_score)
                vectorrag_scores.append(j.answer_a_score)

        avg_graphrag_score  = round(sum(graphrag_scores)  / len(graphrag_scores),  1) if graphrag_scores  else 0.0
        avg_vectorrag_score = round(sum(vectorrag_scores) / len(vectorrag_scores), 1) if vectorrag_scores else 0.0

        decisive = graphrag_wins + vectorrag_wins
        graphrag_win_rate = graphrag_wins / decisive if decisive > 0 else 0.5

        if graphrag_wins > vectorrag_wins:
            majority_winner = Winner.GRAPHRAG
        elif vectorrag_wins > graphrag_wins:
            majority_winner = Winner.VECTORRAG
        else:
            majority_winner = Winner.TIE

        return CriterionResult(
            criterion=criterion,
            question=question,
            judgments=judgments,
            graphrag_wins=graphrag_wins,
            vectorrag_wins=vectorrag_wins,
            ties=ties,
            total_runs=len(judgments),
            graphrag_win_rate=graphrag_win_rate,
            avg_graphrag_score=avg_graphrag_score,
            avg_vectorrag_score=avg_vectorrag_score,
            majority_winner=majority_winner,
        )

    def _compute_summary_stats(
        self,
        question_results: list[QuestionEvalResult],
        criteria: list[EvalCriterion],
        eval_runs: int,
    ) -> list[EvalSummaryStats]:
        """
        Compute aggregate summary statistics per criterion across all questions.

        Returns a list of EvalSummaryStats (one per criterion).
        """
        stats: list[EvalSummaryStats] = []

        for criterion in criteria:
            # Collect per-question win rates and counts
            per_q_win_rates: list[float] = []
            total_graphrag_wins  = 0
            total_vectorrag_wins = 0
            total_ties           = 0

            for q_result in question_results:
                for cr in q_result.criterion_results:
                    if cr.criterion == criterion:
                        per_q_win_rates.append(cr.graphrag_win_rate)
                        total_graphrag_wins  += cr.graphrag_wins
                        total_vectorrag_wins += cr.vectorrag_wins
                        total_ties           += cr.ties
                        break

            if not per_q_win_rates:
                continue

            avg_win_rate = sum(per_q_win_rates) / len(per_q_win_rates)
            # Standard deviation
            variance = sum((r - avg_win_rate) ** 2 for r in per_q_win_rates) / len(per_q_win_rates)
            std_win_rate = variance ** 0.5

            # Simple binomial significance test (one-sided: H0: win_rate <= 0.5)
            total_decisive = total_graphrag_wins + total_vectorrag_wins
            p_value = None
            is_significant = None
            if total_decisive > 0:
                try:
                    from scipy.stats import binom_test  # type: ignore
                    p_value = float(binom_test(
                        total_graphrag_wins, total_decisive, 0.5, alternative="greater"
                    ))
                    is_significant = p_value < 0.05
                except ImportError:
                    # scipy not available — skip significance testing
                    pass
                except Exception:
                    pass

            # Paper targets
            meets_target = None
            if criterion == EvalCriterion.COMPREHENSIVENESS:
                meets_target = avg_win_rate > 0.72
            elif criterion == EvalCriterion.DIVERSITY:
                meets_target = avg_win_rate > 0.62

            stats.append(EvalSummaryStats(
                criterion=criterion,
                total_questions=len(question_results),
                graphrag_win_rate_avg=round(avg_win_rate, 4),
                graphrag_win_rate_std=round(std_win_rate, 4),
                graphrag_total_wins=total_graphrag_wins,
                vectorrag_total_wins=total_vectorrag_wins,
                total_ties=total_ties,
                p_value=p_value,
                is_significant=is_significant,
                meets_paper_target=meets_target,
            ))

        return stats

    # ── Parsing helpers ────────────────────────────────────────────────────────

    def _parse_judgment(self, response_text: str) -> dict:
        """
        Parse the judge LLM's JSON response.

        Expected: {"winner": "A"|"B"|"TIE", "score_a": int, "score_b": int, "reasoning": str}
        Falls back to TIE on parse failure.
        """
        text = response_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match:
                try:
                    return json.loads(obj_match.group())
                except json.JSONDecodeError:
                    pass

        log.debug("Judge JSON parse failed", preview=text[:200])
        return {"winner": "TIE", "score_a": 50, "score_b": 50, "reasoning": "Parse failure"}

    def _truncate_answer(self, answer: str, max_tokens: int = 1500) -> str:
        """Truncate an answer to avoid overwhelming the judge's context window."""
        if self.tokenizer.count_tokens(answer) <= max_tokens:
            return answer
        return self.tokenizer.truncate_to_limit(answer, max_tokens)

    @staticmethod
    def _resolve_criteria(
        criteria: list[str | EvalCriterion] | None,
    ) -> list[EvalCriterion]:
        """Resolve criterion strings to EvalCriterion enums."""
        if criteria is None:
            return list(EvalCriterion)
        resolved = []
        for c in criteria:
            if isinstance(c, EvalCriterion):
                resolved.append(c)
            else:
                resolved.append(EvalCriterion(str(c).lower()))
        return resolved

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> "EvaluationEngine":
        """Build an EvaluationEngine from application settings."""
        openai_svc = None  # Built by the sub-engines; shared reference

        graphrag_engine   = GraphRAGEngine.from_settings()
        vectorrag_engine  = VectorRAGEngine.from_settings()

        # Use the same OpenAI service for judge calls
        from app.config import get_settings
        settings = get_settings()
        from app.services.openai_service import OpenAIService
        from app.services.tokenizer_service import TokenizerService
        judge_openai = OpenAIService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=0.0,
        )
        tokenizer = TokenizerService(model=settings.openai_model)

        return cls(
            openai_service=judge_openai,
            graphrag_engine=graphrag_engine,
            vectorrag_engine=vectorrag_engine,
            tokenizer=tokenizer,
            randomize_answer_order=True,
        )


def get_evaluation_engine() -> EvaluationEngine:
    """FastAPI dependency: return an EvaluationEngine built from settings."""
    return EvaluationEngine.from_settings()


__all__ = [
    "EvaluationEngine",
    "get_evaluation_engine",
]