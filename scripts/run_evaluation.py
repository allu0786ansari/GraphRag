#!/usr/bin/env python3
"""
scripts/run_evaluation.py — Run the LLM-as-judge evaluation suite.

For each question, submits it to both GraphRAG and VectorRAG, then
uses an LLM judge to score the answers on 4 criteria:
  - Comprehensiveness   (covers all aspects?)
  - Diversity           (multiple perspectives?)
  - Empowerment         (helps reader make decisions?)
  - Directness          (concise and specific?)

Runs each comparison N times (--runs) and takes majority vote.
Saves results to data/evaluation/results.json.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --questions-file data/evaluation/questions.json
    python scripts/run_evaluation.py --runs 3 --level c1
    python scripts/run_evaluation.py --output-dir data/evaluation/
    python scripts/run_evaluation.py --dry-run     # show questions without running
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge evaluation on GraphRAG vs VectorRAG.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=_PROJECT_ROOT / "data" / "evaluation" / "questions.json",
        help="JSON file containing list of question strings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "evaluation",
        help="Directory to write results.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        metavar="N",
        help="Number of judge runs per question pair (majority vote, default: 5)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="c1",
        choices=["c0", "c1", "c2", "c3"],
        help="Community level for GraphRAG queries (default: c1)",
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=["comprehensiveness", "diversity", "empowerment", "directness"],
        choices=["comprehensiveness", "diversity", "empowerment", "directness"],
        help="Evaluation criteria to use",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only first N questions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show questions without running evaluation",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    """Load questions from a JSON file. Supports list[str] or list[{question: str}]."""
    if not path.exists():
        print(f"\n  ERROR: Questions file not found: {path}")
        print("  Create it or use the default: data/evaluation/questions.json\n")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        if data and isinstance(data[0], str):
            return data
        if data and isinstance(data[0], dict):
            return [item.get("question", item.get("text", "")) for item in data]

    print(f"  ERROR: questions.json must be a list of strings or [{{'question': '...'}}]")
    sys.exit(1)


async def run(args: argparse.Namespace) -> int:
    os.environ["ARTIFACTS_DIR"] = str(args.artifacts_dir)
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "text"

    from app.config import get_settings
    from app.core.query.evaluation_engine import EvaluationEngine
    from app.core.query.graphrag_engine import GraphRAGEngine
    from app.core.query.vectorrag_engine import VectorRAGEngine
    from app.models.request_models import EvalCriterion, CommunityLevel

    settings = get_settings()

    # ── Load questions ────────────────────────────────────────────────────────
    questions = load_questions(args.questions_file)
    if args.max_questions:
        questions = questions[: args.max_questions]

    if args.dry_run:
        print(f"\n  Evaluation — Dry Run")
        print(f"  Questions file: {args.questions_file}")
        print(f"  {len(questions)} questions:\n")
        for i, q in enumerate(questions, 1):
            print(f"  {i:2}. {q}")
        print(f"\n  Criteria:  {', '.join(args.criteria)}")
        print(f"  Runs/Q:    {args.runs}")
        print(f"  Level:     {args.level}")
        est_cost = len(questions) * args.runs * 0.05
        print(f"  Est. cost: ~${est_cost:.2f} (rough)\n")
        return 0

    print(f"\n  GraphRAG vs VectorRAG — Evaluation")
    print(f"  {'─' * 48}")
    print(f"  Questions:  {len(questions)}")
    print(f"  Criteria:   {', '.join(args.criteria)}")
    print(f"  Runs/Q:     {args.runs}")
    print(f"  Level:      {args.level}")
    print(f"  Output:     {args.output_dir}/results.json")
    print(f"  {'─' * 48}\n")

    # ── Initialize engines ────────────────────────────────────────────────────
    graphrag_engine = GraphRAGEngine.from_settings()
    vectorrag_engine = VectorRAGEngine.from_settings()
    eval_engine = EvaluationEngine.from_settings()

    criteria = [EvalCriterion(c) for c in args.criteria]
    community_level = CommunityLevel(args.level)

    results = []
    t0 = time.time()

    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question[:70]}")

        try:
            # Query both systems
            print(f"         → GraphRAG query...")
            graphrag_answer = await graphrag_engine.query(
                query=question,
                community_level=community_level,
            )

            print(f"         → VectorRAG query...")
            vectorrag_answer = await vectorrag_engine.query(query=question)

            # Run judge N times
            print(f"         → Judging ({args.runs} runs)...")
            judgments = []
            for run_n in range(args.runs):
                judgment = await eval_engine.evaluate_pair(
                    question=question,
                    graphrag_answer=graphrag_answer.answer,
                    vectorrag_answer=vectorrag_answer.answer,
                    criteria=criteria,
                )
                judgments.append(judgment)

            # Aggregate
            result = {
                "question": question,
                "graphrag_answer": graphrag_answer.answer,
                "vectorrag_answer": vectorrag_answer.answer,
                "graphrag_tokens": graphrag_answer.total_tokens,
                "vectorrag_tokens": vectorrag_answer.total_tokens,
                "judgments": judgments,
                "community_level": args.level,
            }
            results.append(result)

            # Show quick summary
            wins = {"graphrag": 0, "vectorrag": 0, "tie": 0}
            for j in judgments:
                for criterion_result in j.get("criteria", {}).values():
                    w = criterion_result.get("winner", "tie")
                    wins[w] = wins.get(w, 0) + 1
            print(f"         ✓ GraphRAG: {wins['graphrag']} | VectorRAG: {wins['vectorrag']} | Tie: {wins['tie']}")

        except Exception as exc:
            print(f"         ✗ Failed: {exc}")
            results.append({"question": question, "error": str(exc)})

    # ── Save results ──────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"

    output = {
        "metadata": {
            "questions": len(questions),
            "criteria": args.criteria,
            "runs_per_question": args.runs,
            "community_level": args.level,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    successful = [r for r in results if "error" not in r]

    print(f"\n  {'─' * 48}")
    print(f"  ✓  Evaluation complete in {elapsed:.1f}s")
    print(f"  Successful: {len(successful)}/{len(questions)}")
    print(f"  Results:    {output_path}")

    if successful:
        # Overall win rate
        total_wins: dict[str, int] = {"graphrag": 0, "vectorrag": 0, "tie": 0}
        for r in successful:
            for j in r.get("judgments", []):
                for cv in j.get("criteria", {}).values():
                    w = cv.get("winner", "tie")
                    total_wins[w] = total_wins.get(w, 0) + 1
        total = sum(total_wins.values()) or 1
        print(f"\n  Overall win rates:")
        print(f"    GraphRAG:  {total_wins['graphrag']/total*100:.1f}%")
        print(f"    VectorRAG: {total_wins['vectorrag']/total*100:.1f}%")
        print(f"    Tie:       {total_wins['tie']/total*100:.1f}%")

    print()
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()