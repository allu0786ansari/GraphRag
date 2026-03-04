"""
models/request_models.py — API request models.

Every inbound request body for every endpoint is defined here.
These models enforce strict validation before any business logic runs.

Covers:
  - IndexRequest        POST /api/v1/index
  - QueryRequest        POST /api/v1/query
  - EvalRequest         POST /api/v1/evaluate
  - QuestionGenRequest  POST /api/v1/evaluate/generate-questions
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class CommunityLevel(str, Enum):
    """
    Hierarchical community levels from the Leiden algorithm.

    C0 = root (fewest communities, broadest, 2.3% token cost vs full text)
    C1 = high-level sub-communities (default, 20.7% cost, best quality/cost)
    C2 = intermediate (57.4% cost)
    C3 = leaf-level, most granular (66.8% cost)
    """
    C0 = "c0"
    C1 = "c1"
    C2 = "c2"
    C3 = "c3"


class QueryMode(str, Enum):
    """Which RAG system to use for answering."""
    GRAPHRAG   = "graphrag"    # map-reduce over community summaries
    VECTORRAG  = "vectorrag"   # semantic search over chunk embeddings
    BOTH       = "both"        # run both and return side-by-side


class EvalCriterion(str, Enum):
    """
    The four evaluation criteria from the paper (Section 4.1).

    Comprehensiveness and Diversity expected to favour GraphRAG.
    Directness expected to favour Vector RAG (control condition).
    Empowerment is mixed.
    """
    COMPREHENSIVENESS = "comprehensiveness"
    DIVERSITY         = "diversity"
    EMPOWERMENT       = "empowerment"
    DIRECTNESS        = "directness"


# ── Index Request ─────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    """
    POST /api/v1/index

    Triggers the full offline indexing pipeline on the loaded corpus.
    Pipeline stages run sequentially with checkpointing:
      chunking → extraction → gleaning → graph → communities → summarization

    All pipeline parameters default to paper-exact values.
    Override only during experimentation.
    """

    # ── Pipeline parameters ───────────────────────────────────────────────────
    chunk_size: Annotated[int, Field(
        default=600, ge=100, le=2400,
        description="Token size per chunk. Paper uses 600 (Section 3.1.1).",
        examples=[600],
    )]

    chunk_overlap: Annotated[int, Field(
        default=100, ge=0, le=300,
        description="Overlap tokens between adjacent chunks to prevent entity loss at boundaries.",
        examples=[100],
    )]

    gleaning_rounds: Annotated[int, Field(
        default=2, ge=0, le=5,
        description=(
            "Self-reflection iterations after initial extraction. "
            "Each round re-prompts the LLM to find missed entities. "
            "Paper Figure 3 shows iteration 0→1 nearly doubles entity references. "
            "Recommend 2 for best recall/cost balance."
        ),
        examples=[2],
    )]

    context_window_size: Annotated[int, Field(
        default=8000, ge=1000, le=128000,
        description=(
            "Token limit for LLM context windows during summarization. "
            "Paper found 8k beats 16k/32k/64k for comprehensiveness (Appendix C). "
            "Do not increase without re-testing."
        ),
        examples=[8000],
    )]

    max_community_levels: Annotated[int, Field(
        default=3, ge=1, le=5,
        description="Maximum depth of community hierarchy. 3 produces C0–C3.",
        examples=[3],
    )]

    # ── Execution control ─────────────────────────────────────────────────────
    force_reindex: Annotated[bool, Field(
        default=False,
        description=(
            "If True, delete all existing artifacts and re-run from scratch. "
            "If False (default), resume from the last completed checkpoint stage."
        ),
    )]

    skip_claims: Annotated[bool, Field(
        default=False,
        description=(
            "If True, skip claim extraction during entity/relationship extraction. "
            "Claims are optional per the paper but improve evaluation quality."
        ),
    )]

    max_chunks: Annotated[int | None, Field(
        default=None, ge=1,
        description=(
            "If set, only process the first N chunks. "
            "Use for development/testing to avoid full corpus cost. "
            "Set to None (default) to process all chunks."
        ),
        examples=[None, 100],
    )]

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be strictly less than chunk_size ({chunk_size})"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_size": 600,
                "chunk_overlap": 100,
                "gleaning_rounds": 2,
                "context_window_size": 8000,
                "max_community_levels": 3,
                "force_reindex": False,
                "skip_claims": False,
                "max_chunks": None,
            }
        }
    }


# ── Query Request ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    POST /api/v1/query

    Submit a question to one or both RAG systems.

    For GraphRAG: map-reduce over community summaries at the specified level.
    For VectorRAG: embed query → FAISS top-k → fill 8k context → answer.
    """

    query: Annotated[str, Field(
        min_length=3, max_length=2000,
        description="The question to answer. Should be a global sensemaking question for GraphRAG.",
        examples=["What are the main themes across all the news articles in this dataset?"],
    )]

    mode: Annotated[QueryMode, Field(
        default=QueryMode.BOTH,
        description=(
            "Which RAG system(s) to use. "
            "'both' runs GraphRAG and VectorRAG in parallel and returns side-by-side answers."
        ),
    )]

    community_level: Annotated[CommunityLevel, Field(
        default=CommunityLevel.C1,
        description=(
            "Community hierarchy level for GraphRAG map-reduce. "
            "C1 is the default (best comprehensiveness/cost balance per paper Table 2). "
            "C0 is 43x cheaper but slightly less detailed."
        ),
    )]

    # ── VectorRAG-specific ────────────────────────────────────────────────────
    top_k: Annotated[int, Field(
        default=10, ge=1, le=100,
        description=(
            "Number of top-k chunks to retrieve for VectorRAG. "
            "Retrieved chunks fill the 8k context window. "
            "10 is a safe default; reduce if chunks are long."
        ),
        examples=[10],
    )]

    # ── GraphRAG-specific ─────────────────────────────────────────────────────
    helpfulness_threshold: Annotated[int, Field(
        default=0, ge=0, le=100,
        description=(
            "Discard map-stage partial answers at or below this helpfulness score. "
            "Paper uses 0 (discard only score=0 answers). "
            "Increase to filter low-quality partial answers more aggressively."
        ),
        examples=[0],
    )]

    # ── Response control ──────────────────────────────────────────────────────
    include_context: Annotated[bool, Field(
        default=True,
        description=(
            "If True, include the context used to generate the answer "
            "(community summaries for GraphRAG, retrieved chunks for VectorRAG)."
        ),
    )]

    include_token_usage: Annotated[bool, Field(
        default=True,
        description="If True, include total token usage for the query.",
    )]

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the main themes across all the news articles in this dataset?",
                "mode": "both",
                "community_level": "c1",
                "top_k": 10,
                "helpfulness_threshold": 0,
                "include_context": True,
                "include_token_usage": True,
            }
        }
    }


# ── Evaluation Request ────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    """
    POST /api/v1/evaluate

    Run the LLM-as-a-Judge evaluation from the paper (Section 4.1).

    For each question, both systems answer it, then a judge LLM scores
    each answer across the four paper criteria. Repeated eval_runs times
    for stochasticity control (paper uses 5 runs).
    """

    questions: Annotated[list[str], Field(
        min_length=1, max_length=125,
        description=(
            "List of global sensemaking questions to evaluate. "
            "Paper uses 125 questions (5 personas × 5 tasks × 5 questions). "
            "All questions must require whole-corpus understanding."
        ),
        examples=[["What are the main themes across all news articles?"]],
    )]

    criteria: Annotated[list[EvalCriterion], Field(
        default=[
            EvalCriterion.COMPREHENSIVENESS,
            EvalCriterion.DIVERSITY,
            EvalCriterion.EMPOWERMENT,
            EvalCriterion.DIRECTNESS,
        ],
        min_length=1,
        description="Which evaluation criteria to score. Default: all four paper criteria.",
    )]

    community_level: Annotated[CommunityLevel, Field(
        default=CommunityLevel.C1,
        description="Community level for GraphRAG answers during evaluation.",
    )]

    eval_runs: Annotated[int, Field(
        default=5, ge=1, le=10,
        description=(
            "Number of judge LLM evaluation runs per question pair. "
            "Paper uses 5 to control for LLM stochasticity. "
            "Results are majority-voted / averaged across runs."
        ),
        examples=[5],
    )]

    randomize_answer_order: Annotated[bool, Field(
        default=True,
        description=(
            "If True, randomly assign GraphRAG and VectorRAG answers to "
            "Answer A / Answer B positions to prevent position bias in the judge. "
            "Paper uses this methodology."
        ),
    )]

    @field_validator("questions")
    @classmethod
    def questions_must_be_non_empty_strings(cls, v: list[str]) -> list[str]:
        for i, q in enumerate(v):
            if not q.strip():
                raise ValueError(f"Question at index {i} is empty or whitespace only.")
            if len(q.strip()) < 10:
                raise ValueError(
                    f"Question at index {i} is too short (min 10 chars): '{q}'"
                )
        return [q.strip() for q in v]

    @field_validator("criteria")
    @classmethod
    def criteria_must_be_unique(cls, v: list[EvalCriterion]) -> list[EvalCriterion]:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate criteria are not allowed.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "questions": [
                    "What are the main themes across all the news articles in this dataset?",
                    "Which public figures are repeatedly mentioned and in what contexts?",
                ],
                "criteria": ["comprehensiveness", "diversity", "empowerment", "directness"],
                "community_level": "c1",
                "eval_runs": 5,
                "randomize_answer_order": True,
            }
        }
    }


# ── Question Generation Request ───────────────────────────────────────────────

class QuestionGenRequest(BaseModel):
    """
    POST /api/v1/evaluate/generate-questions

    Implements Algorithm 1 from the paper (Section 3.3):
    Generate K×N×M global sensemaking questions from the corpus.

    Procedure:
      1. Generate K user personas from corpus description.
      2. For each persona: generate N relevant tasks.
      3. For each (persona, task): generate M global sensemaking questions.
      Total: K × N × M questions (paper default = 5×5×5 = 125).
    """

    corpus_description: Annotated[str, Field(
        min_length=20, max_length=2000,
        description=(
            "A brief description of the corpus and its purpose. "
            "Used by the LLM to generate relevant personas and tasks. "
            "Example: 'A collection of 2023-2024 news articles covering technology, "
            "health, business, science, and entertainment topics.'"
        ),
        examples=["A collection of 2023-2024 news articles covering technology, health, business, science, and entertainment topics."],
    )]

    num_personas: Annotated[int, Field(
        default=5, ge=1, le=10,
        description="K: Number of user personas to generate. Paper uses 5.",
        examples=[5],
    )]

    num_tasks_per_persona: Annotated[int, Field(
        default=5, ge=1, le=10,
        description="N: Number of tasks per persona. Paper uses 5.",
        examples=[5],
    )]

    num_questions_per_task: Annotated[int, Field(
        default=5, ge=1, le=10,
        description="M: Number of questions per (persona, task) pair. Paper uses 5.",
        examples=[5],
    )]

    @property
    def total_questions(self) -> int:
        return self.num_personas * self.num_tasks_per_persona * self.num_questions_per_task

    model_config = {
        "json_schema_extra": {
            "example": {
                "corpus_description": "A collection of 2023-2024 news articles covering technology, health, business, science, and entertainment topics.",
                "num_personas": 5,
                "num_tasks_per_persona": 5,
                "num_questions_per_task": 5,
            }
        }
    }