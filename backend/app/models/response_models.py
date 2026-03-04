"""
models/response_models.py — API response models.

Every outbound response shape for every endpoint is defined here.
These models control exactly what the frontend receives.

Covers:
  - IndexResponse / IndexStatusResponse   GET /api/v1/index/status
  - VectorRAGAnswer                        internal answer model
  - GraphRAGAnswer                         internal answer model
  - QueryResponse                          POST /api/v1/query
  - GraphStatsResponse                     GET /api/v1/graph
  - CommunityListResponse                  GET /api/v1/communities/{level}
  - QuestionGenResponse                    POST /api/v1/evaluate/generate-questions
  - ErrorResponse                          all error shapes
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class PipelineStage(str, Enum):
    """Stages of the indexing pipeline in execution order."""
    PENDING            = "pending"
    CHUNKING           = "chunking"
    EXTRACTION         = "extraction"
    GLEANING           = "gleaning"
    GRAPH_CONSTRUCTION = "graph_construction"
    COMMUNITY_DETECTION= "community_detection"
    SUMMARIZATION      = "summarization"
    COMPLETED          = "completed"
    FAILED             = "failed"


class PipelineStatus(str, Enum):
    """Top-level status of a pipeline job."""
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


# ── Token usage ───────────────────────────────────────────────────────────────

class TokenUsage(BaseModel):
    """Token consumption breakdown for a single LLM operation or full query."""

    prompt_tokens: int = Field(
        description="Tokens consumed in the prompt/input.",
        examples=[1500],
    )
    completion_tokens: int = Field(
        description="Tokens generated in the completion/output.",
        examples=[400],
    )
    total_tokens: int = Field(
        description="Total tokens consumed (prompt + completion).",
        examples=[1900],
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        description="Estimated cost in USD at GPT-4o pricing. None if not computed.",
        examples=[0.019],
    )

    @classmethod
    def zero(cls) -> TokenUsage:
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=(
                (self.estimated_cost_usd or 0.0) + (other.estimated_cost_usd or 0.0)
                if self.estimated_cost_usd is not None or other.estimated_cost_usd is not None
                else None
            ),
        )


# ── Stage progress ────────────────────────────────────────────────────────────

class StageProgress(BaseModel):
    """Progress report for a single pipeline stage."""

    stage: PipelineStage = Field(description="Which pipeline stage this represents.")
    status: PipelineStatus = Field(description="Current status of this stage.")
    started_at: datetime | None = Field(default=None, description="When this stage started.")
    completed_at: datetime | None = Field(default=None, description="When this stage completed.")
    duration_seconds: float | None = Field(default=None, description="Wall-clock seconds taken.")
    items_processed: int | None = Field(
        default=None,
        description="Items processed so far (e.g. chunks extracted, nodes built).",
    )
    items_total: int | None = Field(
        default=None,
        description="Total items to process in this stage.",
    )
    progress_pct: float | None = Field(
        default=None, ge=0.0, le=100.0,
        description="Completion percentage 0–100.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if this stage failed.",
    )


# ── Index responses ───────────────────────────────────────────────────────────

class IndexResponse(BaseModel):
    """
    POST /api/v1/index

    Immediate response when indexing is accepted. The actual indexing
    runs in a background worker — poll /api/v1/index/status for progress.
    """

    job_id: str = Field(
        description="Unique job identifier. Use with GET /api/v1/index/status?job_id=...",
        examples=["idx_a1b2c3d4"],
    )
    status: PipelineStatus = Field(
        default=PipelineStatus.QUEUED,
        description="Initial status — always 'queued' on acceptance.",
    )
    message: str = Field(
        description="Human-readable status message.",
        examples=["Indexing job accepted and queued. Poll /api/v1/index/status for progress."],
    )
    accepted_at: datetime = Field(
        description="UTC timestamp when the job was accepted.",
    )
    estimated_duration_minutes: int | None = Field(
        default=None,
        description=(
            "Rough estimate of total indexing time in minutes. "
            "Paper reports ~281 minutes for a 1M token corpus on Xeon VM at 2M TPM."
        ),
        examples=[60],
    )


class IndexStatusResponse(BaseModel):
    """
    GET /api/v1/index/status

    Real-time progress of a running or completed indexing job.
    """

    job_id: str = Field(description="Job identifier.")
    status: PipelineStatus = Field(description="Overall job status.")
    current_stage: PipelineStage = Field(description="Currently executing pipeline stage.")
    stages: list[StageProgress] = Field(
        description="Per-stage progress for all six pipeline stages.",
    )
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    elapsed_seconds: float | None = Field(default=None)
    total_chunks: int | None = Field(
        default=None,
        description="Total number of 600-token chunks in the corpus.",
    )
    total_nodes: int | None = Field(
        default=None,
        description="Nodes (entities) in the knowledge graph. Expected ~15k for news corpus.",
    )
    total_edges: int | None = Field(
        default=None,
        description="Edges (relationships) in the knowledge graph. Expected ~19k for news corpus.",
    )
    total_communities: dict[str, int] | None = Field(
        default=None,
        description="Community count per level: {'c0': 55, 'c1': 555, 'c2': 1797, 'c3': 2142}",
    )
    total_summaries: int | None = Field(
        default=None,
        description="Total community summaries generated across all levels.",
    )
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Cumulative LLM token usage for the entire indexing pipeline.",
    )
    error_message: str | None = Field(
        default=None,
        description="Top-level error message if the job failed.",
    )


# ── Context items (what was used to generate the answer) ─────────────────────

class RetrievedChunk(BaseModel):
    """A single text chunk retrieved by VectorRAG."""

    chunk_id: str = Field(description="Unique chunk identifier.")
    source_document: str = Field(description="Source document filename or identifier.")
    text: str = Field(description="The chunk text content.")
    similarity_score: float = Field(
        ge=0.0, le=1.0,
        description="Cosine similarity score between chunk embedding and query embedding.",
    )
    token_count: int = Field(description="Number of tokens in this chunk.")


class CommunityContext(BaseModel):
    """A single community summary used in the GraphRAG map stage."""

    community_id: str = Field(description="Unique community identifier.")
    level: str = Field(description="Community level: c0, c1, c2, or c3.")
    title: str = Field(description="Community title from the summary.")
    summary: str = Field(description="Full community summary text.")
    helpfulness_score: int = Field(
        ge=0, le=100,
        description=(
            "Score assigned by the map-stage LLM for how useful this community "
            "summary was for answering the query. 0 = irrelevant (discarded)."
        ),
    )
    partial_answer: str | None = Field(
        default=None,
        description="The partial answer generated for this community in the map stage.",
    )
    token_count: int = Field(description="Token count of the community summary.")


# ── Individual RAG answers ────────────────────────────────────────────────────

class VectorRAGAnswer(BaseModel):
    """
    Answer produced by the Vector RAG (Semantic Search) system.

    The baseline described as 'SS' in the paper (Section 3.2).
    Context window filled with top-k similar chunks.
    """

    answer: str = Field(
        description="The generated answer from VectorRAG.",
        examples=["Based on the retrieved articles, the main topics include..."],
    )
    query: str = Field(description="The original query that was answered.")
    chunks_retrieved: int = Field(
        description="Number of chunks retrieved from the vector store.",
    )
    context_tokens_used: int = Field(
        description="Tokens in the context window used to generate this answer.",
    )
    context: list[RetrievedChunk] | None = Field(
        default=None,
        description="The retrieved chunks used as context. Null if include_context=False.",
    )
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Token usage for this answer. Null if include_token_usage=False.",
    )
    latency_ms: float = Field(
        description="Wall-clock milliseconds taken to produce this answer.",
    )


class GraphRAGAnswer(BaseModel):
    """
    Answer produced by the GraphRAG system.

    Implements the map-reduce pipeline from the paper (Section 3.2):
    community summaries → parallel map → score + filter → reduce → answer.
    """

    answer: str = Field(
        description="The final global answer from the GraphRAG reduce stage.",
        examples=["The dataset reveals several interconnected themes across technology, health..."],
    )
    query: str = Field(description="The original query that was answered.")
    community_level: str = Field(
        description="Community level used for this query (c0/c1/c2/c3).",
        examples=["c1"],
    )

    # ── Map stage statistics ───────────────────────────────────────────────────
    communities_total: int = Field(
        description="Total community summaries available at the selected level.",
    )
    communities_used_in_map: int = Field(
        description="Communities included in the map-stage context windows.",
    )
    map_answers_generated: int = Field(
        description="Number of partial answers generated in the map stage.",
    )
    map_answers_after_filter: int = Field(
        description=(
            "Partial answers remaining after filtering out helpfulness_score=0 answers. "
            "These are ranked and fed into the reduce stage."
        ),
    )
    context_tokens_used: int = Field(
        description="Total tokens consumed across all map-stage LLM calls.",
    )

    # ── Context ───────────────────────────────────────────────────────────────
    context: list[CommunityContext] | None = Field(
        default=None,
        description=(
            "Community summaries used in the map stage with their helpfulness scores. "
            "Null if include_context=False."
        ),
    )
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Token usage breakdown for map + reduce stages combined.",
    )
    latency_ms: float = Field(
        description="Wall-clock milliseconds taken for the full map-reduce pipeline.",
    )


# ── Query response ────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """
    POST /api/v1/query

    Top-level response. Contains one or both RAG system answers
    depending on the requested mode (graphrag / vectorrag / both).
    """

    query: str = Field(description="The original query.")
    mode: str = Field(
        description="Which mode was used: graphrag, vectorrag, or both.",
        examples=["both"],
    )
    request_id: str = Field(description="Request identifier for tracing.")

    graphrag: GraphRAGAnswer | None = Field(
        default=None,
        description="GraphRAG answer. Present when mode is 'graphrag' or 'both'.",
    )
    vectorrag: VectorRAGAnswer | None = Field(
        default=None,
        description="VectorRAG answer. Present when mode is 'vectorrag' or 'both'.",
    )

    total_latency_ms: float = Field(
        description="Total wall-clock milliseconds for the entire request.",
    )
    total_token_usage: TokenUsage | None = Field(
        default=None,
        description="Combined token usage across all systems for this query.",
    )


# ── Graph / community responses ───────────────────────────────────────────────

class GraphStatsResponse(BaseModel):
    """
    GET /api/v1/graph

    High-level statistics about the constructed knowledge graph.
    Used by the frontend to display graph metrics and status.
    """

    is_indexed: bool = Field(
        description="Whether the corpus has been fully indexed.",
    )
    total_nodes: int = Field(
        description="Total entity nodes in the knowledge graph.",
        examples=[15000],
    )
    total_edges: int = Field(
        description="Total relationship edges in the knowledge graph.",
        examples=[19000],
    )
    communities_by_level: dict[str, int] = Field(
        description=(
            "Number of communities at each level. "
            "e.g. {'c0': 55, 'c1': 555, 'c2': 1797, 'c3': 2142}"
        ),
    )
    total_summaries: int = Field(
        description="Total community summaries generated across all levels.",
    )
    top_entities_by_degree: list[dict[str, Any]] = Field(
        default=[],
        description=(
            "Top 20 most connected entities by degree. "
            "Each entry: {'name': str, 'type': str, 'degree': int}"
        ),
    )
    entity_type_distribution: dict[str, int] = Field(
        default={},
        description=(
            "Count of entities per type. "
            "e.g. {'PERSON': 3200, 'ORGANIZATION': 2800, 'LOCATION': 1500, ...}"
        ),
    )
    avg_node_degree: float = Field(
        default=0.0,
        description="Average number of edges per node in the graph.",
    )
    indexed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when indexing was last completed.",
    )


class CommunityBrief(BaseModel):
    """Brief summary of a single community for list endpoints."""

    community_id: str
    level: str
    title: str
    summary_preview: str = Field(
        description="First 200 characters of the community summary.",
    )
    node_count: int = Field(description="Number of nodes in this community.")
    impact_rating: float = Field(
        ge=0.0, le=10.0,
        description="Impact/importance rating 0–10 from the LLM summarization.",
    )


class CommunityListResponse(BaseModel):
    """
    GET /api/v1/communities/{level}

    Paginated list of communities at a given hierarchy level.
    """

    level: str = Field(description="The queried community level.")
    total: int = Field(description="Total number of communities at this level.")
    page: int = Field(description="Current page number.")
    page_size: int = Field(description="Items per page.")
    communities: list[CommunityBrief] = Field(
        description="Communities on the current page.",
    )


# ── Question generation response ──────────────────────────────────────────────

class GeneratedPersona(BaseModel):
    """A single generated user persona with its tasks and questions."""

    persona_id: int = Field(description="Index of this persona (1–K).")
    description: str = Field(description="Persona description generated by the LLM.")
    tasks: list[str] = Field(description="Tasks this persona would perform with the RAG system.")
    questions: list[str] = Field(
        description=(
            "Global sensemaking questions generated for this persona. "
            "All questions require whole-corpus understanding."
        ),
    )


class QuestionGenResponse(BaseModel):
    """
    POST /api/v1/evaluate/generate-questions

    Result of Algorithm 1 question generation.
    """

    corpus_description: str = Field(description="The corpus description used for generation.")
    total_questions: int = Field(
        description="Total questions generated (K × N × M).",
        examples=[125],
    )
    personas: list[GeneratedPersona] = Field(
        description="All generated personas with their tasks and questions.",
    )
    all_questions: list[str] = Field(
        description="Flat list of all generated questions for easy use in EvalRequest.",
    )
    token_usage: TokenUsage | None = Field(default=None)


# ── Error responses ───────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Structured error detail for a single validation or business logic error."""

    field: str | None = Field(
        default=None,
        description="The field that caused the error, if applicable.",
        examples=["query"],
    )
    message: str = Field(description="Human-readable error description.")
    code: str = Field(
        description="Machine-readable error code.",
        examples=["field_too_short", "not_indexed", "rate_limit_exceeded"],
    )


class ErrorResponse(BaseModel):
    """
    Standard error response shape returned by all error handlers.

    All HTTP 4xx and 5xx responses use this shape for consistency.
    The frontend can always rely on this structure.
    """

    error: str = Field(
        description="Short machine-readable error type.",
        examples=["validation_error", "not_indexed", "not_found"],
    )
    message: str = Field(
        description="Human-readable explanation of the error.",
        examples=["The corpus has not been indexed yet. Run POST /api/v1/index first."],
    )
    request_id: str | None = Field(
        default=None,
        description="Request ID for tracing this error in logs.",
    )
    details: list[ErrorDetail] | None = Field(
        default=None,
        description="Additional structured error details (e.g. validation failures).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "not_indexed",
                "message": "The corpus has not been indexed yet. Run POST /api/v1/index first.",
                "request_id": "req_abc123",
                "details": None,
            }
        }
    }