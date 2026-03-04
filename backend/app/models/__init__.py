"""
models/__init__.py — Public re-exports for all model classes.

Import from here rather than from individual files to avoid
import churn when models are moved or renamed.

Usage:
    from app.models import QueryRequest, QueryResponse, NodeSchema, EvalResponse
"""

# ── Request models ─────────────────────────────────────────────────────────────
from app.models.request_models import (
    CommunityLevel as RequestCommunityLevel,
    EvalCriterion as RequestEvalCriterion,
    EvalRequest,
    IndexRequest,
    QueryMode,
    QueryRequest,
    QuestionGenRequest,
)

# ── Response models ────────────────────────────────────────────────────────────
from app.models.response_models import (
    CommunityBrief,
    CommunityContext,
    CommunityListResponse,
    ErrorDetail,
    ErrorResponse,
    GeneratedPersona,
    GraphRAGAnswer,
    GraphStatsResponse,
    IndexResponse,
    IndexStatusResponse,
    PipelineStage,
    PipelineStatus,
    QuestionGenResponse,
    QueryResponse,
    RetrievedChunk,
    StageProgress,
    TokenUsage,
    VectorRAGAnswer,
)

# ── Graph models ───────────────────────────────────────────────────────────────
from app.models.graph_models import (
    ChunkExtraction,
    ChunkSchema,
    CommunityFinding,
    CommunityLevel as GraphCommunityLevel,
    CommunityMembership,
    CommunitySchema,
    CommunitySummary,
    EdgeSchema,
    EntityType,
    ExtractedClaim as GraphExtractedClaim,
    ExtractedEntity,
    ExtractedRelationship,
    NodeSchema,
    PipelineArtifacts,
)

# ── Evaluation models ──────────────────────────────────────────────────────────
from app.models.evaluation_models import (
    ClaimComparisonResult,
    ClaimEvalResponse,
    ClaimMetrics,
    CriterionResult,
    EvalCriterion,
    EvalResponse,
    EvalSummaryStats,
    ExtractedClaim as EvalExtractedClaim,
    QuestionEvalResult,
    SingleJudgment,
    UserPersona,
    Winner,
)

__all__ = [
    # Requests
    "IndexRequest", "QueryRequest", "QueryMode", "EvalRequest",
    "QuestionGenRequest", "RequestCommunityLevel", "RequestEvalCriterion",
    # Responses
    "IndexResponse", "IndexStatusResponse", "PipelineStage", "PipelineStatus",
    "StageProgress", "TokenUsage", "VectorRAGAnswer", "GraphRAGAnswer",
    "QueryResponse", "RetrievedChunk", "CommunityContext", "GraphStatsResponse",
    "CommunityBrief", "CommunityListResponse", "QuestionGenResponse",
    "GeneratedPersona", "ErrorResponse", "ErrorDetail",
    # Graph models
    "ChunkSchema", "ChunkExtraction", "ExtractedEntity", "ExtractedRelationship",
    "GraphExtractedClaim", "NodeSchema", "EdgeSchema", "CommunitySchema",
    "CommunityFinding", "CommunitySummary", "CommunityMembership",
    "PipelineArtifacts", "EntityType", "GraphCommunityLevel",
    # Evaluation models
    "Winner", "EvalCriterion", "SingleJudgment", "CriterionResult",
    "QuestionEvalResult", "EvalSummaryStats", "EvalResponse",
    "EvalExtractedClaim", "ClaimMetrics", "ClaimComparisonResult",
    "ClaimEvalResponse", "UserPersona",
]