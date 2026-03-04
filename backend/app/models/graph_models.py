"""
models/graph_models.py — Knowledge graph data models.

These models represent the in-memory and on-disk structure of the GraphRAG
knowledge graph at every stage of the pipeline.

Pipeline artifact shapes:
  chunks.json           → list[ChunkSchema]
  extractions.json      → list[ChunkExtraction]
  graph.pkl             → serialized NetworkX graph (nodes/edges described here)
  community_map.json    → CommunityMap
  community_summaries.json → list[CommunitySummary]

All models are used by:
  - core/pipeline/*.py  (to write artifacts)
  - core/query/*.py     (to read artifacts)
  - storage/*.py        (to serialize/deserialize artifacts)
  - api/routes_graph.py (to expose graph data to the frontend)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    """
    Common entity types extracted by the LLM.
    The extraction prompt uses 'named entity' broadly — these are the
    most common types found in news corpora.

    Note: The paper does not restrict entity types — the LLM decides.
    This enum is used for display/filtering only.
    """
    PERSON        = "PERSON"
    ORGANIZATION  = "ORGANIZATION"
    LOCATION      = "LOCATION"
    EVENT         = "EVENT"
    PRODUCT       = "PRODUCT"
    CONCEPT       = "CONCEPT"
    TECHNOLOGY    = "TECHNOLOGY"
    GEO           = "GEO"
    OTHER         = "OTHER"
    UNKNOWN       = "UNKNOWN"


class CommunityLevel(str, Enum):
    """Hierarchy levels from Leiden community detection."""
    C0 = "c0"   # root: fewest, broadest
    C1 = "c1"   # high-level sub-communities (default)
    C2 = "c2"   # intermediate
    C3 = "c3"   # leaf: most granular


# ── Stage 1 artifact: chunks.json ─────────────────────────────────────────────

class ChunkSchema(BaseModel):
    """
    A single 600-token text chunk produced by the chunking stage.
    Stored in: data/processed/chunks.json as list[ChunkSchema]
    """

    chunk_id: str = Field(
        description="Unique identifier: '{source_doc_stem}_{chunk_index:04d}'",
        examples=["news_article_001_0000"],
    )
    source_document: str = Field(
        description="Source filename (without path).",
        examples=["news_article_001.json"],
    )
    text: str = Field(
        description="Raw text content of this chunk.",
    )
    token_count: int = Field(
        ge=1,
        description="Number of tokens in this chunk (tiktoken count).",
        examples=[600],
    )
    start_char: int = Field(
        ge=0,
        description="Character offset in the source document where this chunk starts.",
    )
    end_char: int = Field(
        ge=0,
        description="Character offset in the source document where this chunk ends.",
    )
    chunk_index: int = Field(
        ge=0,
        description="Zero-based index of this chunk within its source document.",
    )
    total_chunks_in_doc: int = Field(
        ge=1,
        description="Total number of chunks in the source document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional metadata from the source document "
            "(e.g. date, category, url for news articles)."
        ),
    )


# ── Stage 2 artifact: extractions.json ────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """
    A single entity extracted from a chunk by the LLM.

    Format from paper Appendix E.1:
    ("entity"{TUPLE_DELIM}<NAME>{TUPLE_DELIM}<TYPE>{TUPLE_DELIM}<DESCRIPTION>)
    """

    name: str = Field(
        description="Entity name as extracted. Will be normalized for deduplication.",
        examples=["OpenAI", "Sam Altman"],
    )
    entity_type: str = Field(
        description="Entity type as classified by the LLM (free-form string).",
        examples=["ORGANIZATION", "PERSON"],
    )
    description: str = Field(
        description="LLM-generated description of this entity instance.",
        examples=["OpenAI is an AI research company founded in 2015."],
    )
    source_chunk_id: str = Field(
        description="The chunk this entity was extracted from.",
    )
    extraction_round: int = Field(
        default=0, ge=0,
        description=(
            "Which gleaning round produced this entity. "
            "0 = primary extraction, 1+ = gleaning rounds."
        ),
    )


class ExtractedRelationship(BaseModel):
    """
    A single relationship between two entities extracted from a chunk.

    Format from paper Appendix E.1:
    ("relationship"{TUPLE_DELIM}<SOURCE>{TUPLE_DELIM}<TARGET>{TUPLE_DELIM}<DESCRIPTION>{TUPLE_DELIM}<STRENGTH>)
    """

    source_entity: str = Field(
        description="Name of the source entity.",
        examples=["Microsoft"],
    )
    target_entity: str = Field(
        description="Name of the target entity.",
        examples=["OpenAI"],
    )
    description: str = Field(
        description="LLM-generated description of this relationship.",
        examples=["Microsoft invested $10 billion in OpenAI in 2023."],
    )
    strength: int = Field(
        ge=1, le=10,
        description=(
            "Relationship strength score 1–10 as assigned by the LLM. "
            "Used as edge weight in the knowledge graph."
        ),
        examples=[9],
    )
    source_chunk_id: str = Field(
        description="The chunk this relationship was extracted from.",
    )
    extraction_round: int = Field(
        default=0, ge=0,
        description="Which gleaning round produced this relationship.",
    )


class ExtractedClaim(BaseModel):
    """
    A verifiable factual claim about an entity, optionally extracted alongside
    entity/relationship extraction (paper Section 3.1.2).

    Claims are optional but recommended — they improve evaluation quality
    by providing atomic facts that can be validated independently.
    """

    subject_entity: str = Field(
        description="The entity this claim is about.",
        examples=["NeoChip"],
    )
    claim_type: str = Field(
        description="Type of claim (e.g. 'acquisition', 'founding', 'investment').",
        examples=["IPO"],
    )
    claim_description: str = Field(
        description="Full claim text.",
        examples=["NeoChip shares surged during their first week of trading after IPO."],
    )
    status: str = Field(
        default="TRUE",
        description="Claim status: TRUE, FALSE, SUSPECTED, INFERRED, NOT EVALUATED.",
        examples=["TRUE"],
    )
    start_date: str | None = Field(
        default=None,
        description="Claim start date in ISO format if applicable.",
        examples=["2023-01-15"],
    )
    end_date: str | None = Field(
        default=None,
        description="Claim end date in ISO format if applicable.",
    )
    source_chunk_id: str = Field(
        description="The chunk this claim was extracted from.",
    )


class ChunkExtraction(BaseModel):
    """
    All entities, relationships, and claims extracted from a single chunk.
    Stored as one element in: data/processed/extractions.json

    This is the atomic output of the extraction + gleaning pipeline for one chunk.
    """

    chunk_id: str = Field(description="The chunk that was processed.")
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="All entities extracted from this chunk (across all gleaning rounds).",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="All relationships extracted from this chunk (across all gleaning rounds).",
    )
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="Claims extracted from this chunk (empty if skip_claims=True).",
    )
    gleaning_rounds_completed: int = Field(
        default=0, ge=0,
        description="Number of gleaning rounds that ran for this chunk.",
    )
    extraction_completed: bool = Field(
        default=True,
        description="False if extraction failed or was skipped for this chunk.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if extraction_completed=False.",
    )


# ── Stage 3 artifact: graph.pkl (node/edge schemas) ───────────────────────────

class NodeSchema(BaseModel):
    """
    A single entity node in the knowledge graph.

    Multiple extracted instances of the same entity (across chunks) are
    merged into one node. Descriptions are aggregated and summarized.
    Node degree = number of relationships this entity participates in.

    Serialized as node attributes in the NetworkX graph (graph.pkl).
    """

    node_id: str = Field(
        description=(
            "Normalized entity name used as the graph node identifier. "
            "Normalization: lowercase, strip punctuation."
        ),
        examples=["openai", "sam_altman"],
    )
    name: str = Field(
        description="Display name (original casing from most common extraction).",
        examples=["OpenAI", "Sam Altman"],
    )
    entity_type: str = Field(
        description="Most common entity type assigned across all extractions.",
        examples=["ORGANIZATION"],
    )
    description: str = Field(
        description=(
            "LLM-summarized description aggregating all extracted descriptions "
            "for this entity across all chunks."
        ),
        examples=["OpenAI is an AI research company founded in 2015, known for GPT-4 and ChatGPT."],
    )
    degree: int = Field(
        default=0, ge=0,
        description="Number of edges (relationships) this node participates in.",
        examples=[42],
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="All chunk IDs where this entity was mentioned.",
    )
    claims: list[str] = Field(
        default_factory=list,
        description="All claims linked to this entity.",
    )
    mention_count: int = Field(
        default=1, ge=1,
        description="Number of times this entity was extracted across all chunks.",
    )
    community_ids: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Community membership at each level. "
            "e.g. {'c0': 'comm_0_003', 'c1': 'comm_1_045', ...}"
        ),
    )


class EdgeSchema(BaseModel):
    """
    A single relationship edge between two entity nodes.

    Multiple extractions of the same relationship are merged into one edge.
    Edge weight = number of times this relationship appeared across chunks.

    Serialized as edge attributes in the NetworkX graph (graph.pkl).
    """

    edge_id: str = Field(
        description="Unique edge identifier: '{source_node_id}__{target_node_id}'",
        examples=["microsoft__openai"],
    )
    source_node_id: str = Field(
        description="Normalized name of the source entity node.",
        examples=["microsoft"],
    )
    target_node_id: str = Field(
        description="Normalized name of the target entity node.",
        examples=["openai"],
    )
    description: str = Field(
        description=(
            "Aggregated description of this relationship across all extractions. "
            "LLM-summarized if there are multiple conflicting descriptions."
        ),
        examples=["Microsoft invested $10 billion in OpenAI in 2023, becoming its primary partner."],
    )
    weight: float = Field(
        default=1.0, ge=0.0,
        description=(
            "Edge weight = number of times this relationship was independently "
            "extracted across different chunks. Higher = stronger/more frequent relationship."
        ),
        examples=[7.0],
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="All chunks where this relationship was observed.",
    )
    combined_degree: int = Field(
        default=0, ge=0,
        description=(
            "Source node degree + target node degree. "
            "Used to prioritize edges when building community context "
            "(paper Section 3.1.4: edges sorted descending by combined_degree)."
        ),
    )


# ── Stage 4 artifact: community_map.json ──────────────────────────────────────

class CommunityMembership(BaseModel):
    """
    Describes which community a single node belongs to at each hierarchy level.
    """

    node_id: str = Field(description="The entity node.")
    memberships: dict[str, str] = Field(
        description=(
            "Level → community_id mapping for every level this node appears in. "
            "e.g. {'c0': 'comm_c0_003', 'c1': 'comm_c1_045', 'c2': 'comm_c2_234'}"
        ),
    )


class CommunitySchema(BaseModel):
    """
    A single community in the Leiden hierarchy.

    Stored in: data/processed/community_map.json as list[CommunitySchema]

    The paper uses hierarchical Leiden (graspologic) to detect communities
    at four levels (C0–C3). Each level is mutually exclusive and collectively
    exhaustive — every node belongs to exactly one community at each level.
    """

    community_id: str = Field(
        description="Unique community ID: 'comm_{level}_{index:04d}'",
        examples=["comm_c1_0045"],
    )
    level: CommunityLevel = Field(
        description="Hierarchy level of this community.",
    )
    level_index: int = Field(
        ge=0,
        description="Numeric index of this community within its level.",
    )
    node_ids: list[str] = Field(
        description="Normalized node IDs of all entities in this community.",
        min_length=1,
    )
    edge_ids: list[str] = Field(
        default_factory=list,
        description="Edge IDs for all relationships within this community.",
    )
    parent_community_id: str | None = Field(
        default=None,
        description=(
            "ID of the parent community at the next higher level (C0 has no parent). "
            "Used for the sub-community substitution strategy in higher-level summarization."
        ),
    )
    child_community_ids: list[str] = Field(
        default_factory=list,
        description="IDs of sub-communities at the next lower level.",
    )
    node_count: int = Field(
        default=0, ge=0,
        description="Number of nodes in this community.",
    )
    edge_count: int = Field(
        default=0, ge=0,
        description="Number of edges within this community.",
    )


# ── Stage 5 artifact: community_summaries.json ────────────────────────────────

class CommunityFinding(BaseModel):
    """
    A single key insight / finding in a community summary.

    The paper's structured JSON template (Section 3.1.4) specifies
    5–10 findings per community summary.
    """

    finding_id: int = Field(
        ge=0,
        description="Zero-based index of this finding within the community summary.",
    )
    summary: str = Field(
        description="One-sentence summary of the key insight.",
        examples=["Microsoft's $10B investment made it OpenAI's primary commercial partner."],
    )
    explanation: str = Field(
        description=(
            "Detailed paragraph explaining this finding with supporting evidence "
            "from the graph data (entity descriptions, relationship details, claims)."
        ),
    )


class CommunitySummary(BaseModel):
    """
    A structured report-style summary for a single community.

    Generated by the LLM using the paper's JSON template (Section 3.1.4).
    Stored in: data/processed/community_summaries.json as list[CommunitySummary]

    Two generation strategies (paper Section 3.1.4):
    - Leaf communities: fill context with edges sorted by combined_degree
    - Higher-level communities: first try element summaries directly;
      if too large, substitute sub-community summaries (shorter) to fit 8k window.
    """

    community_id: str = Field(
        description="The community this summary was generated for.",
        examples=["comm_c1_0045"],
    )
    level: CommunityLevel = Field(description="Community hierarchy level.")

    # ── LLM-generated content (paper JSON template) ───────────────────────────
    title: str = Field(
        description=(
            "Community title including representative entity names. "
            "Example: 'Microsoft-OpenAI Partnership and AI Investment Ecosystem'"
        ),
        examples=["Microsoft-OpenAI Partnership and AI Investment Ecosystem"],
    )
    summary: str = Field(
        description=(
            "Executive summary of the community's structure, key entities, "
            "and most important relationships."
        ),
    )
    impact_rating: float = Field(
        ge=0.0, le=10.0,
        description=(
            "LLM-assigned impact/importance rating 0–10. "
            "Reflects how significant this community is to the overall corpus themes."
        ),
        examples=[7.5],
    )
    rating_explanation: str = Field(
        description="One sentence explaining the impact_rating.",
        examples=["This community is central to understanding AI industry investment dynamics."],
    )
    findings: list[CommunityFinding] = Field(
        min_length=1,
        description="Key insights, 5–10 per community per paper specification.",
    )

    # ── Generation metadata ───────────────────────────────────────────────────
    node_ids: list[str] = Field(
        description="Entity nodes included in this community's context.",
    )
    context_tokens_used: int = Field(
        description="Tokens in the context window used to generate this summary.",
    )
    was_truncated: bool = Field(
        default=False,
        description=(
            "True if the community context was truncated to fit the 8k window. "
            "For higher-level communities, this triggers sub-community substitution."
        ),
    )
    used_sub_community_substitution: bool = Field(
        default=False,
        description=(
            "True if sub-community summaries were substituted for raw element summaries "
            "to fit within the 8k context window. Only applies to non-leaf communities."
        ),
    )
    generated_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when this summary was generated.",
    )
    token_usage: dict[str, int] | None = Field(
        default=None,
        description="LLM token usage for generating this summary: {'prompt': int, 'completion': int}",
    )

    @field_validator("findings")
    @classmethod
    def validate_findings_count(cls, v: list[CommunityFinding]) -> list[CommunityFinding]:
        if len(v) < 1:
            raise ValueError("Community summary must have at least 1 finding.")
        if len(v) > 20:
            raise ValueError(
                f"Community summary has {len(v)} findings, expected 5–10 per paper spec."
            )
        return v


# ── Pipeline artifact registry ─────────────────────────────────────────────────

class PipelineArtifacts(BaseModel):
    """
    Tracks the existence and metadata of all pipeline artifacts.
    Used by the readiness check and pipeline runner to determine
    which stages have been completed.
    """

    chunks_exists: bool = False
    chunks_count: int = 0
    chunks_size_mb: float = 0.0

    extractions_exists: bool = False
    extractions_count: int = 0
    extractions_size_mb: float = 0.0

    graph_exists: bool = False
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_size_mb: float = 0.0

    community_map_exists: bool = False
    community_map_size_mb: float = 0.0
    communities_by_level: dict[str, int] = Field(default_factory=dict)

    community_summaries_exists: bool = False
    community_summaries_count: int = 0
    community_summaries_size_mb: float = 0.0

    faiss_index_exists: bool = False
    faiss_index_size_mb: float = 0.0

    embeddings_exists: bool = False
    embeddings_count: int = 0

    @property
    def is_fully_indexed(self) -> bool:
        """True only when all artifacts required for querying exist."""
        return (
            self.chunks_exists
            and self.extractions_exists
            and self.graph_exists
            and self.community_map_exists
            and self.community_summaries_exists
            and self.faiss_index_exists
            and self.embeddings_exists
        )

    @property
    def graphrag_ready(self) -> bool:
        """True when GraphRAG query engine has what it needs."""
        return self.community_summaries_exists

    @property
    def vectorrag_ready(self) -> bool:
        """True when VectorRAG query engine has what it needs."""
        return self.faiss_index_exists and self.embeddings_exists