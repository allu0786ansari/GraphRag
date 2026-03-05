"""
api/routes_graph.py — Knowledge graph and community endpoints.

Endpoints:
  GET /api/v1/graph                   High-level graph statistics + top entities.
  GET /api/v1/communities/{level}     Paginated community summaries at C0/C1/C2/C3.
  GET /api/v1/communities/{level}/{id} Single community summary by ID.

Design:
  - Graph stats are computed from graph.pkl (NetworkX) and community_map.json.
  - Community lists use SummaryStore.load_summaries_paginated() for memory efficiency.
  - Loading the full graph is expensive; stats endpoints use lightweight metadata
    where possible and only load graph.pkl when entity degree info is requested.
  - All endpoints return 503 if the corpus has not been indexed yet.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status

from app.dependencies import AuthDep, PaginationDep
from app.models.response_models import (
    CommunityBrief,
    CommunityListResponse,
    GraphStatsResponse,
)
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_indexed() -> tuple:
    """
    Return (graph_store, summary_store) or raise HTTP 503.

    Raises HTTP 503 if the pipeline has never been run.
    """
    from app.config import get_settings
    from app.storage.graph_store import GraphStore
    from app.storage.summary_store import SummaryStore

    settings      = get_settings()
    graph_store   = GraphStore(artifacts_dir=settings.artifacts_dir)
    summary_store = SummaryStore(artifacts_dir=settings.artifacts_dir)

    if not graph_store.graph_exists() and not summary_store.summaries_exist():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "not_indexed",
                "message": (
                    "The corpus has not been indexed yet. "
                    "Run POST /api/v1/index first and wait for completion."
                ),
            },
        )
    return graph_store, summary_store


def _get_top_entities(graph: Any, top_n: int = 20) -> list[dict]:
    """
    Return the top_n entities by node degree from a NetworkX graph.

    Each dict: {"name": str, "type": str, "degree": int, "description": str}
    """
    nodes_by_degree = sorted(
        graph.degree(),
        key=lambda nd: nd[1],
        reverse=True,
    )[:top_n]

    result = []
    for node_name, degree in nodes_by_degree:
        attrs = graph.nodes[node_name]
        result.append({
            "name":        node_name,
            "type":        attrs.get("type", attrs.get("entity_type", "UNKNOWN")),
            "degree":      degree,
            "description": (attrs.get("description", "") or "")[:200],
        })
    return result


def _get_entity_type_distribution(graph: Any) -> dict[str, int]:
    """Count entities per type across all nodes in the graph."""
    dist: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        etype = attrs.get("type", attrs.get("entity_type", "UNKNOWN")) or "UNKNOWN"
        dist[etype] = dist.get(etype, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/graph",
    response_model=GraphStatsResponse,
    responses={
        200: {"description": "Graph statistics."},
        503: {"description": "Corpus not indexed."},
    },
    summary="Knowledge graph statistics",
    description=(
        "Returns high-level statistics about the knowledge graph: node and edge "
        "counts, communities per level, top entities by degree, and entity type "
        "distribution. Used by the frontend dashboard."
    ),
)
async def get_graph_stats(
    _auth: AuthDep,
    include_top_entities: Annotated[bool, Query(
        description="If true, include top-20 entities by degree (requires loading graph.pkl).",
    )] = True,
    include_type_distribution: Annotated[bool, Query(
        description="If true, include entity type counts (requires loading graph.pkl).",
    )] = True,
) -> GraphStatsResponse:
    """Return knowledge graph statistics and optional entity details."""
    graph_store, summary_store = _require_indexed()

    # ── Basic counts (lightweight — no full graph load) ────────────────────────
    graph_stats = graph_store.get_graph_stats()
    total_nodes = graph_stats.get("nodes", 0)
    total_edges = graph_stats.get("edges", 0)
    is_indexed  = graph_store.graph_exists()

    communities_by_level = graph_store.get_community_counts()
    total_summaries      = sum(summary_store.get_summary_counts().values())

    # Determine indexing timestamp from graph.pkl mtime
    indexed_at = None
    if graph_store.graph_exists():
        try:
            import os
            from datetime import timezone
            from datetime import datetime
            mtime = os.path.getmtime(graph_store._graph_path)
            indexed_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            pass

    # Average degree (simple calculation)
    avg_degree = (2 * total_edges / total_nodes) if total_nodes > 0 else 0.0

    # ── Entity details (require full graph load) ───────────────────────────────
    top_entities:      list[dict] = []
    type_distribution: dict[str, int] = {}

    if (include_top_entities or include_type_distribution) and graph_store.graph_exists():
        try:
            graph = graph_store.load_graph()
            if include_top_entities:
                top_entities = _get_top_entities(graph, top_n=20)
            if include_type_distribution:
                type_distribution = _get_entity_type_distribution(graph)
        except Exception as exc:
            log.warning("Could not load graph for entity details", error=str(exc))

    return GraphStatsResponse(
        is_indexed=is_indexed,
        total_nodes=total_nodes,
        total_edges=total_edges,
        communities_by_level=communities_by_level,
        total_summaries=total_summaries,
        top_entities_by_degree=top_entities,
        entity_type_distribution=type_distribution,
        avg_node_degree=round(avg_degree, 2),
        indexed_at=indexed_at,
    )


@router.get(
    "/communities/{level}",
    response_model=CommunityListResponse,
    responses={
        200: {"description": "Paginated community list."},
        400: {"description": "Invalid community level."},
        503: {"description": "Corpus not indexed."},
    },
    summary="List communities at a hierarchy level",
    description=(
        "Returns a paginated list of community summaries at the specified level. "
        "Valid levels: c0 (root, ~55 communities), c1 (~555), c2 (~1797), c3 (~2142). "
        "Each entry includes the community title, impact rating, and summary preview."
    ),
)
async def list_communities(
    level: str,
    _auth: AuthDep,
    pagination: PaginationDep,
) -> CommunityListResponse:
    """Return paginated community summaries at the specified hierarchy level."""
    # Validate level
    valid_levels = {"c0", "c1", "c2", "c3"}
    level_lower  = level.lower()
    if level_lower not in valid_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "invalid_level",
                "message": f"Invalid community level '{level}'. Must be one of: {sorted(valid_levels)}.",
            },
        )

    _, summary_store = _require_indexed()

    if not summary_store.summaries_exist():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error":   "summaries_missing",
                "message": (
                    "Community summaries have not been generated yet. "
                    "The summarization pipeline stage must complete first."
                ),
            },
        )

    try:
        page_summaries, total = summary_store.load_summaries_paginated(
            level=level_lower,
            page=pagination.page,
            page_size=pagination.page_size,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error":   "summaries_missing",
                "message": "Community summaries file not found. Run the pipeline first.",
            },
        )

    communities = [
        CommunityBrief(
            community_id=s.community_id,
            level=s.level.value if hasattr(s.level, "value") else str(s.level),
            title=s.title,
            summary_preview=s.summary[:200],
            node_count=len(s.node_ids),
            impact_rating=s.impact_rating,
        )
        for s in page_summaries
    ]

    log.debug(
        "Communities listed",
        level=level_lower,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )

    return CommunityListResponse(
        level=level_lower,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        communities=communities,
    )


@router.get(
    "/communities/{level}/{community_id}",
    responses={
        200: {"description": "Full community summary."},
        404: {"description": "Community not found."},
        503: {"description": "Corpus not indexed."},
    },
    summary="Get a single community summary",
    description=(
        "Returns the full community summary for a specific community ID at the "
        "given level, including all findings, node IDs, and impact rating."
    ),
)
async def get_community(
    level: str,
    community_id: str,
    _auth: AuthDep,
) -> dict:
    """Return the full community summary for a specific community ID."""
    valid_levels = {"c0", "c1", "c2", "c3"}
    level_lower  = level.lower()
    if level_lower not in valid_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "invalid_level",
                "message": f"Invalid community level '{level}'. Must be one of: {sorted(valid_levels)}.",
            },
        )

    _, summary_store = _require_indexed()

    try:
        summaries = summary_store.load_summaries_by_level(level_lower)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error":   "summaries_missing",
                "message": "Community summaries not found. Run the pipeline first.",
            },
        )

    # Find the requested community
    target = next(
        (s for s in summaries if s.community_id == community_id),
        None,
    )
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":   "community_not_found",
                "message": f"Community '{community_id}' not found at level '{level_lower}'.",
            },
        )

    return target.model_dump(mode="json")