#!/usr/bin/env python3
"""
scripts/run_community_detection.py — Run graph build + community detection.

Reads data/processed/extractions.json and produces:
  - data/processed/graph.pkl          (NetworkX graph)
  - data/processed/community_map.json (Leiden community hierarchy)

Run after extraction is complete. Requires no LLM calls — purely
algorithmic (graph deduplication + Leiden community detection).

Usage:
    python scripts/run_community_detection.py
    python scripts/run_community_detection.py --max-levels 2
    python scripts/run_community_detection.py --min-community-size 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build knowledge graph and detect communities.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=_PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--max-levels", type=int, default=3,
                        help="Maximum community hierarchy depth (default: 3 → C0..C3)")
    parser.add_argument("--min-community-size", type=int, default=2,
                        help="Minimum nodes per community (smaller merged into parent)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    os.environ["ARTIFACTS_DIR"] = str(args.artifacts_dir)
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "text"

    from app.config import get_settings
    from app.core.pipeline.graph_builder import GraphBuilder
    from app.core.pipeline.community_detection import CommunityDetector
    from app.storage.artifact_store import ArtifactStore

    settings = get_settings()
    store = ArtifactStore(settings)

    # ── Load extractions ──────────────────────────────────────────────────────
    extractions = store.load_extractions()
    if not extractions:
        print("\n  ERROR: No extractions found.")
        print("  Run python scripts/run_extraction.py first.\n")
        return 1

    print(f"\n  Loaded {len(extractions)} extractions")

    # ── Stage 1: Build graph ──────────────────────────────────────────────────
    print("\n  Stage 1/2 — Building knowledge graph...")
    t0 = time.time()

    builder = GraphBuilder(settings)
    graph = builder.build(extractions)

    print(f"  ✓ Graph built in {time.time()-t0:.1f}s")
    print(f"    Nodes: {graph.number_of_nodes()}")
    print(f"    Edges: {graph.number_of_edges()}")
    store.save_graph(graph)

    # ── Stage 2: Community detection ─────────────────────────────────────────
    print(f"\n  Stage 2/2 — Detecting communities (max_levels={args.max_levels})...")
    t1 = time.time()

    detector = CommunityDetector(settings)
    communities = detector.detect(graph, max_levels=args.max_levels)

    level_counts: dict[str, int] = {}
    for c in communities:
        level_counts[c.level.value] = level_counts.get(c.level.value, 0) + 1

    print(f"  ✓ Communities detected in {time.time()-t1:.1f}s")
    for level, count in sorted(level_counts.items()):
        print(f"    {level}: {count} communities")

    store.save_community_map(communities)

    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print(f"\n  Run next: python scripts/run_summarization.py\n")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()