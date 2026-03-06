#!/usr/bin/env python3
"""
scripts/run_summarization.py — Run LLM community summarization stage.

Reads community_map.json + graph.pkl and writes:
  - data/processed/community_summaries.json

Each community gets an LLM-generated summary including:
  - Title and narrative summary
  - Key findings (bullet points)
  - Impact rating (0–10)
  - Representative entities

This is typically the most expensive stage after extraction.
Use --level to summarize only one community level.

Usage:
    python scripts/run_summarization.py
    python scripts/run_summarization.py --level c1
    python scripts/run_summarization.py --max-communities 20
    python scripts/run_summarization.py --resume
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
        description="Run LLM summarization on detected communities.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=_PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--level", type=str, default=None,
                        choices=["c0", "c1", "c2", "c3"],
                        help="Summarize only this community level (default: all levels)")
    parser.add_argument("--max-communities", type=int, default=None, metavar="N",
                        help="Stop after N communities (dev/cost-control)")
    parser.add_argument("--context-window", type=int, default=8000)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-summarized communities")
    parser.add_argument("--force", action="store_true",
                        help="Re-summarize all communities")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    os.environ["ARTIFACTS_DIR"] = str(args.artifacts_dir)
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "text"

    from app.config import get_settings
    from app.core.pipeline.summarization import SummarizationPipeline
    from app.storage.artifact_store import ArtifactStore
    from app.storage.summary_store import SummaryStore
    from app.utils.token_utils import estimate_cost_usd

    settings = get_settings()
    store = ArtifactStore(settings)
    summary_store = SummaryStore(settings)

    # ── Load prerequisites ────────────────────────────────────────────────────
    communities = store.load_community_map()
    if not communities:
        print("\n  ERROR: No communities found.")
        print("  Run python scripts/run_community_detection.py first.\n")
        return 1

    graph = store.load_graph()

    # ── Filter by level ───────────────────────────────────────────────────────
    if args.level:
        communities = [c for c in communities if c.level.value == args.level]
        print(f"\n  Filtering to level {args.level}: {len(communities)} communities")

    # ── Filter already summarized ─────────────────────────────────────────────
    if args.resume and not args.force:
        existing_ids = {s.community_id for s in summary_store.load_summaries()}
        before = len(communities)
        communities = [c for c in communities if c.community_id not in existing_ids]
        skipped = before - len(communities)
        if skipped:
            print(f"  ↩  Skipping {skipped} already-summarized communities")

    if args.max_communities:
        communities = communities[: args.max_communities]

    if not communities:
        print("\n  All communities already summarized. Use --force to re-run.\n")
        return 0

    # ── Estimate cost ─────────────────────────────────────────────────────────
    # ~6k tokens per community (context) + ~1k output
    est_cost = estimate_cost_usd(
        prompt_tokens=len(communities) * 6000,
        completion_tokens=len(communities) * 1000,
    )
    print(f"\n  Communities to summarize: {len(communities)}")
    print(f"  Estimated cost:           ${est_cost:.4f}")

    # ── Run summarization ─────────────────────────────────────────────────────
    print(f"\n  Running summarization (context_window={args.context_window})...\n")
    t0 = time.time()

    pipeline = SummarizationPipeline.from_settings()
    total = len(communities)
    success_count = 0

    for i, community in enumerate(communities, 1):
        print(f"\r  Summarizing {i}/{total}: {community.community_id:<40}", end="", flush=True)
        try:
            summary = await pipeline.summarize_community(
                community=community,
                graph=graph,
                context_window_size=args.context_window,
            )
            summary_store.save_summary(summary)
            success_count += 1
        except Exception as exc:
            print(f"\n  ✗ Failed on {community.community_id}: {exc}")

    elapsed = time.time() - t0
    all_summaries = summary_store.load_summaries()

    print(f"\n\n  ✓ {success_count}/{total} summaries written in {elapsed:.1f}s")
    print(f"  Total summaries on disk: {len(all_summaries)}")
    print(f"\n  Run next: start the server and query!")
    print(f"    make dev\n")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()