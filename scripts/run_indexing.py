#!/usr/bin/env python3
"""
scripts/run_indexing.py — Run the full GraphRAG indexing pipeline from the CLI.

This script wires the PipelineRunner directly without starting the FastAPI
server. Useful for:
  - Initial indexing of a new corpus
  - Re-indexing after document updates
  - CI pipelines and batch processing
  - Development testing with --max-chunks

Pipeline stages (in order):
  1. Chunking       — split documents into 600-token chunks
  2. Extraction     — extract entities + relationships via LLM
  3. Gleaning       — self-reflection to catch missed entities
  4. Graph Build    — deduplicate + merge into NetworkX graph
  5. Communities    — Leiden/Louvain hierarchical community detection
  6. Summarization  — LLM summary per community at each level
  7. Embeddings     — embed all chunks + build FAISS index

Usage:
    python scripts/run_indexing.py
    python scripts/run_indexing.py --data-dir data/raw --force
    python scripts/run_indexing.py --max-chunks 50 --gleaning-rounds 0
    python scripts/run_indexing.py --resume

Cost estimate before running:
    python scripts/run_indexing.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the project root: python scripts/run_indexing.py
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"

sys.path.insert(0, str(_BACKEND_DIR))

# Set a default .env path so Settings can find it
os.environ.setdefault("ENV_FILE", str(_BACKEND_DIR / ".env"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GraphRAG indexing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with paper-exact parameters
  python scripts/run_indexing.py

  # Quick dev run: 50 chunks, no gleaning, so cheap + fast
  python scripts/run_indexing.py --max-chunks 50 --gleaning-rounds 0

  # Force full re-index (deletes existing artifacts)
  python scripts/run_indexing.py --force

  # Resume from last checkpoint (default behaviour)
  python scripts/run_indexing.py --resume

  # Estimate cost without running
  python scripts/run_indexing.py --dry-run
        """,
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw",
        help="Directory containing input .json documents (default: data/raw)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed",
        help="Directory to write pipeline artifacts (default: data/processed)",
    )

    # ── Pipeline parameters ───────────────────────────────────────────────────
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        metavar="N",
        help="Tokens per chunk — paper uses 600 (default: 600)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        metavar="N",
        help="Overlap tokens between chunks (default: 100)",
    )
    parser.add_argument(
        "--gleaning-rounds",
        type=int,
        default=2,
        metavar="N",
        help="Self-reflection gleaning iterations — paper uses 2 (default: 2)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=8000,
        metavar="N",
        help="LLM context window size in tokens — paper uses 8000 (default: 8000)",
    )
    parser.add_argument(
        "--max-community-levels",
        type=int,
        default=3,
        metavar="N",
        help="Max community hierarchy depth — produces C0..CN (default: 3)",
    )

    # ── Execution control ─────────────────────────────────────────────────────
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N chunks — use for dev/testing to control cost",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing artifacts and re-run from scratch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)",
    )
    parser.add_argument(
        "--skip-claims",
        action="store_true",
        help="Skip claim extraction during entity/relationship extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost without running the pipeline",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose / DEBUG logging",
    )

    return parser.parse_args()


def estimate_cost(data_dir: Path, chunk_size: int, gleaning_rounds: int) -> None:
    """Print a cost estimate before running."""
    from app.utils.token_utils import estimate_pipeline_cost

    # Count source documents
    doc_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.txt"))
    if not doc_files:
        print(f"  No documents found in {data_dir}")
        return

    # Rough chunk count: average 2000 tokens per doc → ~3 chunks each at 600
    avg_tokens_per_doc = 2000
    estimated_chunks = max(1, sum(
        max(1, round(f.stat().st_size / 4 / chunk_size))
        for f in doc_files
    ))
    estimated_communities = max(5, estimated_chunks // 10)

    breakdown = estimate_pipeline_cost(
        n_chunks=estimated_chunks,
        avg_tokens_per_chunk=chunk_size,
        gleaning_rounds=gleaning_rounds,
        n_communities=estimated_communities,
    )

    print("\n  ── Cost Estimate ────────────────────────────────")
    print(f"  Documents found:       {len(doc_files)}")
    print(f"  Estimated chunks:      ~{estimated_chunks}")
    print(f"  Estimated communities: ~{estimated_communities}")
    print(f"  Extraction:            ${breakdown['extraction_usd']:.4f}")
    print(f"  Summarization:         ${breakdown['summarization_usd']:.4f}")
    print(f"  Embeddings:            ${breakdown['embeddings_usd']:.4f}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Total estimate:        ${breakdown['total_usd']:.4f}")
    print(f"  Model:                 {breakdown['model']}")
    print("  (Estimates are rough — actual cost depends on document complexity)\n")


async def run(args: argparse.Namespace) -> int:
    """Main async entry point. Returns exit code."""
    from app.config import get_settings
    from app.core.pipeline.pipeline_runner import PipelineRunner
    from app.models.request_models import IndexRequest
    from app.utils.logger import get_logger

    log = get_logger("run_indexing")

    # ── Override settings from CLI args ───────────────────────────────────────
    os.environ["DATA_DIR"] = str(args.artifacts_dir.parent)
    os.environ["ARTIFACTS_DIR"] = str(args.artifacts_dir)
    os.environ["RAW_DATA_DIR"] = str(args.data_dir)
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "text"

    settings = get_settings()

    # ── Dry run: estimate only ─────────────────────────────────────────────────
    if args.dry_run:
        print("\n  GraphRAG Pipeline — Cost Estimate")
        estimate_cost(args.data_dir, args.chunk_size, args.gleaning_rounds)
        return 0

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not args.data_dir.exists():
        print(f"\n  ERROR: --data-dir not found: {args.data_dir}")
        print("  Create the directory and add .json documents before indexing.\n")
        return 1

    doc_files = list(args.data_dir.glob("*.json")) + list(args.data_dir.glob("*.txt"))
    if not doc_files:
        print(f"\n  ERROR: No documents found in {args.data_dir}")
        print("  Add .json files with {'text': '...'} format and try again.\n")
        return 1

    print(f"\n  GraphRAG Indexing Pipeline")
    print(f"  {'─' * 48}")
    print(f"  Documents:        {len(doc_files)} files in {args.data_dir}")
    print(f"  Artifacts:        {args.artifacts_dir}")
    print(f"  Chunk size:       {args.chunk_size} tokens")
    print(f"  Chunk overlap:    {args.chunk_overlap} tokens")
    print(f"  Gleaning rounds:  {args.gleaning_rounds}")
    print(f"  Context window:   {args.context_window} tokens")
    print(f"  Max chunks:       {args.max_chunks or 'all'}")
    print(f"  Force reindex:    {args.force}")
    if args.max_chunks:
        print(f"  ⚠  max-chunks={args.max_chunks}: development mode, not full corpus")
    print(f"  {'─' * 48}\n")

    # Show cost estimate before starting
    estimate_cost(args.data_dir, args.chunk_size, args.gleaning_rounds)

    # ── Build request and run ─────────────────────────────────────────────────
    request = IndexRequest(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        gleaning_rounds=args.gleaning_rounds,
        context_window_size=args.context_window,
        max_community_levels=args.max_community_levels,
        force_reindex=args.force,
        skip_claims=args.skip_claims,
        max_chunks=args.max_chunks,
    )

    def on_progress(stage: str, pct: float) -> None:
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {stage:<30}", end="", flush=True)

    start = time.time()
    runner = PipelineRunner.from_settings()

    try:
        result = await runner.run(
            request=request,
            on_progress=on_progress,
        )
    except KeyboardInterrupt:
        print("\n\n  ⚠  Interrupted — pipeline checkpointed. Run again to resume.\n")
        return 130
    except Exception as exc:
        print(f"\n\n  ✗  Pipeline failed: {exc}\n")
        log.exception("Pipeline failed", error=str(exc))
        return 1

    elapsed = time.time() - start

    print(f"\n\n  {'─' * 48}")
    if result.success:
        print(f"  ✓  Pipeline complete in {elapsed:.1f}s")
        print(f"\n  Artifacts written to: {args.artifacts_dir}")
        print(f"\n  Stage breakdown:")
        for stage, t in result.stage_elapsed.items():
            print(f"    {stage:<30} {t:.1f}s")
    else:
        print(f"  ✗  Pipeline failed after {elapsed:.1f}s")
        print(f"     Check logs for details.")
        return 1

    print(f"\n  Next step: start the backend and query the index!")
    print(f"    make dev\n")
    return 0


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()