#!/usr/bin/env python3
"""
scripts/run_extraction.py — Run only the extraction stage of the pipeline.

Runs chunking → extraction → gleaning and saves:
  - data/processed/chunks.json
  - data/processed/extractions.json

Useful when you want to:
  - Re-run extraction with different parameters without re-doing graph/communities
  - Debug extraction quality on a small subset of chunks
  - Resume a failed extraction mid-way through

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --max-chunks 20 --gleaning-rounds 0
    python scripts/run_extraction.py --resume           # skip already-extracted chunks
    python scripts/run_extraction.py --chunk-id news_article_001_0003  # one chunk only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunking + extraction + gleaning stages only.",
    )
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--artifacts-dir", type=Path, default=_PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--gleaning-rounds", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=None, metavar="N",
                        help="Stop after N chunks (dev/cost-control)")
    parser.add_argument("--chunk-id", type=str, default=None, metavar="ID",
                        help="Extract a single specific chunk by ID")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip chunks already in extractions.json (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract all chunks even if already done")
    parser.add_argument("--skip-claims", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    os.environ["RAW_DATA_DIR"] = str(args.data_dir)
    os.environ["ARTIFACTS_DIR"] = str(args.artifacts_dir)
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "text"

    from app.config import get_settings
    from app.core.pipeline.chunking import ChunkingPipeline
    from app.core.pipeline.extraction import ExtractionPipeline
    from app.storage.artifact_store import ArtifactStore
    from app.storage.cache_manager import CacheManager
    from app.utils.logger import get_logger

    log = get_logger("run_extraction")
    settings = get_settings()
    store = ArtifactStore(settings)
    cache = CacheManager(settings)

    # ── Stage 1: Chunking ─────────────────────────────────────────────────────
    print("\n  Stage 1/2 — Chunking")
    chunker = ChunkingPipeline(settings)
    chunks = chunker.chunk_directory(args.data_dir)

    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    if args.chunk_id:
        chunks = [c for c in chunks if c.chunk_id == args.chunk_id]
        if not chunks:
            print(f"  ERROR: chunk_id '{args.chunk_id}' not found.")
            return 1

    store.save_chunks(chunks)
    print(f"  ✓ {len(chunks)} chunks written to {args.artifacts_dir}/chunks.json")

    # ── Stage 2: Extraction ───────────────────────────────────────────────────
    print(f"\n  Stage 2/2 — Extraction + Gleaning (rounds={args.gleaning_rounds})")

    # Filter already-extracted chunks if resuming
    if args.resume and not args.force:
        existing = {e.chunk_id for e in store.load_extractions()}
        chunks_to_extract = [c for c in chunks if c.chunk_id not in existing]
        skipped = len(chunks) - len(chunks_to_extract)
        if skipped:
            print(f"  ↩  Skipping {skipped} already-extracted chunks (--resume)")
        chunks = chunks_to_extract

    if not chunks:
        print("  All chunks already extracted. Use --force to re-extract.")
        return 0

    extractor = ExtractionPipeline.from_settings()
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        print(f"\r  Extracting {i}/{total}: {chunk.chunk_id[:50]:<50}", end="", flush=True)
        try:
            extraction = await extractor.extract_chunk(
                chunk,
                gleaning_rounds=args.gleaning_rounds,
                skip_claims=args.skip_claims,
            )
            store.append_extraction(extraction)
        except Exception as exc:
            print(f"\n  ✗ Failed on {chunk.chunk_id}: {exc}")
            log.exception("Extraction failed", chunk_id=chunk.chunk_id, error=str(exc))

    all_extractions = store.load_extractions()
    print(f"\n\n  ✓ {len(all_extractions)} total extractions in extractions.json")
    print(f"\n  Run next: python scripts/run_community_detection.py\n")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()