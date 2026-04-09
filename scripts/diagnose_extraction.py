#!/usr/bin/env python3
"""
scripts/diagnose_extraction.py — Test extraction on ONE chunk and print the exact error.

Run this to diagnose why extraction is failing before running the full pipeline.

Usage:
    python scripts/diagnose_extraction.py
"""
from __future__ import annotations
import asyncio, json, os, sys, traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_BACKEND_DIR  = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

# Load .env
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

async def main() -> None:
    from app.config import get_settings
    from app.services.openai_service import OpenAIService
    from app.services.tokenizer_service import TokenizerService
    from app.core.pipeline.extraction import ExtractionPipeline
    from app.core.pipeline.chunking import ChunkingPipeline
    from app.models.graph_models import ChunkSchema

    settings = get_settings()

    print(f"\n  OpenAI model:  {settings.openai_model}")
    print(f"  API key set:   {'yes (' + settings.openai_api_key[:8] + '...)' if settings.openai_api_key else 'NO - missing!'}")

    # ── Build ONE fake chunk ──────────────────────────────────────────────────
    chunk = ChunkSchema(
        chunk_id="test_0000",
        source_document="test.json",
        text="Amazon announced record profits in Q3 2023. CEO Andy Jassy said the company "
             "plans to invest heavily in AI infrastructure. Microsoft and Google are competing "
             "for cloud market share.",
        token_count=42,
        start_char=0,
        end_char=200,
        chunk_index=0,
        total_chunks_in_doc=1,
        metadata={},
    )

    print(f"\n  Testing extraction on chunk: '{chunk.text[:60]}...'\n")

    # ── Test 1: raw OpenAI call ────────────────────────────────────────────────
    print("  [1/2] Testing raw OpenAI chat completion...")
    try:
        svc = OpenAIService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_retries=1,
            timeout=30,
        )
        result = await svc.async_chat_completion(
            messages=[{"role": "user", "content": "Say hello in one word."}]
        )
        print(f"  ✓  OpenAI OK: '{result.content[:50]}'")
    except Exception as e:
        print(f"  ✗  OpenAI FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n  Fix your OPENAI_API_KEY in backend/.env and retry.\n")
        return

    # ── Test 2: full extraction on one chunk ───────────────────────────────────
    print("\n  [2/2] Testing full extraction pipeline on one chunk...")
    try:
        tokenizer = TokenizerService(model=settings.openai_model)
        pipeline = ExtractionPipeline(
            openai_service=svc,
            tokenizer=tokenizer,
            skip_claims=True,
        )
        extraction = await pipeline.extract_chunk(chunk, gleaning_rounds=0)

        if extraction.extraction_completed:
            print(f"  ✓  Extraction OK!")
            print(f"     Entities:      {len(extraction.entities)}")
            print(f"     Relationships: {len(extraction.relationships)}")
            for e in extraction.entities[:3]:
                print(f"       - {e.name} ({e.entity_type})")
        else:
            print(f"  ✗  Extraction marked failed.")
            print(f"     Error: {extraction.error_message}")

    except Exception as e:
        print(f"\n  ✗  Extraction EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

    print()

if __name__ == "__main__":
    asyncio.run(main())