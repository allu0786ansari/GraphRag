#!/usr/bin/env python3
"""
scripts/diagnose_extraction.py — Test Gemini API connection and one extraction.

Run this after updating .env to verify everything works before the full pipeline.

Usage:
    python scripts/diagnose_extraction.py
"""
from __future__ import annotations
import asyncio, os, sys, traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_BACKEND_DIR  = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

async def main() -> None:
    from app.config import get_settings
    from app.services.openai_service import OpenAIService
    from app.services.tokenizer_service import TokenizerService
    from app.core.pipeline.extraction import ExtractionPipeline
    from app.models.graph_models import ChunkSchema

    settings = get_settings()

    print(f"\n  Gemini API key set: {'yes (' + settings.gemini_api_key[:12] + '...)' if settings.gemini_api_key else 'NO — set GEMINI_API_KEY in backend/.env'}")
    print(f"  Model:              {settings.openai_model}")
    print(f"  Embedding model:    {settings.openai_embedding_model}")

    # ── Test 1: raw Gemini chat completion ────────────────────────────────────
    print("\n  [1/3] Testing Gemini chat completion...")
    try:
        svc = OpenAIService(
            api_key=settings.gemini_api_key,
            model=settings.openai_model,
            max_retries=1,
            timeout=30,
        )
        result = await svc.async_chat_completion(
            messages=[{"role": "user", "content": "Say hello in one word."}]
        )
        print(f"  ✓  Gemini OK: '{result.content.strip()}'")
    except Exception as e:
        print(f"  ✗  Gemini FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # ── Test 2: Gemini embeddings ─────────────────────────────────────────────
    print("\n  [2/3] Testing Gemini embeddings...")
    try:
        from app.services.embedding_service import EmbeddingService
        emb_svc = EmbeddingService(
            api_key=settings.gemini_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimension,
        )
        vector = await emb_svc.embed_text("Test embedding for GraphRAG pipeline.")
        print(f"  ✓  Embedding OK: shape={vector.shape}, norm={float(vector @ vector):.4f} (should be ~1.0)")
    except Exception as e:
        print(f"  ✗  Embedding FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    # ── Test 3: full extraction on one chunk ──────────────────────────────────
    print("\n  [3/3] Testing extraction pipeline on one chunk...")
    chunk = ChunkSchema(
        chunk_id="test_0000",
        source_document="test.json",
        text=(
            "Amazon announced record profits in Q3 2023. CEO Andy Jassy said "
            "the company plans to invest heavily in AI infrastructure. "
            "Microsoft and Google are competing for cloud market share."
        ),
        token_count=42,
        start_char=0,
        end_char=200,
        chunk_index=0,
        total_chunks_in_doc=1,
        metadata={},
    )

    try:
        tokenizer = TokenizerService(model="gemini-2.5-flash")
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
            for e in extraction.entities[:5]:
                print(f"       - {e.name} ({e.entity_type})")
        else:
            print(f"  ✗  Extraction failed: {extraction.error_message}")
    except Exception as e:
        print(f"  ✗  Extraction exception: {type(e).__name__}: {e}")
        traceback.print_exc()

    print(f"\n  All tests done. If all ✓, run:\n")
    print(f"    python scripts/run_indexing.py --data-dir data/raw/articles --max-chunks 100 --gleaning-rounds 0\n")

if __name__ == "__main__":
    asyncio.run(main())