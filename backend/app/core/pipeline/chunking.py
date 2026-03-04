"""
core/pipeline/chunking.py — Split corpus documents into 600-token chunks.

Implements paper Section 3.1.1:
  "We divide each source document into chunks of 600 tokens with 100 token
   overlap between consecutive chunks."

Input:  raw_data_dir/*.json files — each file is one source document.
Output: list[ChunkSchema] → saved to artifacts_dir/chunks.json

Source document format expected:
  {
    "text": "Full document text...",
    "metadata": {"date": "...", "category": "...", ...}   ← optional
  }

If a file has no "text" key, its content is used as-is (plain text files
are also supported).

The chunker uses TokenizerService for exact tiktoken token counts, ensuring
the chunk_size boundary is a true token boundary and not just a character
approximation.

Chunk IDs are deterministic:
  "{document_stem}_{chunk_index:04d}"
  e.g. "news_article_001_0000" for the first chunk of news_article_001.json

This determinism is critical for the cache_manager — chunk IDs are how
the pipeline identifies which chunks have already been extracted on resume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.models.graph_models import ChunkSchema
from app.services.tokenizer_service import TokenizerService
from app.utils.logger import get_logger

log = get_logger(__name__)


class ChunkingPipeline:
    """
    Splits a directory of raw source documents into ChunkSchema objects.

    Usage:
        pipeline = ChunkingPipeline(tokenizer)
        chunks = pipeline.run(raw_data_dir, chunk_size=600, chunk_overlap=100)
    """

    def __init__(self, tokenizer: TokenizerService) -> None:
        self.tokenizer = tokenizer

    def run(
        self,
        raw_data_dir: Path,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        max_chunks: int | None = None,
    ) -> list[ChunkSchema]:
        """
        Process all documents in raw_data_dir and return chunks.

        Args:
            raw_data_dir:  Directory containing source document files.
                           Supports .json, .txt, .md files.
            chunk_size:    Tokens per chunk. Default: 600 (paper-exact).
            chunk_overlap: Overlap tokens. Default: 100 (paper-exact).
            max_chunks:    If set, stop after producing this many chunks total.
                           Used for development to limit API cost.

        Returns:
            List of ChunkSchema objects, ordered by (document, chunk_index).

        Raises:
            FileNotFoundError: If raw_data_dir does not exist.
        """
        raw_data_dir = Path(raw_data_dir)
        if not raw_data_dir.exists():
            raise FileNotFoundError(f"raw_data_dir not found: {raw_data_dir}")

        t0 = time.monotonic()
        doc_files = sorted(_find_document_files(raw_data_dir))

        if not doc_files:
            log.warning("No documents found in raw_data_dir", path=str(raw_data_dir))
            return []

        log.info(
            "Chunking started",
            documents=len(doc_files),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )

        all_chunks: list[ChunkSchema] = []

        for doc_path in doc_files:
            if max_chunks and len(all_chunks) >= max_chunks:
                break

            try:
                doc_chunks = self._chunk_document(
                    doc_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                all_chunks.extend(doc_chunks)

                log.debug(
                    "Document chunked",
                    file=doc_path.name,
                    chunks=len(doc_chunks),
                )
            except Exception as e:
                log.error(
                    "Failed to chunk document — skipping",
                    file=doc_path.name,
                    error=str(e),
                )

        # Apply max_chunks limit after full processing
        if max_chunks and len(all_chunks) > max_chunks:
            all_chunks = all_chunks[:max_chunks]

        elapsed = time.monotonic() - t0
        total_tokens = sum(c.token_count for c in all_chunks)

        log.info(
            "Chunking complete",
            total_chunks=len(all_chunks),
            total_tokens=total_tokens,
            documents_processed=len(doc_files),
            elapsed_seconds=round(elapsed, 2),
            avg_tokens_per_chunk=round(total_tokens / len(all_chunks), 1) if all_chunks else 0,
        )

        return all_chunks

    def chunk_document(
        self,
        text: str,
        source_document: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> list[ChunkSchema]:
        """
        Chunk a single document text string directly.

        Public method for testing and direct use.

        Args:
            text:            Full document text.
            source_document: Source filename (used for chunk_id generation).
            metadata:        Optional document metadata to attach to each chunk.
            chunk_size:      Tokens per chunk.
            chunk_overlap:   Overlap tokens.

        Returns:
            List of ChunkSchema for this document.
        """
        if not text or not text.strip():
            log.warning("Empty document text, skipping", source=source_document)
            return []

        raw_chunks = self.tokenizer.chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not raw_chunks:
            return []

        total = len(raw_chunks)
        stem = Path(source_document).stem

        chunks = []
        for raw in raw_chunks:
            chunk_id = f"{stem}_{raw['chunk_index']:04d}"
            chunks.append(ChunkSchema(
                chunk_id=chunk_id,
                source_document=source_document,
                text=raw["text"],
                token_count=raw["token_count"],
                start_char=raw["start_char"],
                end_char=raw["end_char"],
                chunk_index=raw["chunk_index"],
                total_chunks_in_doc=total,
                metadata=metadata or {},
            ))

        return chunks

    def _chunk_document(
        self,
        doc_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[ChunkSchema]:
        """Load a document file and produce chunks."""
        text, metadata = _load_document(doc_path)
        return self.chunk_document(
            text=text,
            source_document=doc_path.name,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def get_stats(self, chunks: list[ChunkSchema]) -> dict:
        """Return summary statistics about a list of chunks."""
        if not chunks:
            return {"total": 0}

        token_counts = [c.token_count for c in chunks]
        docs = {c.source_document for c in chunks}

        return {
            "total_chunks": len(chunks),
            "total_documents": len(docs),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }


# ── File loading helpers ───────────────────────────────────────────────────────

def _find_document_files(directory: Path) -> list[Path]:
    """Return all supported document files in directory (non-recursive)."""
    extensions = {".json", ".txt", ".md"}
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
        and not f.name.startswith(".")
    ]


def _load_document(path: Path) -> tuple[str, dict]:
    """
    Load a document file and return (text, metadata).

    Supports:
      .json: expects {"text": "...", "metadata": {...}}
             also accepts raw string JSON
      .txt / .md: entire file content is the text
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path.name}: {e}")

        if isinstance(data, str):
            return data, {}
        elif isinstance(data, dict):
            text = data.get("text", "")
            if not text:
                # Try alternative keys
                text = data.get("content", data.get("body", data.get("article", "")))
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            # Include top-level fields as metadata (excluding text/content/body)
            for k, v in data.items():
                if k not in ("text", "content", "body", "article", "metadata"):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        metadata.setdefault(k, v)
            return str(text), metadata
        else:
            raise ValueError(f"Unexpected JSON structure in {path.name}: {type(data).__name__}")
    else:
        # Plain text / markdown
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, {}


# ── Factory function ───────────────────────────────────────────────────────────

def get_chunking_pipeline() -> ChunkingPipeline:
    """Build a ChunkingPipeline from application settings."""
    from app.services.tokenizer_service import get_tokenizer
    return ChunkingPipeline(tokenizer=get_tokenizer())


__all__ = [
    "ChunkingPipeline",
    "get_chunking_pipeline",
]