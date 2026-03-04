"""
services/tokenizer_service.py — Tiktoken wrapper for all token operations.

Every token count, truncation, and chunking operation in the pipeline
goes through this service. Never call tiktoken directly from pipeline code.

Key responsibilities:
  - count_tokens()         : exact tiktoken token count for any string
  - truncate_to_limit()    : cut a string to fit within N tokens
  - chunk_text()           : split a document into 600-token chunks with 100-token overlap
                             (paper Section 3.1.1 — the exact chunking strategy)
  - decode_tokens()        : convert token ids back to text (used in truncation)
  - fits_in_window()       : boolean check before sending to LLM
  - batch_count_tokens()   : count tokens for a list of strings efficiently

Paper-exact defaults:
  chunk_size    = 600 tokens
  chunk_overlap = 100 tokens
  context_window= 8000 tokens  (used throughout summarization and map-reduce)
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Iterator

import tiktoken

from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Model → encoding mapping ───────────────────────────────────────────────────
# tiktoken encoding names for the models we use.
# cl100k_base is used by: gpt-4, gpt-4o, gpt-4-turbo, text-embedding-3-small
_GPT4_ENCODING = "cl100k_base"
_FALLBACK_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """
    Return a cached tiktoken encoding.

    lru_cache ensures we only load each encoding once — loading is
    expensive (~100ms on first call, free thereafter).
    """
    return tiktoken.get_encoding(encoding_name)


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    """Return the correct tiktoken encoding for a given OpenAI model name."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        log.warning(
            "Unknown model for tiktoken, falling back to cl100k_base",
            model=model,
        )
        return _get_encoding(_FALLBACK_ENCODING)


class TokenizerService:
    """
    Stateless service wrapping tiktoken for all token operations.

    Thread-safe: tiktoken encodings are thread-safe for concurrent reads.
    All methods are synchronous — tiktoken has no async API and is fast
    enough that running in the main thread is fine. For very large batches,
    use run_in_executor() from async_utils.

    Usage:
        tokenizer = TokenizerService()
        count = tokenizer.count_tokens("Hello world")
        chunks = tokenizer.chunk_text(document_text)
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Args:
            model: OpenAI model name. Used to select the correct tokenizer.
                   All GPT-4 family models use cl100k_base.
        """
        self.model = model
        self._encoding = _encoding_for_model(model)
        self._lock = threading.Lock()  # defensive lock for any future stateful ops

        log.debug(
            "TokenizerService initialized",
            model=model,
            encoding=self._encoding.name,
        )

    # ── Core token operations ──────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """
        Return the exact tiktoken token count for a string.

        This is the authoritative token count used everywhere in the pipeline.
        Never estimate token counts — always use this method.

        Args:
            text: Any string to count tokens for.

        Returns:
            Exact number of tokens as tiktoken would count them.

        Example:
            count = tokenizer.count_tokens("Hello, world!")
            # → 4
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        """
        Encode text to a list of token IDs.

        Used internally for truncation and chunking.
        """
        if not text:
            return []
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back to text.

        Used in truncation to reconstruct text after slicing token arrays.
        Note: decoded text may differ slightly from the original due to
        byte-level tokenization at boundaries.
        """
        if not token_ids:
            return ""
        return self._encoding.decode(token_ids)

    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
        truncation_marker: str = " [TRUNCATED]",
    ) -> str:
        """
        Truncate text to fit within max_tokens, adding a marker if truncated.

        This is used when community content exceeds the 8k context window
        during summarization. The paper's strategy for leaf-level communities
        is to add edges in order of combined_degree until the limit is hit —
        this method handles the final truncation of that content.

        Args:
            text:              Text to truncate.
            max_tokens:        Maximum allowed tokens (inclusive).
            truncation_marker: String appended when truncation occurs.
                               Set to "" for clean truncation.

        Returns:
            Truncated text that fits within max_tokens.
            Original text if it already fits.

        Example:
            short = tokenizer.truncate_to_limit(long_text, max_tokens=8000)
        """
        if not text:
            return text

        token_ids = self._encoding.encode(text)
        if len(token_ids) <= max_tokens:
            return text

        # Reserve tokens for the truncation marker
        marker_tokens = len(self._encoding.encode(truncation_marker)) if truncation_marker else 0
        keep_tokens = max_tokens - marker_tokens

        if keep_tokens <= 0:
            return truncation_marker.strip() if truncation_marker else ""

        truncated_ids = token_ids[:keep_tokens]
        truncated_text = self._encoding.decode(truncated_ids)

        log.debug(
            "Text truncated",
            original_tokens=len(token_ids),
            max_tokens=max_tokens,
            result_tokens=len(token_ids[:keep_tokens]),
        )

        return truncated_text + truncation_marker

    def fits_in_window(self, text: str, max_tokens: int) -> bool:
        """
        Return True if text fits within max_tokens.

        Use this as a fast check before building context windows.

        Args:
            text:       Text to check.
            max_tokens: Token limit to check against.

        Returns:
            True if count_tokens(text) <= max_tokens.
        """
        return self.count_tokens(text) <= max_tokens

    def tokens_remaining(self, text: str, max_tokens: int) -> int:
        """
        Return how many more tokens can be added before hitting max_tokens.

        Returns negative if text already exceeds max_tokens.
        Used when building context windows iteratively.

        Example:
            remaining = tokenizer.tokens_remaining(context_so_far, 8000)
            if remaining > 100:
                context_so_far += next_chunk
        """
        return max_tokens - self.count_tokens(text)

    # ── Batch operations ───────────────────────────────────────────────────────

    def batch_count_tokens(self, texts: list[str]) -> list[int]:
        """
        Count tokens for a list of strings.

        More efficient than calling count_tokens() in a loop because
        we avoid repeated encoding overhead.

        Args:
            texts: List of strings to count.

        Returns:
            List of token counts in the same order as inputs.
        """
        return [len(ids) for ids in self._encoding.encode_batch(texts)]

    def total_tokens(self, texts: list[str]) -> int:
        """
        Return the total token count across all strings in a list.

        Used to compute corpus statistics after chunking.
        """
        return sum(self.batch_count_tokens(texts))

    # ── Chunking — paper Section 3.1.1 ────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> list[dict]:
        """
        Split a document into overlapping token-based chunks.

        Implements the paper's exact chunking strategy (Section 3.1.1):
          - chunk_size    = 600 tokens (optimal for entity extraction recall)
          - chunk_overlap = 100 tokens (prevents entity loss at boundaries)

        The paper found that 600-token chunks with gleaning produce ~2x
        more entity references than 2400-token chunks without gleaning.

        Args:
            text:          Full document text to chunk.
            chunk_size:    Target token count per chunk. Default: 600 (paper).
            chunk_overlap: Overlap tokens between consecutive chunks. Default: 100 (paper).

        Returns:
            List of chunk dicts, each containing:
              {
                "text":        str  — the chunk text,
                "token_count": int  — exact token count,
                "start_char":  int  — character offset in source document,
                "end_char":    int  — end character offset,
                "chunk_index": int  — 0-based index within this document,
              }

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.

        Example:
            chunks = tokenizer.chunk_text(document_text)
            # → [{"text": "...", "token_count": 600, "start_char": 0, ...}, ...]
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        if not text or not text.strip():
            return []

        # Encode the full document once
        all_token_ids = self._encoding.encode(text)
        total_tokens = len(all_token_ids)

        if total_tokens == 0:
            return []

        # Build chunks by sliding a window over token IDs
        step = chunk_size - chunk_overlap
        chunks = []
        chunk_index = 0

        for start_tok in range(0, total_tokens, step):
            end_tok = min(start_tok + chunk_size, total_tokens)
            chunk_ids = all_token_ids[start_tok:end_tok]

            if not chunk_ids:
                break

            chunk_text = self._encoding.decode(chunk_ids)

            # Compute approximate character offsets
            # We decode prefix tokens to find the character position
            prefix_text = self._encoding.decode(all_token_ids[:start_tok]) if start_tok > 0 else ""
            start_char = len(prefix_text)
            end_char = start_char + len(chunk_text)

            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_ids),
                "start_char": start_char,
                "end_char": end_char,
                "chunk_index": chunk_index,
            })

            chunk_index += 1

            # Stop if we've consumed all tokens
            if end_tok >= total_tokens:
                break

        log.debug(
            "Text chunked",
            total_tokens=total_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=len(chunks),
        )

        return chunks

    def chunk_text_iter(
        self,
        text: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> Iterator[dict]:
        """
        Memory-efficient iterator version of chunk_text().

        Yields chunks one at a time instead of building the full list.
        Use for very large documents (100k+ tokens) to avoid large
        intermediate lists. For normal pipeline use, chunk_text() is simpler.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        if not text or not text.strip():
            return

        all_token_ids = self._encoding.encode(text)
        total_tokens = len(all_token_ids)
        step = chunk_size - chunk_overlap
        chunk_index = 0

        for start_tok in range(0, total_tokens, step):
            end_tok = min(start_tok + chunk_size, total_tokens)
            chunk_ids = all_token_ids[start_tok:end_tok]

            if not chunk_ids:
                break

            chunk_text_str = self._encoding.decode(chunk_ids)
            prefix_text = self._encoding.decode(all_token_ids[:start_tok]) if start_tok > 0 else ""

            yield {
                "text": chunk_text_str,
                "token_count": len(chunk_ids),
                "start_char": len(prefix_text),
                "end_char": len(prefix_text) + len(chunk_text_str),
                "chunk_index": chunk_index,
            }

            chunk_index += 1
            if end_tok >= total_tokens:
                break

    # ── Context window builder ─────────────────────────────────────────────────

    def build_context_window(
        self,
        items: list[str],
        max_tokens: int = 8000,
        separator: str = "\n\n",
        shuffle: bool = False,
    ) -> tuple[str, list[int], bool]:
        """
        Greedily pack items into a context window up to max_tokens.

        This is the core building block for both the community summarization
        (paper Section 3.1.4) and the map-stage context filling
        (paper Section 3.2).

        The paper always uses 8k token context windows — pass max_tokens=8000
        unless explicitly experimenting with other values.

        Args:
            items:      List of text strings to pack (e.g. edge descriptions,
                        community summary texts).
            max_tokens: Token budget. Default: 8000 (paper-exact).
            separator:  String placed between items. Default: double newline.
            shuffle:    If True, randomly shuffle items before packing
                        (used in map stage to prevent position bias — paper).

        Returns:
            Tuple of:
              - context:        The packed context string.
              - included_ids:   0-based indices of items that were included.
              - was_truncated:  True if not all items fit.

        Example:
            context, included, truncated = tokenizer.build_context_window(
                edge_texts, max_tokens=8000
            )
        """
        if shuffle:
            import random
            indices = list(range(len(items)))
            random.shuffle(indices)
            items_ordered = [items[i] for i in indices]
            original_indices = indices
        else:
            items_ordered = items
            original_indices = list(range(len(items)))

        sep_tokens = self.count_tokens(separator)
        context_parts: list[str] = []
        included_indices: list[int] = []
        used_tokens = 0

        for i, item in enumerate(items_ordered):
            item_tokens = self.count_tokens(item)
            # Account for separator before every item except the first
            extra = sep_tokens if context_parts else 0

            if used_tokens + item_tokens + extra > max_tokens:
                # This item doesn't fit — stop filling
                was_truncated = True
                # Return what we have
                context = separator.join(context_parts)
                return context, [original_indices[j] for j in included_indices], was_truncated

            context_parts.append(item)
            included_indices.append(i)
            used_tokens += item_tokens + extra

        context = separator.join(context_parts)
        return context, [original_indices[j] for j in included_indices], False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def encoding_name(self) -> str:
        """Name of the tiktoken encoding in use."""
        return self._encoding.name

    def __repr__(self) -> str:
        return f"TokenizerService(model={self.model!r}, encoding={self.encoding_name!r})"


# ── Module-level singleton ─────────────────────────────────────────────────────
# Shared instance used by pipeline code.
# Tests should construct their own instance via TokenizerService().

_default_tokenizer: TokenizerService | None = None
_tokenizer_lock = threading.Lock()


def get_tokenizer(model: str = "gpt-4o") -> TokenizerService:
    """
    Return the module-level shared TokenizerService.

    Creates it on first call (lazy init). Thread-safe.

    Args:
        model: OpenAI model name. Only used on first call.

    Returns:
        The shared TokenizerService singleton.
    """
    global _default_tokenizer
    if _default_tokenizer is None:
        with _tokenizer_lock:
            if _default_tokenizer is None:
                _default_tokenizer = TokenizerService(model=model)
    return _default_tokenizer


__all__ = [
    "TokenizerService",
    "get_tokenizer",
]