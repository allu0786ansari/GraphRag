"""
utils/token_utils.py — Pure token math helpers for the GraphRAG pipeline.

These are stateless functions that perform token counting, cost estimation,
and context window arithmetic without requiring a TokenizerService instance.
They are the lightweight alternative to TokenizerService for places that just
need a quick integer estimate rather than the full service.

Three tiers of accuracy:
  1. Exact  — uses tiktoken (this module's count_tokens / truncate_text).
  2. Fast   — character-based approximation (approx_token_count), good for
              logging and pre-checks where ±10% is acceptable.
  3. Budget — cost estimation helpers for LLM call planning.

Model pricing (as of 2024-11, for logging / budgeting only — not billing):
  gpt-4o:                $2.50 / 1M input,  $10.00 / 1M output
  gpt-4o-mini:           $0.15 / 1M input,   $0.60 / 1M output
  text-embedding-3-small: $0.02 / 1M tokens (input only)
  text-embedding-3-large: $0.13 / 1M tokens (input only)

Usage examples:
    from app.utils.token_utils import (
        count_tokens, fits_in_window, truncate_text,
        approx_token_count, estimate_cost_usd, estimate_pipeline_cost,
    )

    n = count_tokens("Hello world", model="gpt-4o")          # exact
    n = approx_token_count("Hello world")                    # fast estimate
    ok = fits_in_window("some long text", max_tokens=8000)   # bool check
    text = truncate_text("very long text...", max_tokens=500) # hard truncation
    cost = estimate_cost_usd(prompt_tokens=1000,
                              completion_tokens=200,
                              model="gpt-4o")                 # float in USD
"""

from __future__ import annotations

import functools
from typing import Any

from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Default model ──────────────────────────────────────────────────────────────
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_ENCODING = "o200k_base"   # tiktoken encoding for gpt-4o family

# ── Model pricing table (USD per 1M tokens) ───────────────────────────────────
# input_usd: cost per 1M *input* (prompt) tokens
# output_usd: cost per 1M *output* (completion) tokens
# For embedding models, set output_usd=0.0.
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":                    {"input_usd":  2.50, "output_usd": 10.00},
    "gpt-4o-2024-08-06":         {"input_usd":  2.50, "output_usd": 10.00},
    "gpt-4o-2024-05-13":         {"input_usd":  5.00, "output_usd": 15.00},
    "gpt-4o-mini":               {"input_usd":  0.15, "output_usd":  0.60},
    "gpt-4o-mini-2024-07-18":    {"input_usd":  0.15, "output_usd":  0.60},
    "gpt-4-turbo":               {"input_usd": 10.00, "output_usd": 30.00},
    "gpt-4":                     {"input_usd": 30.00, "output_usd": 60.00},
    "gpt-3.5-turbo":             {"input_usd":  0.50, "output_usd":  1.50},
    "text-embedding-3-small":    {"input_usd":  0.02, "output_usd":  0.00},
    "text-embedding-3-large":    {"input_usd":  0.13, "output_usd":  0.00},
    "text-embedding-ada-002":    {"input_usd":  0.10, "output_usd":  0.00},
}

# Fallback pricing for unknown models (conservative / high estimate)
_FALLBACK_PRICING: dict[str, float] = {"input_usd": 10.00, "output_usd": 30.00}

# ── Characters-per-token approximation ────────────────────────────────────────
# English prose: ~4 chars/token for cl100k_base / o200k_base.
# This is an approximation — use count_tokens() when exact counts are needed.
_CHARS_PER_TOKEN = 4.0


# ── Tiktoken encoding cache ───────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _get_encoding(model: str):
    """Return a cached tiktoken encoding for the given model name."""
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except ImportError:
        log.warning("tiktoken not installed — token counting will use approximation")
        return None


# ── Exact token counting ───────────────────────────────────────────────────────

def count_tokens(text: str, model: str = _DEFAULT_MODEL) -> int:
    """
    Count the exact number of tokens in text using tiktoken.

    Falls back to character-based approximation if tiktoken is not installed.

    Args:
        text:  Text to count tokens for.
        model: Model name (determines the tokenizer encoding).
               Defaults to "gpt-4o".

    Returns:
        Integer token count.
    """
    if not text:
        return 0

    enc = _get_encoding(model)
    if enc is None:
        return approx_token_count(text)

    return len(enc.encode(text))


def count_tokens_for_messages(
    messages: list[dict[str, str]],
    model: str = _DEFAULT_MODEL,
) -> int:
    """
    Count tokens for a list of chat messages including role overhead.

    The OpenAI chat format adds ~4 tokens per message for formatting
    and 3 tokens for the reply prime. This matches the formula from
    OpenAI's tiktoken cookbook.

    Args:
        messages: List of {"role": str, "content": str} dicts.
        model:    Model name.

    Returns:
        Total token count including chat formatting overhead.
    """
    tokens_per_message = 4   # role + content + separators
    tokens_per_reply   = 3   # every reply is primed with assistant token

    total = tokens_per_reply
    for message in messages:
        total += tokens_per_message
        for value in message.values():
            total += count_tokens(str(value), model=model)
    return total


# ── Window / budget checks ─────────────────────────────────────────────────────

def fits_in_window(
    text: str,
    max_tokens: int,
    model: str = _DEFAULT_MODEL,
) -> bool:
    """
    Return True if text fits within max_tokens.

    Uses exact tiktoken counting.

    Args:
        text:       Text to check.
        max_tokens: Token limit.
        model:      Tokenizer model.

    Returns:
        True if len(tokens) <= max_tokens.
    """
    return count_tokens(text, model=model) <= max_tokens


def tokens_remaining(
    text: str,
    max_tokens: int,
    model: str = _DEFAULT_MODEL,
) -> int:
    """
    Return how many tokens remain in the window after text.

    Negative means the text overflows by that many tokens.

    Args:
        text:       Text already in the context.
        max_tokens: Total token budget.
        model:      Tokenizer model.

    Returns:
        max_tokens - count_tokens(text). May be negative.
    """
    return max_tokens - count_tokens(text, model=model)


def truncate_text(
    text: str,
    max_tokens: int,
    model: str = _DEFAULT_MODEL,
    suffix: str = "…",
) -> str:
    """
    Truncate text to fit within max_tokens.

    Decodes the first max_tokens token IDs back to a string.
    Falls back to character-based truncation if tiktoken is not installed.

    Args:
        text:       Text to truncate.
        max_tokens: Hard token limit.
        model:      Tokenizer model.
        suffix:     Appended to indicate truncation. Default: "…".
                    Set to "" for no indicator.

    Returns:
        Truncated text string. Original text if it already fits.
    """
    if not text or max_tokens <= 0:
        return ""

    enc = _get_encoding(model)

    if enc is None:
        # Character-based fallback
        char_limit = int(max_tokens * _CHARS_PER_TOKEN)
        if len(text) <= char_limit:
            return text
        return text[:char_limit].rstrip() + suffix

    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text

    # Reserve room for the suffix token(s)
    suffix_tokens = len(enc.encode(suffix)) if suffix else 0
    limit = max(0, max_tokens - suffix_tokens)
    truncated = enc.decode(token_ids[:limit])
    return truncated.rstrip() + (suffix if suffix else "")


# ── Fast approximation ────────────────────────────────────────────────────────

def approx_token_count(text: str) -> int:
    """
    Fast character-based token count approximation (~4 chars/token).

    This is O(1) and requires no tiktoken installation.
    Accuracy: typically ±10–15% for English prose.
    Use when speed matters more than precision (e.g. pre-filtering, logging).

    Args:
        text: Text to estimate.

    Returns:
        Approximate integer token count.
    """
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def approx_fits_in_window(text: str, max_tokens: int) -> bool:
    """
    Fast check whether text likely fits within max_tokens.

    Uses character approximation. Adds a 10% safety margin so the
    approximation errs on the side of caution.

    Args:
        text:       Text to check.
        max_tokens: Token limit.

    Returns:
        True if the text almost certainly fits.
    """
    safe_limit = int(max_tokens * 0.90)   # 10% safety margin
    return approx_token_count(text) <= safe_limit


# ── Cost estimation ────────────────────────────────────────────────────────────

def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int = 0,
    model: str = _DEFAULT_MODEL,
) -> float:
    """
    Estimate the USD cost of an LLM call.

    Uses the pricing table in this module. Falls back to a conservative
    estimate for unknown model names.

    Args:
        prompt_tokens:     Number of input tokens.
        completion_tokens: Number of output tokens. Default: 0 (for embeddings).
        model:             Model name.

    Returns:
        Estimated cost in USD. Rounded to 6 decimal places.
    """
    pricing = _PRICING.get(model)

    if pricing is None:
        # Try prefix match: "gpt-4o-2025-xx" → "gpt-4o"
        for key in _PRICING:
            if model.startswith(key):
                pricing = _PRICING[key]
                break

    if pricing is None:
        log.debug("No pricing found for model, using fallback", model=model)
        pricing = _FALLBACK_PRICING

    input_cost  = (prompt_tokens     / 1_000_000) * pricing["input_usd"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_usd"]
    return round(input_cost + output_cost, 6)


def estimate_pipeline_cost(
    n_chunks: int,
    avg_tokens_per_chunk: int = 600,
    gleaning_rounds: int = 2,
    n_communities: int = 50,
    model: str = _DEFAULT_MODEL,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, float]:
    """
    Estimate total USD cost of running the full indexing pipeline.

    Provides a per-stage breakdown and a total.  All figures are rough
    estimates — actual costs depend on actual LLM responses.

    Calculation approach (paper-aligned):
      - Extraction: 1 call per chunk × avg chunk tokens input
                    + gleaning_rounds checks (1 token output each, forced YES/NO)
                    + gleaning_rounds continuations (avg ~200 tokens output each)
      - Summarization: 1 call per community × 8k context window
      - Map stage (query): 1 call per community × ~1k context
      - Reduce stage (query): 1 call × ~8k context
      - Embeddings: 1 embed per chunk

    Args:
        n_chunks:               Number of text chunks in the corpus.
        avg_tokens_per_chunk:   Average tokens per chunk. Paper: 600.
        gleaning_rounds:        Gleaning iterations. Paper: 2.
        n_communities:          Estimated communities. Depends on corpus size.
        model:                  LLM model for extraction, summarization, query.
        embedding_model:        Embedding model for vector indexing.

    Returns:
        Dict with keys: extraction, summarization, map_query, reduce_query,
        embeddings, total_usd.
    """
    # ── Extraction cost ───────────────────────────────────────────────────────
    # Primary extraction: avg_tokens_per_chunk input + ~300 tokens output
    extraction_input  = n_chunks * avg_tokens_per_chunk
    extraction_output = n_chunks * 300

    # Gleaning: check (1 token output) + continuation (~200 tokens output)
    gleaning_output = n_chunks * gleaning_rounds * (1 + 200)

    extraction_cost = estimate_cost_usd(
        prompt_tokens=extraction_input,
        completion_tokens=extraction_output + gleaning_output,
        model=model,
    )

    # ── Summarization cost ────────────────────────────────────────────────────
    # 1 call per community: ~6000 token context + ~1000 tokens output
    summ_input  = n_communities * 6_000
    summ_output = n_communities * 1_000

    summarization_cost = estimate_cost_usd(
        prompt_tokens=summ_input,
        completion_tokens=summ_output,
        model=model,
    )

    # ── Map stage query cost (per query) ──────────────────────────────────────
    # 1 call per community: ~1500 tokens in + ~100 tokens out
    map_input  = n_communities * 1_500
    map_output = n_communities * 100

    map_cost = estimate_cost_usd(
        prompt_tokens=map_input,
        completion_tokens=map_output,
        model=model,
    )

    # ── Reduce stage query cost (per query) ───────────────────────────────────
    # Single call: ~4000 tokens in (partial answers) + ~800 tokens out
    reduce_cost = estimate_cost_usd(
        prompt_tokens=4_000,
        completion_tokens=800,
        model=model,
    )

    # ── Embedding cost ────────────────────────────────────────────────────────
    total_embed_tokens = n_chunks * avg_tokens_per_chunk
    embedding_cost = estimate_cost_usd(
        prompt_tokens=total_embed_tokens,
        completion_tokens=0,
        model=embedding_model,
    )

    total = round(
        extraction_cost + summarization_cost + map_cost + reduce_cost + embedding_cost,
        4,
    )

    return {
        "extraction_usd":     round(extraction_cost, 4),
        "summarization_usd":  round(summarization_cost, 4),
        "map_query_usd":      round(map_cost, 4),
        "reduce_query_usd":   round(reduce_cost, 4),
        "embeddings_usd":     round(embedding_cost, 4),
        "total_usd":          total,
        # Human-readable inputs
        "n_chunks":           n_chunks,
        "n_communities":      n_communities,
        "model":              model,
        "embedding_model":    embedding_model,
    }


# ── Convenience re-exports ────────────────────────────────────────────────────

__all__ = [
    # Exact counting
    "count_tokens",
    "count_tokens_for_messages",
    # Window checks
    "fits_in_window",
    "tokens_remaining",
    "truncate_text",
    # Fast approximation
    "approx_token_count",
    "approx_fits_in_window",
    # Cost estimation
    "estimate_cost_usd",
    "estimate_pipeline_cost",
    # Pricing table (for external inspection)
    "_PRICING",
]