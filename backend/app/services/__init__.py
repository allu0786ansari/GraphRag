"""
services/__init__.py — Public re-exports for all service classes.

Every external dependency (Gemini, FAISS, tokenization) is wrapped here.
Pipeline code imports from app.services, never from google/faiss/tiktoken directly.
"""

from app.services.tokenizer_service import TokenizerService, get_tokenizer
from app.services.llm_service import (
    LLMService,
    CompletionResult,
    get_llm_service,
    build_messages,
    system_message,
    user_message,
    assistant_message,
)
from app.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    EMBEDDING_DIM_SMALL,
    EMBEDDING_DIM_LARGE,
)
from app.services.faiss_service import (
    FAISSService,
    SearchResult,
    get_faiss_service,
)

__all__ = [
    # Tokenizer
    "TokenizerService",
    "get_tokenizer",
    # LLM
    "LLMService",
    "CompletionResult",
    "get_llm_service",
    "build_messages",
    "system_message",
    "user_message",
    "assistant_message",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    "EMBEDDING_DIM_SMALL",
    "EMBEDDING_DIM_LARGE",
    # FAISS
    "FAISSService",
    "SearchResult",
    "get_faiss_service",
]