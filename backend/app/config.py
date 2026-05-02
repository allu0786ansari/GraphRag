"""
config.py — Application configuration via Pydantic BaseSettings.

All settings are loaded from environment variables or a .env file.
Every module imports settings from here — never use os.environ directly.

Migration note: switched from OpenAI to Google Gemini (AI Studio) for
both LLM and embeddings. Gemini API is free with no credit card required.
Get your key at: https://aistudio.google.com
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="GraphRAG System")
    app_version: str = Field(default="1.0.0")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = Field(default=False)

    # ── Security ──────────────────────────────────────────────────────────────
    api_key: str = Field(description="API key for endpoint authentication")
    allowed_origins: str = Field(default="http://localhost:5173,http://localhost:3000")

    # ── Google Gemini ───────────────────────────────────────
    # Get free key at: https://aistudio.google.com — no credit card needed
    gemini_api_key: str = Field(description="Google Gemini API key from AI Studio")

    # LLM model — gemini-1.0-pro for extraction and summarization
    gemini_model: str = Field(
        default="gemini-pro",
        description="Gemini model for chat completions",
    )
    openai_model: str | None = Field(
        default=None,
        description="Alias for gemini_model (for backward compatibility)",
    )

    # Embedding model — text-embedding-004 for vector indexing
    embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini model for text embeddings",
    )
    openai_embedding_model: str | None = Field(
        default=None,
        description="Alias for embedding_model (for backward compatibility)",
    )

    # Base URL for Gemini OpenAI-compatible API (deprecated, kept for reference)
    openai_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Base URL for Gemini OpenAI-compatible API (no longer used)",
    )

    # Moved from openai_* to keep names consistent with settings
    openai_max_retries: int = Field(default=3, ge=1, le=10)
    openai_timeout: int = Field(default=60, ge=10, le=300)
    openai_max_tokens: int = Field(default=4096, ge=256, le=16384)
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── Pipeline Parameters ───────────────────────────────────────────────────
    chunk_size: int = Field(default=600, ge=100, le=2400)
    chunk_overlap: int = Field(default=100, ge=0, le=300)
    gleaning_rounds: int = Field(default=2, ge=0, le=5)
    context_window_size: int = Field(default=8000, ge=1000, le=128000)
    community_level: Literal["c0", "c1", "c2", "c3"] = Field(default="c1")
    helpfulness_score_threshold: int = Field(default=0, ge=0, le=100)
    evaluation_runs: int = Field(default=5, ge=1, le=10)
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max concurrent LLM requests. Lower values are safer on Gemini free tier.",
    )

    # ── Storage Paths ─────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("../../data"))
    artifacts_dir: Path = Field(default=Path("../../data/processed"))
    raw_data_dir: Path = Field(default=Path("../../data/raw/articles"))
    evaluation_dir: Path = Field(default=Path("../../data/evaluation"))
    logs_dir: Path = Field(default=Path("./logs"))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="json")
    log_rotation: str = Field(default="100 MB")
    log_retention: str = Field(default="30 days")

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_index_path: Path = Field(default=Path("../../data/processed/faiss_index.bin"))
    embeddings_path: Path = Field(default=Path("../../data/processed/embeddings.npy"))
    # Gemini embedding-exp-03-07 outputs 3072 dims; set 768 for reduced cost
    embedding_dimension: int = Field(default=768, description="Gemini embedding output dimension")

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @model_validator(mode="after")
    def ensure_directories_exist(self) -> Settings:
        for d in [self.data_dir, self.artifacts_dir, self.raw_data_dir,
                  self.evaluation_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        normalized = model.strip().lower()
        if normalized.startswith("models/"):
            normalized = normalized[len("models/"):]
        alias_map = {
            "gemini-1.5-flash": "gemini-2.5-flash",
            "gemini-1.5-pro": "gemini-2.5-pro",
            "gemini-1.5": "gemini-2.5-pro",
            "gemini-1.0-flash": "gemini-flash-latest",
            "gemini-1.0-pro": "gemini-pro-latest",
            "gemini-pro": "gemini-pro-latest",
            "gemini-flash": "gemini-flash-latest",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
        }
        model_name = alias_map.get(normalized, normalized)
        if model_name.startswith("models/") or not model_name.startswith("gemini"):
            return model_name
        return f"models/{model_name}"

    @model_validator(mode="after")
    def normalize_legacy_model_names(self) -> Settings:
        # Preserve existing OPENAI_* values for compatibility, but use the
        # Gemini-specific settings in new code paths.
        if self.openai_model is not None:
            self.gemini_model = self.openai_model
        else:
            self.openai_model = self.gemini_model

        self.gemini_model = self._normalize_model_name(self.gemini_model)
        self.openai_model = self.gemini_model

        if self.openai_embedding_model is not None:
            self.embedding_model = self.openai_embedding_model
        else:
            self.openai_embedding_model = self.embedding_model

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()