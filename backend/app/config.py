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

    # ── Google Gemini (replaces OpenAI) ───────────────────────────────────────
    # Get free key at: https://aistudio.google.com — no credit card needed
    gemini_api_key: str = Field(description="Google Gemini API key from AI Studio")

    # LLM model — gemini-2.5-flash-lite has 1000 RPD free (most generous)
    openai_model: str = Field(
        default="gemini-2.5-flash-lite-preview-06-17",
        description="Gemini model for chat completions",
    )

    # Embedding model — gemini has free embeddings too (1000 RPD)
    openai_embedding_model: str = Field(
        default="gemini-embedding-exp-03-07",
        description="Gemini model for text embeddings",
    )

    # Kept as openai_* names so zero other code needs to change
    openai_max_retries: int = Field(default=3, ge=1, le=10)
    openai_timeout: int = Field(default=60, ge=10, le=300)
    openai_max_tokens: int = Field(default=4096, ge=256, le=16384)
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── Backwards compat alias so existing code reading openai_api_key works ──
    @property
    def openai_api_key(self) -> str:
        """Alias so existing service code doesn't need changing."""
        return self.gemini_api_key

    # ── Pipeline Parameters ───────────────────────────────────────────────────
    chunk_size: int = Field(default=600, ge=100, le=2400)
    chunk_overlap: int = Field(default=100, ge=0, le=300)
    gleaning_rounds: int = Field(default=2, ge=0, le=5)
    context_window_size: int = Field(default=8000, ge=1000, le=128000)
    community_level: Literal["c0", "c1", "c2", "c3"] = Field(default="c1")
    helpfulness_score_threshold: int = Field(default=0, ge=0, le=100)
    evaluation_runs: int = Field(default=5, ge=1, le=10)

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()