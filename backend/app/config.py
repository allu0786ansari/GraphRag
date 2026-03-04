"""
config.py — Application configuration via Pydantic BaseSettings.

All settings are loaded from environment variables or a .env file.
Every module imports settings from here — never use os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the GraphRAG system.

    Pydantic-settings reads values in this priority order:
      1. Environment variables
      2. .env file
      3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # ignore unknown env vars gracefully
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="GraphRAG System", description="Application display name")
    app_version: str = Field(default="1.0.0", description="Semantic version string")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count")
    reload: bool = Field(default=False, description="Hot reload — development only")

    # ── Security ──────────────────────────────────────────────────────────────
    api_key: str = Field(description="API key for endpoint authentication")
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated CORS allowed origins",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="Chat completion model")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model"
    )
    openai_max_retries: int = Field(default=3, ge=1, le=10)
    openai_timeout: int = Field(default=60, ge=10, le=300, description="Request timeout in seconds")
    openai_max_tokens: int = Field(default=4096, ge=256, le=16384)
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── Pipeline Parameters ───────────────────────────────────────────────────
    chunk_size: int = Field(
        default=600, ge=100, le=2400,
        description="Token size per chunk — paper uses 600",
    )
    chunk_overlap: int = Field(
        default=100, ge=0, le=300,
        description="Overlap tokens between consecutive chunks",
    )
    gleaning_rounds: int = Field(
        default=2, ge=0, le=5,
        description="Self-reflection gleaning iterations per chunk",
    )
    context_window_size: int = Field(
        default=8000, ge=1000, le=128000,
        description="Token limit for all LLM context windows — paper uses 8k",
    )
    community_level: Literal["c0", "c1", "c2", "c3"] = Field(
        default="c1", description="Default community level for queries"
    )
    helpfulness_score_threshold: int = Field(
        default=0, ge=0, le=100,
        description="Discard map-stage answers at or below this helpfulness score",
    )
    evaluation_runs: int = Field(
        default=5, ge=1, le=10,
        description="Number of LLM-as-judge runs per question pair for majority vote",
    )

    # ── Storage Paths ─────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("../../data"))
    artifacts_dir: Path = Field(default=Path("../../data/processed"))
    raw_data_dir: Path = Field(default=Path("../../data/raw"))
    evaluation_dir: Path = Field(default=Path("../../data/evaluation"))
    logs_dir: Path = Field(default=Path("./logs"))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Use json in production, text in development"
    )
    log_rotation: str = Field(default="100 MB")
    log_retention: str = Field(default="30 days")

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window: int = Field(default=60, ge=1, description="Window in seconds")

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_index_path: Path = Field(default=Path("../../data/processed/faiss_index.bin"))
    embeddings_path: Path = Field(default=Path("../../data/processed/embeddings.npy"))
    embedding_dimension: int = Field(default=1536, description="text-embedding-3-small dimension")

    # ── Computed Properties ───────────────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
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
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("reload")
    @classmethod
    def no_reload_in_production(cls, v: bool, info) -> bool:
        env = info.data.get("app_env", "development")
        if v and env == "production":
            raise ValueError("reload=True is not allowed in production")
        return v

    @model_validator(mode="after")
    def ensure_directories_exist(self) -> Settings:
        """Create required directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.artifacts_dir,
            self.raw_data_dir,
            self.evaluation_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def warn_debug_in_production(self) -> Settings:
        if self.debug and self.is_production:
            import warnings
            warnings.warn(
                "DEBUG=True is set in a production environment. "
                "This may expose sensitive information.",
                stacklevel=2,
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Using lru_cache ensures the .env file is read exactly once.
    In tests, call get_settings.cache_clear() to reload settings.
    """
    return Settings()