"""
app/config.py
─────────────
Central settings module.  All configuration is read from environment variables
(loaded from .env by python-dotenv).  Every other module should import from
here — never call os.environ directly.

Design choice: pydantic-settings BaseSettings automatically reads env vars and
validates types.  This gives us free type-checking, default values, and easy
overrides in tests via env var injection.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown vars from .env
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai",
        description="Which LLM backend to use: openai, anthropic, or gemini.",
    )
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(
        default="gpt-4o-mini", description="OpenAI chat model name."
    )
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model name.",
    )

    # ── Google Gemini ─────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini chat model name (e.g. gemini-2.5-flash).",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model name (e.g. gemini-embedding-001).",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "huggingface", "gemini"] = Field(
        default="openai",
        description="Which embedding backend to use: openai, huggingface, or gemini.",
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # ── Vector Store ──────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="Directory where Chroma persists its data.",
    )

    # ── Web Search ────────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Tavily search API key.")

    # ── LangSmith Observability ───────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing when True.",
    )
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="insightagent")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _lower_provider(cls, v: str) -> str:
        return v.lower()

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def _lower_embed(cls, v: str) -> str:
        return v.lower()

    def active_llm_key(self) -> str:
        """Return the API key for the currently selected provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.gemini_api_key  # gemini

    def active_model(self) -> str:
        """Return the model name for the currently selected provider."""
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.gemini_model  # gemini


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Trade-off: lru_cache means .env changes during process lifetime are
    ignored — perfect for production, but in tests you may need to call
    get_settings.cache_clear() before re-loading different env vars.
    """
    return Settings()
