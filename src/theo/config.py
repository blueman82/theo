"""Configuration settings for the Theo MCP server.

This module provides Pydantic Settings for configuration management.
All settings are loaded from environment variables with the THEO_ prefix.
No defaults - all values must be explicitly set in .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Type alias for embedding backend selection
EmbeddingBackend = Literal["ollama", "mlx"]


class TheoSettings(BaseSettings):
    """Configuration settings for the Theo MCP server.

    All settings are loaded from environment variables with the THEO_ prefix.
    No defaults - all values must be explicitly set in .env file.

    Attributes:
        sqlite_path: Path to SQLite database (single source of truth for all storage)
        ollama_host: Ollama server host URL
        ollama_model: Embedding model name
        embedding_backend: Embedding backend ('ollama' or 'mlx')
        mlx_model: MLX model identifier
        log_level: Logging level
    """

    model_config = SettingsConfigDict(
        env_prefix="THEO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra env vars without error
    )

    # Storage paths (required)
    sqlite_path: Path = Field(description="Path to SQLite database")

    # Ollama configuration (required)
    ollama_host: str = Field(description="Ollama server host URL")
    ollama_model: str = Field(description="Embedding model name for Ollama backend")
    ollama_llm_model: str = Field(description="LLM model for relationship classification")
    ollama_timeout: int = Field(description="Ollama request timeout in seconds")

    # Embedding backend configuration (required)
    embedding_backend: EmbeddingBackend = Field(
        description="Embedding backend to use ('ollama' or 'mlx')"
    )
    mlx_model: str = Field(
        description="MLX embedding model identifier (used when embedding_backend='mlx')"
    )

    # Logging (required)
    log_level: str = Field(
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # Memory defaults (required)
    default_namespace: str = Field(description="Default namespace for memories")
    default_importance: float = Field(
        ge=0.0,
        le=1.0,
        description="Default importance score for memories",
    )

    # Token budget (required)
    default_token_budget: int = Field(
        description="Default token budget for context generation"
    )

    def get_sqlite_path(self) -> Path:
        """Get the SQLite path, expanding user home."""
        return self.sqlite_path.expanduser().resolve()


# Alias for backward compatibility with code expecting RecallSettings
RecallSettings = TheoSettings
