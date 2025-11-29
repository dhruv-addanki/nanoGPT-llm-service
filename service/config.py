"""Configuration management for the nanoGPT inference service."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from pydantic import BaseSettings, Field, validator


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


class ServiceSettings(BaseSettings):
    """Pydantic-powered settings for the inference service."""

    model_path: Optional[str] = Field(
        None, env="MODEL_PATH", description="Path to a nanoGPT checkpoint (ckpt.pt)."
    )
    checkpoint_path: Optional[str] = Field(
        None,
        env="CHECKPOINT_PATH",
        description="Alias for model_path for compatibility with training outputs.",
    )
    model_type: str = Field(
        "gpt2",
        env="MODEL_TYPE",
        description="GPT-2 variant to load when a checkpoint path is not provided.",
    )
    device: str = Field(
        default_factory=_default_device,
        env="DEVICE",
        description="Target device for inference (cuda|mps|cpu).",
    )
    max_context_tokens: int = Field(
        512,
        env="MAX_CONTEXT_TOKENS",
        description="Maximum allowed prompt length in tokens.",
    )
    max_new_tokens_default: int = Field(
        64, env="MAX_NEW_TOKENS_DEFAULT", description="Default tokens to generate."
    )
    max_new_tokens_limit: int = Field(
        256, env="MAX_NEW_TOKENS_LIMIT", description="Hard cap on generated tokens."
    )
    log_level: str = Field(
        "INFO", env="LOG_LEVEL", description="Logging level for the service."
    )
    port: int = Field(8000, env="PORT", description="Port for the HTTP server.")
    host: str = Field("0.0.0.0", env="HOST", description="Host for the HTTP server.")
    max_concurrent_generations: int = Field(
        2,
        env="MAX_CONCURRENT_GENERATIONS",
        description="Maximum concurrent generation requests.",
    )
    enable_queue: bool = Field(
        True,
        env="ENABLE_QUEUE",
        description="Track queue length and active generations.",
    )
    mock_model: bool = Field(
        False,
        env="MOCK_MODEL",
        description="Use a tiny mock model instead of loading full weights (for tests).",
    )
    request_timeout_seconds: float = Field(
        60.0, env="REQUEST_TIMEOUT_SECONDS", description="Request timeout budget."
    )
    stream_chunk_size: int = Field(
        1, env="STREAM_CHUNK_SIZE", description="Number of tokens per streamed chunk."
    )
    log_file: Optional[str] = Field(
        None, env="LOG_FILE", description="Optional file path for service logs."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("device")
    def validate_device(cls, value: str) -> str:
        normalized = value.lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized == "mps" and not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "cpu"
        return normalized

    @validator("max_new_tokens_default", "max_new_tokens_limit")
    def positive_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Token limits must be positive integers.")
        return value

    @validator("log_level")
    def uppercase_log_level(cls, value: str) -> str:
        return value.upper()

    @property
    def resolved_model_path(self) -> Optional[str]:
        """Return a single checkpoint path if provided via either env."""
        return self.model_path or self.checkpoint_path

    def configure_logging(self) -> None:
        """Configure root logging according to settings."""
        handlers = [logging.StreamHandler()]
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(self.log_file))
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            handlers=handlers,
        )


settings = ServiceSettings()
