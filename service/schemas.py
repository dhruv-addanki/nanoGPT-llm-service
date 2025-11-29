"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to continue.")
    max_new_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate."
    )
    temperature: Optional[float] = Field(
        1.0, description="Sampling temperature (>0)."
    )
    top_k: Optional[int] = Field(None, description="Top-k sampling cutoff.")
    top_p: Optional[float] = Field(None, description="Top-p nucleus sampling cutoff (0-1).")

    @validator("prompt")
    def validate_prompt(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("prompt cannot be empty.")
        return value

    @validator("temperature")
    def validate_temperature(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("temperature must be > 0")
        return value

    @validator("top_p")
    def validate_top_p(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0 < value <= 1:
            raise ValueError("top_p must be in (0, 1].")
        return value


class GenerateResponse(BaseModel):
    completion: str
    prompt: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    loaded: bool
    queue_waiting: int
    queue_active: int

