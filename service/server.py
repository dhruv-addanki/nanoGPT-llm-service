"""FastAPI application exposing generation endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from service.config import ServiceSettings, settings
from service.models import NanoGPTModelService
from service.queue import GenerationQueue
from service.schemas import GenerateRequest, GenerateResponse, HealthResponse
from service.tokenizer import GPT2BPETokenizer

logger = logging.getLogger("service.server")


def create_app(runtime_settings: ServiceSettings = settings) -> FastAPI:
    runtime_settings.configure_logging()

    tokenizer = GPT2BPETokenizer()
    queue = GenerationQueue(runtime_settings.max_concurrent_generations)
    model_service = NanoGPTModelService(
        settings=runtime_settings, tokenizer=tokenizer, semaphore=queue.semaphore
    )

    app = FastAPI(
        title="nanoGPT Inference Service",
        version="0.1.0",
        description="A minimal production-style API for nanoGPT generation.",
    )

    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health():
        stats = await queue.stats()
        return HealthResponse(
            status="ok",
            model=model_service._model_name,
            device=str(model_service.device),
            loaded=True,
            queue_waiting=stats.waiting,
            queue_active=stats.active,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest):
        async with queue.acquire():
            try:
                result = await model_service.generate(
                    prompt=body.prompt,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature or 1.0,
                    top_k=body.top_k,
                    top_p=body.top_p,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return GenerateResponse(
            completion=result.completion,
            prompt=result.prompt,
            model=result.model_name,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
        )

    async def _sse_stream(body: GenerateRequest) -> AsyncGenerator[bytes, None]:
        async with queue.acquire():
            try:
                async for chunk in model_service.stream_generate(
                    prompt=body.prompt,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature or 1.0,
                    top_k=body.top_k,
                    top_p=body.top_p,
                ):
                    payload = f"data: {chunk}\n\n".encode("utf-8")
                    yield payload
            except ValueError as exc:
                error_payload = f"event: error\ndata: {str(exc)}\n\n".encode("utf-8")
                yield error_payload
            yield b"data: [DONE]\n\n"

    @app.get("/generate/stream")
    async def generate_stream(params: GenerateRequest = Depends()) -> StreamingResponse:
        generator = _sse_stream(params)
        return StreamingResponse(generator, media_type="text/event-stream")

    return app


app = create_app()
