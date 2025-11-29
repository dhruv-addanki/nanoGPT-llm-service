# Mini LLM Service Architecture

This fork keeps the original nanoGPT training code intact and layers a small production-style inference service on top.

## Components

- `service/config.py` – central configuration using environment-aware settings (model path, device, limits, logging, concurrency).
- `service/tokenizer.py` – tokenizer abstraction (GPT-2 BPE via `tiktoken`) with encode/decode helpers.
- `service/models.py` – `NanoGPTModelService` for loading checkpoints or GPT-2 weights, device placement, and text generation.
- `service/schemas.py` – Pydantic request/response models for the HTTP API.
- `service/server.py` – FastAPI app with health, generate, and streaming endpoints plus timing middleware.
- `service/queue.py` – lightweight in-process concurrency limiter/queue metrics.
- `scripts/run_server.py` – entrypoint for running the API with Uvicorn.

## Request Flow

1. HTTP request hits FastAPI (`/generate` or streaming endpoint).
2. Request validated with Pydantic models and prompt/token limits checked via the tokenizer.
3. Prompt is encoded to token IDs (`tokenizer.encode`).
4. `NanoGPTModelService.generate` runs the model on the configured device and decodes the output tokens.
5. Response returns the completion plus metadata (model name, token counts, latency). Streaming endpoint yields incremental chunks as SSE.

## Concurrency

- An `asyncio.Semaphore` caps concurrent generations (configurable).
- A small in-process queue tracks waiting/running counts for observability.
- Generation runs in an executor to avoid blocking the event loop during PyTorch inference.
