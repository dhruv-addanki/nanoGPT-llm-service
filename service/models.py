"""Model loading and text generation utilities for the service."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from service.config import ServiceSettings, settings as default_settings
from service.tokenizer import GPT2BPETokenizer

logger = logging.getLogger("service.models")


@dataclass
class GenerationResult:
    prompt: str
    completion: str
    model_name: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class MockModel(torch.nn.Module):
    """Tiny deterministic model for testing without real weights."""

    def __init__(self):
        super().__init__()
        self.config = GPTConfig(block_size=128, vocab_size=256, n_layer=2, n_head=2, n_embd=64)

    def forward(self, idx, targets=None):  # pragma: no cover - not used directly
        raise RuntimeError("MockModel forward should not be called directly.")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        device = idx.device
        for _ in range(max_new_tokens):
            last_token = idx[:, -1:]
            # simple predictable pattern to keep tests stable
            next_token = (last_token + 1) % self.config.vocab_size
            idx = torch.cat((idx, next_token.to(device)), dim=1)
        return idx


class NanoGPTModelService:
    """High-level model wrapper for inference and streaming generation."""

    def __init__(
        self,
        settings: ServiceSettings | None = None,
        tokenizer: Optional[GPT2BPETokenizer] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ):
        self.settings = settings or default_settings
        self.tokenizer = tokenizer or GPT2BPETokenizer()
        self.device = torch.device(self.settings.device)
        self.model = self._load_model()
        self.model.eval()
        self.semaphore = semaphore or asyncio.Semaphore(self.settings.max_concurrent_generations)
        logger.info(
            "Model initialized: model=%s device=%s mock=%s",
            self.model.__class__.__name__,
            self.device,
            self.settings.mock_model,
        )

    # ---------------------------
    # Public API
    # ---------------------------
    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> GenerationResult:
        """Generate text for a prompt."""
        max_tokens = self._resolve_max_new_tokens(max_new_tokens)
        prompt_tokens = self.tokenizer.encode(prompt)
        self._validate_prompt(prompt_tokens)
        tokens_in = len(prompt_tokens)

        async with self.semaphore:
            start = time.perf_counter()

            def _run_generation() -> Sequence[int]:
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    generated = self._generate_tokens(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                return generated[0].tolist()

            token_ids = await asyncio.to_thread(_run_generation)
            tokens_out = len(token_ids) - tokens_in
            completion = self.tokenizer.decode(token_ids[tokens_in:])
            latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "request completed tokens_in=%s tokens_out=%s latency_ms=%.2f device=%s",
            tokens_in,
            tokens_out,
            latency_ms,
            self.device,
        )
        return GenerationResult(
            prompt=prompt,
            completion=completion,
            model_name=self._model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

    async def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream incremental chunks as text."""
        max_tokens = self._resolve_max_new_tokens(max_new_tokens)
        prompt_tokens = self.tokenizer.encode(prompt)
        self._validate_prompt(prompt_tokens)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async with self.semaphore:
            def _worker() -> None:
                try:
                    for chunk in self._generate_streaming_chunks(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    ):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                thread.join()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _load_model(self) -> torch.nn.Module:
        if self.settings.mock_model:
            return MockModel().to(self.device)

        if self.settings.resolved_model_path:
            checkpoint = torch.load(self.settings.resolved_model_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint["model_args"])
            model = GPT(gptconf)
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for key, value in list(state_dict.items()):
                if key.startswith(unwanted_prefix):
                    state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
            model.load_state_dict(state_dict)
        else:
            model = GPT.from_pretrained(self.settings.model_type, dict(dropout=0.0))
        return model.to(self.device)

    def _resolve_max_new_tokens(self, max_new_tokens: Optional[int]) -> int:
        value = max_new_tokens or self.settings.max_new_tokens_default
        return min(value, self.settings.max_new_tokens_limit)

    def _validate_prompt(self, prompt_tokens: Sequence[int]) -> None:
        if len(prompt_tokens) > self.settings.max_context_tokens:
            raise ValueError(
                f"Prompt too long: {len(prompt_tokens)} tokens, limit is {self.settings.max_context_tokens}"
            )

    def _generate_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size :]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            logits = self._apply_sampling(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def _generate_streaming_chunks(
        self,
        *,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> Iterable[str]:
        idx = input_ids
        new_tokens: List[int] = []
        chunk_size = max(1, self.settings.stream_chunk_size)
        for step in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size :]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            logits = self._apply_sampling(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            new_tokens.append(int(idx_next.item()))

            emit_ready = (step + 1) % chunk_size == 0
            if emit_ready:
                yield self.tokenizer.decode(new_tokens[-chunk_size:])
        if new_tokens and len(new_tokens) % chunk_size != 0:
            remainder = len(new_tokens) % chunk_size
            yield self.tokenizer.decode(new_tokens[-remainder:])

    @staticmethod
    def _apply_sampling(
        logits: torch.Tensor, *, top_k: Optional[int], top_p: Optional[float]
    ) -> torch.Tensor:
        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            for batch_idx in range(logits.size(0)):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                logits[batch_idx, indices_to_remove] = -float("inf")

        return logits

    @property
    def _model_name(self) -> str:
        if self.settings.mock_model:
            return "mock-gpt"
        if self.settings.resolved_model_path:
            return self.settings.resolved_model_path
        return self.settings.model_type
