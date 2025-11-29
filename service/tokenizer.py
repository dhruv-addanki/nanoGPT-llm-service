"""Tokenizer abstraction for the inference service."""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - dependency issue is explicit
    raise RuntimeError(
        "tiktoken is required for the service tokenizer. Install it via requirements.txt."
    ) from exc


class GPT2BPETokenizer:
    """Minimal GPT-2 BPE tokenizer wrapper."""

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        """Encode text into token ids."""
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: Sequence[int] | Iterable[int]) -> str:
        """Decode token ids back into text."""
        return self.encoding.decode(list(tokens))


__all__ = ["GPT2BPETokenizer"]
