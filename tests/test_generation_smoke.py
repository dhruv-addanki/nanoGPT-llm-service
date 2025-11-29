import pytest

from service.config import ServiceSettings
from service.models import NanoGPTModelService
from service.tokenizer import GPT2BPETokenizer


@pytest.mark.asyncio
async def test_generation_smoke():
    settings = ServiceSettings(
        mock_model=True,
        max_new_tokens_default=4,
        max_new_tokens_limit=6,
        max_context_tokens=64,
    )
    tokenizer = GPT2BPETokenizer()
    service = NanoGPTModelService(settings=settings, tokenizer=tokenizer)

    result = await service.generate("Hello nanoGPT", max_new_tokens=3)
    assert result.completion
    assert result.tokens_out > 0
