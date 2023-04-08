import json
from pathlib import Path

import pytest

from qdv._src.common import MissingDependency

from .openai_lm import OpenAILanguageModel, openai

with open(Path(__file__).parent / "openai_api_dummy_output.json") as fh:
    _DUMMY_RESPONSE = json.load(fh)


_skip_if_openai_not_installed = pytest.mark.skipif(
    isinstance(openai, MissingDependency),
    reason="OpenAI not installed",
)


@pytest.fixture
def openai_lm():
    return OpenAILanguageModel(
        api_key="dummy-api-key",
        create_fn=lambda **_: _DUMMY_RESPONSE,
    )


@_skip_if_openai_not_installed
def test_openai_lm(openai_lm: OpenAILanguageModel):
    openai_lm.call_and_collect("Hello world", max_tokens=10)


@_skip_if_openai_not_installed
def test_openai_lm_prompt_too_long(openai_lm: OpenAILanguageModel):
    with pytest.raises(ValueError, match="contain at most [0-9]+ tokens"):
        openai_lm.call_and_collect("a b c " * 10_000, 1_000)
