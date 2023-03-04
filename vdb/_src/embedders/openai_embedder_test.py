import json
from pathlib import Path

import numpy as np
import pytest

from vdb._src.common import MissingDependency

from .openai_embedder import OpenAIEmbedder, openai

with open(Path(__file__).parent / "openai_api_dummy_output.json") as fh:
    _DUMMY_RESPONSE = json.load(fh)


_skip_if_openai_not_installed = pytest.mark.skipif(
    isinstance(openai, MissingDependency),
    reason="OpenAI not installed",
)


@pytest.fixture
def embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(
        api_key="dummy-api-key",
        model_name="dummy-model",
        create_fn=lambda **_: _DUMMY_RESPONSE,
    )


@_skip_if_openai_not_installed
def test_openai_embedder(embedder: OpenAIEmbedder) -> None:
    embeddings = embedder(["Content ignored, output hard-coded"])
    assert embeddings.shape == (1, 1536)
    assert embeddings.dtype == np.float32


@_skip_if_openai_not_installed
def test_openai_embedder_input_too_long(embedder: OpenAIEmbedder) -> None:
    with pytest.raises(ValueError, match="contain at most \d+ tokens"):
        embedder(["a b c " * 10000])
