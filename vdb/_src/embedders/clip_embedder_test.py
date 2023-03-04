import numpy as np
import pytest

from vdb._src.common import MissingDependency

from .clip_embedder import CLIPImageEmbedder, CLIPTextEmbedder, transformers

_skip_if_transformers_not_installed = pytest.mark.skipif(
    isinstance(transformers, MissingDependency),
    reason="Transformers not installed",
)


@pytest.fixture
def clip_text_embedder() -> CLIPTextEmbedder:
    return CLIPTextEmbedder()


@pytest.fixture
def clip_image_embedder() -> CLIPImageEmbedder:
    return CLIPImageEmbedder()


@_skip_if_transformers_not_installed
def test_clip_text_embedder(clip_text_embedder: CLIPTextEmbedder) -> None:
    items = ["hello", "world"]
    embeddings = clip_text_embedder(items)
    assert embeddings.shape == (2, 512)


@_skip_if_transformers_not_installed
def test_clip_text_embedder_input_too_long(
    clip_text_embedder: CLIPTextEmbedder,
) -> None:
    items = ["hello " * 100]
    with pytest.raises(ValueError, match="contain at most [0-9]+ tokens"):
        clip_text_embedder(items)


@_skip_if_transformers_not_installed
def test_clip_image_embedder(clip_image_embedder: CLIPImageEmbedder) -> None:
    items = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    embeddings = clip_image_embedder(items)
    assert embeddings.shape == (2, 512)


@_skip_if_transformers_not_installed
def test_clip_image_embedder_input_wrong_ndim(
    clip_image_embedder: CLIPImageEmbedder,
) -> None:
    items = np.zeros((1, 224, 224), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-dimensional"):
        clip_image_embedder(items)
