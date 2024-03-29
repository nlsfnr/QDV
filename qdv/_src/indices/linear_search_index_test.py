from pathlib import Path

import numpy as np
import pytest

from qdv import LMDBEmbeddingStore

from .linear_search_index import LinearSearchIndex


@pytest.fixture
def store(tmpdir: Path):
    store = LMDBEmbeddingStore(Path(tmpdir), 16)
    generator = np.random.default_rng(0)
    embeddings = generator.random((3, 16), dtype=np.float32)
    store.store(["a", "b", "c"], embeddings)
    return store


@pytest.fixture
def index(store: LMDBEmbeddingStore) -> LinearSearchIndex:
    return LinearSearchIndex(store)


def test_query(index: LinearSearchIndex):
    generator = np.random.default_rng(0)
    q = generator.random((1, 16), dtype=np.float32)
    ids, distances = index.query(q, 2)
    assert len(ids) == 1
    assert len(ids[0]) == 2
    assert distances.shape == (1, 2)
