from pathlib import Path

import numpy as np
import pytest

from vdb import LMDBStore

from .brute_force_index import BruteForceIndex


@pytest.fixture
def store(tmpdir: Path):
    store = LMDBStore(Path(tmpdir), 16)
    generator = np.random.default_rng(0)
    embeddings = generator.random((3, 16), dtype=np.float32)
    store.store(["a", "b", "c"], embeddings)
    return store


@pytest.fixture
def index(store: LMDBStore) -> BruteForceIndex:
    return BruteForceIndex(store)


def test_query(index: BruteForceIndex):
    generator = np.random.default_rng(0)
    q = generator.random((1, 16), dtype=np.float32)
    ids, distances = index.query(q, 2)
    assert len(ids) == 1
    assert len(ids[0]) == 2
    assert distances.shape == (1, 2)
