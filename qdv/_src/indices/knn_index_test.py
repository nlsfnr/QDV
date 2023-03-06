from pathlib import Path

import numpy as np
import pytest

from qdv import LMDBStore

from .knn_index import KNNIndex


@pytest.fixture
def store(tmpdir: Path):
    store = LMDBStore(Path(tmpdir), 16)
    generator = np.random.default_rng(0)
    embeddings = generator.random((3, 16), dtype=np.float32)
    store.store(["a", "b", "c"], embeddings)
    return store


@pytest.fixture
def index(store: LMDBStore) -> KNNIndex:
    return KNNIndex.from_store(store)


def test_from_data() -> None:
    generator = np.random.default_rng(0)
    embeddings = generator.random((3, 16), dtype=np.float32)
    ids = ["a", "b", "c"]
    index = KNNIndex.from_data(ids, embeddings)
    assert index.dim == 16
    embeddings = generator.random((1, 16), dtype=np.float32)
    ids_, distances = index.query(embeddings, 3)
    assert distances.shape == (1, 3)
    assert len(ids_) == 1


def test_from_store(store: LMDBStore) -> None:
    index = KNNIndex.from_store(store)
    assert index.dim == 16
    generator = np.random.default_rng(0)
    embeddings = generator.random((1, 16), dtype=np.float32)
    ids, distances = index.query(embeddings, 3)
    assert distances.shape == (1, 3)
    assert len(ids) == 1


def test_save_and_load(tmpdir: Path, index: KNNIndex) -> None:
    index.save(Path(tmpdir))
    index2 = KNNIndex.load(Path(tmpdir))
    assert index2.dim == 16
    embeddings = np.random.default_rng(0).random((1, 16), dtype=np.float32)
    ids1, distances1 = index.query(embeddings, 3)
    ids2, distances2 = index2.query(embeddings, 3)
    assert ids1 == ids2
    assert (distances1 == distances2).all()
