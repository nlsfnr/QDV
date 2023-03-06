from pathlib import Path

import numpy as np
import pytest

from qdv import LMDBStore
from qdv._src.common import MissingDependency

from .knn_index import KNNIndex, pynndescent

_skip_if_pynndescent_not_installed = pytest.mark.skipif(
    isinstance(pynndescent, MissingDependency),
    reason="PyNNDescent not installed",
)


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


@_skip_if_pynndescent_not_installed
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


@_skip_if_pynndescent_not_installed
def test_from_store(store: LMDBStore) -> None:
    index = KNNIndex.from_store(store)
    assert index.dim == 16
    generator = np.random.default_rng(0)
    embeddings = generator.random((1, 16), dtype=np.float32)
    ids, distances = index.query(embeddings, 3)
    assert distances.shape == (1, 3)
    assert len(ids) == 1


@_skip_if_pynndescent_not_installed
def test_save_and_load(tmpdir: Path, index: KNNIndex) -> None:
    index.save(Path(tmpdir))
    index2 = KNNIndex.load(Path(tmpdir))
    assert index2.dim == 16
    embeddings = np.random.default_rng(0).random((1, 16), dtype=np.float32)
    ids1, distances1 = index.query(embeddings, 3)
    ids2, distances2 = index2.query(embeddings, 3)
    assert ids1 == ids2
    assert (distances1 == distances2).all()


@_skip_if_pynndescent_not_installed
def test_query() -> None:
    generator = np.random.default_rng(0)
    data = generator.random((10, 16), dtype=np.float32)
    index = KNNIndex.from_data(map(str, range(len(data))), data)
    ids, distances = index.query([data[4]], 3)
    assert distances.shape == (1, 3)
    assert len(ids) == 1
    assert len(ids[0]) == 3
    assert ids[0][0] == "4"
    assert distances[0][0] == 0.0
