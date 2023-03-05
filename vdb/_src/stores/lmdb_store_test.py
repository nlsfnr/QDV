from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture as Benchmark  # type: ignore

from vdb._src.common import MissingDependency
from vdb._src.types import ArrayLike

from .lmdb_store import LMDBStore, lmdb

_skip_if_lmdb_not_installed = pytest.mark.skipif(
    isinstance(lmdb, MissingDependency),
    reason="LMDB not installed",
)


@pytest.fixture
def store(tmpdir: Path) -> LMDBStore:
    return LMDBStore(path=Path(tmpdir), embedding_dim=3)


_VALID_ID_EMBEDDING_PAIRS = [
    ([], []),
    (["a", "b", "c"], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    (["a", "b", "c"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    (["a"], [[1, 2, 3]]),
    ([], np.empty((0, 3), dtype=np.float32)),
    (
        ["a", "b", "c"],
        np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
        ),
    ),
    (
        np.asarray(["a", "b", "c"]),
        np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
        ),
    ),
]


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize("ids, embeddings", _VALID_ID_EMBEDDING_PAIRS)
def test_store(
    store: LMDBStore,
    ids: Sequence[str],
    embeddings: ArrayLike,
) -> None:
    store.store(ids=ids, embeddings=embeddings)


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize(
    "ids, embeddings, error_type, match",
    (
        (
            [1, 2, 3],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            TypeError,
            "ids.*sequence of str",
        ),
        (
            [["a"]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            TypeError,
            "ids.*sequence of str",
        ),
        (
            np.asarray([1]),
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            TypeError,
            "ids.*sequence of str",
        ),
        (
            ["a"],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ValueError,
            "same length",
        ),
        (
            ["a", "b"],
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            ValueError,
            "same length",
        ),
        (
            ["a"],
            np.asarray([[1, 2, 3]], dtype=np.int32),
            TypeError,
            "embedding.*dtype np.float32",
        ),
    ),
)
def test_store_invalid_inputs(
    store: LMDBStore,
    ids: Sequence[str],
    embeddings: ArrayLike,
    error_type: type,
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        store.store(ids=ids, embeddings=embeddings)


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize("ids, embeddings", _VALID_ID_EMBEDDING_PAIRS)
def test_store_retrieve(
    store: LMDBStore,
    ids: Sequence[str],
    embeddings: ArrayLike,
) -> None:
    store.store(ids=ids, embeddings=embeddings)
    retrieved = store.retrieve(ids=ids)
    assert isinstance(retrieved, np.ndarray)
    assert retrieved.ndim == 2
    assert retrieved.shape[0] == 0 or (retrieved == embeddings).all()


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize(
    "query_ids, ids, embeddings, error_type, match",
    [
        (["a"], [], [], KeyError, "not found: a"),
        (["a"], ["b"], [[1.0, 2.0, 3.0]], KeyError, "not found: a"),
        (["a"], ["A"], [[1.0, 2.0, 3.0]], KeyError, "not found: a"),
        ([1], ["a"], [[1.0, 2.0, 3.0]], TypeError, "ids.*sequence of str"),
    ],
)
def test_store_retrieve_invalid_ids(
    store: LMDBStore,
    query_ids: Sequence[str],
    ids: Sequence[str],
    embeddings: ArrayLike,
    error_type: type,
    match: str,
) -> None:
    store.store(ids=ids, embeddings=embeddings)
    with pytest.raises(error_type, match=match):
        store.retrieve(ids=query_ids)


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize("ids, embeddings", _VALID_ID_EMBEDDING_PAIRS)
def test_store_ids(store: LMDBStore, ids: Sequence[str], embeddings: ArrayLike) -> None:
    store.store(ids=ids, embeddings=embeddings)
    assert set(store.ids()) == set(list(ids))


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize(
    "delete_ids, ids, embeddings",
    [
        (["a"], ["a"], [[1.0, 2.0, 3.0]]),
        (["a"], ["a", "b"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        (["a", "b"], ["a", "b"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        (
            ["a", "b"],
            ["a", "b", "c"],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ),
    ],
)
def test_store_delete(
    store: LMDBStore,
    delete_ids: Sequence[str],
    ids: Sequence[str],
    embeddings: ArrayLike,
) -> None:
    store.store(ids=ids, embeddings=embeddings)
    store.delete(ids=delete_ids)
    assert set(store.ids()) == set(ids) - set(delete_ids)
    with pytest.raises(KeyError, match="not found: a"):
        store.retrieve(ids=delete_ids)


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize("ids, embeddings", _VALID_ID_EMBEDDING_PAIRS)
def test_store_iter(
    store: LMDBStore,
    ids: Sequence[str],
    embeddings: ArrayLike,
) -> None:
    store.store(ids=ids, embeddings=embeddings)
    visited_ids = []
    for id, embedding in store:
        assert id in ids
        assert (embedding == store.retrieve(ids=[id])[0]).all()
        visited_ids.append(id)
    assert len(visited_ids) == len(ids) == len(list(store.ids()))
    assert set(visited_ids) == set(ids) == set(store.ids())


@_skip_if_lmdb_not_installed
@pytest.mark.parametrize("ids, embeddings", _VALID_ID_EMBEDDING_PAIRS)
def test_store_len(
    store: LMDBStore,
    ids: Sequence[str],
    embeddings: ArrayLike,
) -> None:
    store.store(ids=ids, embeddings=embeddings)
    assert len(store) == len(ids)


@_skip_if_lmdb_not_installed
def test_lmdb_retrieve_bench(tmpdir: Path, benchmark: Benchmark) -> None:
    store = LMDBStore(path=Path(tmpdir), embedding_dim=512)
    ids = [str(i) for i in range(10_000)]
    generator = np.random.default_rng(0)
    embeddings = generator.random((10_000, 512)).astype(np.float32)
    store.store(ids=ids, embeddings=embeddings)
    all_query_ids = [
        [str(i) for i in generator.integers(0, 10_000 - 1, 100)] for _ in range(100)
    ]

    def fn() -> None:
        for query_ids in all_query_ids:
            store.retrieve(ids=query_ids)

    benchmark(fn)
