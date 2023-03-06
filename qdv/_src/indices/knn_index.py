from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np

from qdv._src.common import get_logger, try_import
from qdv._src.types import ArrayLike, Index, Store

if TYPE_CHECKING:
    import pynndescent
else:
    pynndescent = try_import("pynndescent", "PyNNDescent", "pynndescent")

logger = get_logger()


class KNNIndex(Index):
    def __init__(
        self,
        index: pynndescent.NNDescent,
        ids: Iterable[str],
    ) -> None:
        self._index = index
        self.ids = list(ids)
        self._index_to_id = {i: id for i, id in enumerate(self.ids)}

    @property
    def dim(self) -> int:
        return self._index.dim

    @classmethod
    def load(cls, path: Path) -> KNNIndex:
        """Load a KNNIndex from a directory.

        Args:
            path: The directory to load from.

        Returns:
            The loaded KNNIndex.
        """
        if not path.exists():
            raise FileNotFoundError(f"Does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        index = _load_index(path)
        ids = _load_ids(path)
        logger.info(f"Loaded index from {path}")
        return cls(index, ids)

    @classmethod
    def from_data(
        cls,
        ids: Iterable[str],
        embeddings: ArrayLike,
    ) -> KNNIndex:
        """Create a new index from data.

        Args:
            data: The data to index with shape (x, d) where n is the number of
                samples and d is the dimensionality of the data.

        Returns:
            A new index.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = pynndescent.NNDescent(np.asarray(embeddings))
        logger.info(f"Created new index from {len(embeddings)} samples")
        return cls(index, ids)

    @classmethod
    def from_store(cls, store: Store) -> KNNIndex:
        """Create a new index from a store.

        Args:
            store: The store to index.

        Returns:
            A new index.
        """
        embeddings = np.empty((len(store), store.dim), dtype=store.dtype)
        ids = []
        index = -1
        for index, (id, embedding) in enumerate(store):
            embeddings[index] = embedding
            ids.append(id)
        if not index == len(store) - 1:
            raise ValueError(
                f"Indexing failed, expected {len(store)} samples, "
                f"but got {index + 1}"
            )
        return cls.from_data(ids, embeddings)

    def save(self, path: Path) -> KNNIndex:
        """Save the index to disk.

        Args:
            path: The path to save the index to.

        Returns:
            The index.
        """
        if not path.exists():
            path.mkdir(parents=True)
        else:
            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {path}")
        _save_index(self._index, path)
        _save_ids(self.ids, path)
        logger.info(f"Saved index to {path}")
        return self

    def query(
        self,
        embeddings: ArrayLike,
        k: int,
    ) -> Tuple[List[List[str]], np.ndarray]:
        """Query the index.

        Args:
            embeddings: The embeddings to query.
            k: The number of neighbors to return.

        Returns:
            A tuple of the ids of the neighbors and the distances to the neighbors.
        """
        embeddings = np.asarray(embeddings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indices, distances = self._index.query(embeddings, k=k)
        ids = [[self._index_to_id[i] for i in row] for row in indices]
        return ids, distances


def _load_index(path: Path) -> pynndescent.NNDescent:
    with open(path / "index.pkl", "rb") as fh:
        index = pickle.load(fh)
    assert isinstance(index, pynndescent.NNDescent)
    return index


def _load_ids(path: Path) -> List[str]:
    with open(path / "ids.json", "r") as fh:
        ids = json.load(fh)
    assert isinstance(ids, list)
    assert all(isinstance(id, str) for id in ids)
    return ids


def _save_index(index: pynndescent.NNDescent, path: Path) -> None:
    with open(path / "index.pkl", "wb") as fh:
        pickle.dump(index, fh)


def _save_ids(ids: Iterable[str], path: Path) -> None:
    with open(path / "ids.json", "w") as fh:
        json.dump(list(ids), fh)
