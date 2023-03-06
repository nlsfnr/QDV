from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from qdv._src.common import MissingDependency, get_logger
from qdv._src.types import ArrayLike, Index, Store

if TYPE_CHECKING:
    from pynndescent import NNDescent  # type: ignore
else:
    try:
        from pynndescent import NNDescent  # type: ignore
    except ModuleNotFoundError:
        NNDescent = MissingDependency("PyNNDescent", "pynndescent")  # type: ignore

logger = get_logger()


class KNNIndex(Index):
    def __init__(
        self,
        index: NNDescent,
        id_to_index: Dict[str, int],
    ) -> None:
        self._index = index
        self._id_to_index = id_to_index
        self._index_to_id = {v: k for k, v in id_to_index.items()}

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
        id_to_index = _load_id_to_index(path)
        logger.info(f"Loaded index from {path}")
        return cls(index, id_to_index)

    @classmethod
    def from_data(
        cls,
        ids: List[str],
        embeddings: ArrayLike,
    ) -> KNNIndex:
        """Create a new index from data.

        Args:
            data: The data to index with shape (x, d) where n is the number of
                samples and d is the dimensionality of the data.

        Returns:
            A new index.
        """
        index = NNDescent(np.asarray(embeddings))
        id_to_index = {id_: i for i, id_ in enumerate(ids)}
        logger.info(f"Created new index from {len(embeddings)} samples")
        return cls(index, id_to_index)

    @classmethod
    def from_store(cls, store: Store) -> KNNIndex:
        """Create a new index from a store.

        Args:
            store: The store to index.

        Returns:
            A new index.
        """
        embeddings = np.empty((len(store), store.dim))
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
        _save_id_to_index(self._id_to_index, path)
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
        indices, distances = self._index.query(embeddings, k=k)
        ids = [[self._index_to_id[i] for i in row] for row in indices]
        return ids, distances


def _load_index(path: Path) -> NNDescent:
    with open(path / "index.pkl", "rb") as fh:
        index = pickle.load(fh)
    assert isinstance(index, NNDescent)
    return index


def _load_id_to_index(path: Path) -> Dict[str, int]:
    with open(path / "id_to_index.json", "r") as fh:
        id_to_index = json.load(fh)
    assert isinstance(id_to_index, dict)
    assert all(isinstance(k, str) for k in id_to_index)
    assert all(isinstance(v, int) for v in id_to_index.values())
    return id_to_index


def _save_index(index: NNDescent, path: Path) -> None:
    with open(path / "index.pkl", "wb") as fh:
        pickle.dump(index, fh)


def _save_id_to_index(id_to_index: Dict[str, int], path: Path) -> None:
    with open(path / "id_to_index.json", "w") as fh:
        json.dump(id_to_index, fh)
