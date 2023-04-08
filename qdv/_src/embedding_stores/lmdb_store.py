from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qdv._src.common import get_logger, try_import
from qdv._src.types import ArrayLike, EmbeddingStore

if TYPE_CHECKING:
    import lmdb
else:
    lmdb = try_import("lmdb", "LMDB", "lmdb")

logger = get_logger()

_DEFAULT_DTYPE = np.float32
_DEFAULT_MAP_SIZE = 2**40  # 1TB


class LMDBEmbeddingStore(EmbeddingStore):
    """A store backed by an LMDB database."""

    def __init__(
        self,
        path: Path,
        embedding_dim: int,
        map_size: int = _DEFAULT_MAP_SIZE,
        dtype: npt.DTypeLike = _DEFAULT_DTYPE,
    ) -> None:
        """Initialize the store.

        Args:
            path: The path to the LMDB database directory.
            map_size: The maximum size of the memory map.
        """
        if not path.exists():
            logger.info(f"Creating new LMDB database at {path}")
            path.mkdir(parents=True)
        else:
            if not path.is_dir():
                raise NotADirectoryError(f"Expected path to be a directory: {path}")
            logger.info(f"Opening existing LMDB database at {path}")
        self.path = path
        self.environment = lmdb.Environment(
            str(path), readahead=False, meminit=False, subdir=True, map_size=map_size
        )
        self._dtype = np.dtype(dtype)
        self._embedding_dim = embedding_dim

    @property
    def dim(self) -> int:
        """The dimension of the embeddings."""
        return self._embedding_dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def ids(self) -> Iterator[str]:
        """The ids of the embeddings in the store."""
        with self.environment.begin(write=False) as txn:
            yield from (id.decode() for id, _ in txn.cursor())

    def store(
        self,
        ids: Sequence[str],
        embeddings: ArrayLike,
    ) -> LMDBEmbeddingStore:
        """EmbeddingStore embeddings in the store.

        Args:
            ids: The ids of the embeddings.
            embeddings: The embeddings to store.

        Returns:
            The store.
        """
        ids, embeddings = self._validate_ids_and_embeddings(ids, embeddings)
        with self.environment.begin(write=True) as txn:
            for id, embedding in zip(ids, embeddings):
                if not isinstance(id, str):
                    raise TypeError(f"Expected id to be a str, got {id}")
                key: bytes = id.encode()
                value: bytes = np.asarray(embedding).tobytes()
                txn.put(key, value, overwrite=True)
        return self

    def retrieve(
        self,
        ids: Sequence[str],
    ) -> np.ndarray:
        """Retrieve embeddings from the store.

        Args:
            ids: The ids of the embeddings to retrieve.

        Returns:
            The embeddings as a 2-dimensional numpy array.
        """
        ids = self._validate_ids(ids)
        if len(ids) == 0:
            return np.empty(shape=(0, 0), dtype=self.dtype)
        with self.environment.begin(write=False) as txn:
            embeddings = []
            for id in ids:
                result = txn.get(id.encode())
                if result is None:
                    raise KeyError(f"Key not found: {id}")
                embeddings.append(np.frombuffer(result, dtype=self.dtype))
        return np.asarray(embeddings)

    def delete(
        self,
        ids: Sequence[str],
    ) -> LMDBEmbeddingStore:
        """Delete embeddings from the store.

        Args:
            ids: The ids of the embeddings to delete.

        Returns:
            The store.
        """
        ids = self._validate_ids(ids)
        with self.environment.begin(write=True) as txn:
            for id in ids:
                success = txn.delete(id.encode())
                if not success:
                    raise KeyError(f"Key not found: {id}")
        return self

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over the ids and embeddings in the store."""
        with self.environment.begin(write=False) as txn:
            for id, value in txn.cursor():
                yield id.decode(), np.frombuffer(value, dtype=self.dtype)

    def __len__(self) -> int:
        """The number of embeddings in the store."""
        length = self.environment.stat()["entries"]
        assert isinstance(length, int)
        return length

    def _validate_ids(
        self,
        ids: Sequence[str],
    ) -> List[str]:
        ids = list(ids)
        if not all(isinstance(id, str) for id in ids):
            raise TypeError(f"Expected ids to be a sequence of str, got {ids}")
        return ids

    def _validate_embeddings(
        self,
        embeddings: ArrayLike,
    ) -> np.ndarray:
        if isinstance(embeddings, list) and len(embeddings) == 0:
            return np.empty(shape=(0, self.dim), dtype=self.dtype)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=self.dtype)
        if not np.issubdtype(embeddings.dtype, self.dtype):
            raise TypeError(
                f"Expected embeddings with dtype {self.dtype}, got {embeddings.dtype}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings to be 2-dimensional, got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings to have dimension {self.dim}, "
                f"got shape {embeddings.shape}"
            )
        return embeddings

    def _validate_ids_and_embeddings(
        self,
        ids: Sequence[str],
        embeddings: ArrayLike,
    ) -> Tuple[List[str], np.ndarray]:
        ids = self._validate_ids(ids)
        embeddings = self._validate_embeddings(embeddings)
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Expected ids and embeddings to have the same length, got {len(ids)} "
                f"and {embeddings.shape[0]}"
            )
        return ids, embeddings
