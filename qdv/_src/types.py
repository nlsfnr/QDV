from __future__ import annotations

from typing import (
    Generic,
    Iterable,
    Iterator,
    Protocol,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np


class _SupportsArray(Sized, Iterable, Protocol):
    def __array__(self) -> np.ndarray:
        ...


ArrayLike = Union[
    _SupportsArray,
    Sequence["ArrayLike"],
    Sequence[float],
]


EmbeddingStoreT = TypeVar("EmbeddingStoreT", bound="EmbeddingStore")


class EmbeddingStore(Protocol):
    def ids(self) -> Iterator[str]:
        """Iterate over the ids of the stored embeddings."""
        ...

    @property
    def dim(self) -> int:
        """The dimensionality of the stored embeddings."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the stored embeddings."""
        ...

    def store(
        self: EmbeddingStoreT,
        ids: Sequence[str],
        embeddings: ArrayLike,
    ) -> EmbeddingStoreT:
        """Store the embeddings with the given ids. Overwrites any existing
        embeddings with the same ids."""
        ...

    def retrieve(
        self,
        ids: Sequence[str],
    ) -> np.ndarray:
        """Retrieve the embeddings with the given ids."""
        ...

    def delete(
        self: EmbeddingStoreT,
        ids: Sequence[str],
    ) -> EmbeddingStoreT:
        """Delete the embeddings with the given ids."""
        ...

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over the stored embeddings."""
        ...

    def __len__(self) -> int:
        """The number of stored embeddings."""
        ...


class Index(Protocol):
    @property
    def dim(self) -> int:
        """The dimensionality of the stored embeddings."""
        ...

    def query(
        self,
        embeddings: ArrayLike,
        k: int,
    ) -> Tuple[Sequence[Sequence[str]], np.ndarray]:
        """Query the index for the k nearest neighbors of the given embeddings."""
        ...


ItemT = TypeVar("ItemT", contravariant=True)


class Embedder(Protocol, Generic[ItemT]):
    @property
    def dim(self) -> int:
        """The dimensionality of the embeddings."""
        ...

    def __call__(self, items: Sequence[ItemT]) -> np.ndarray:
        """Embed the given items."""
        ...
