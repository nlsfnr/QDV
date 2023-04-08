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


class LanguageModel(Protocol):
    def __call__(
        self,
        prompt: str,
        max_tokens: int,
    ) -> Iterable[str]:
        """Generate text from the given prompt and stream the generated text."""
        ...

    def call_and_collect(
        self,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Generate text from the given prompt and collect the output into a str."""
        return "".join(self(prompt, max_tokens))


ItemTCov = TypeVar("ItemTCov")


class ItemStore(Protocol, Generic[ItemTCov]):
    @property
    def ids(self) -> Iterable[str]:
        """Iterate over the ids of the stored items."""
        ...

    def store(self, ids: Sequence[str], items: Sequence[ItemTCov]) -> None:
        """Store the items with the given ids. Overwrites any existing
        items with the same ids."""
        ...

    def retrieve(self, ids: Sequence[str]) -> Sequence[ItemTCov]:
        """Retrieve the items with the given ids."""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """Delete the items with the given ids."""
        ...

    def __iter__(self) -> Iterator[Tuple[str, ItemTCov]]:
        """Iterate over the stored items."""
        ...

    def __len__(self) -> int:
        """The number of stored items."""
        ...
