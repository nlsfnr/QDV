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


StoreT = TypeVar("StoreT", bound="Store")


class Store(Protocol):
    def ids(self) -> Iterator[str]:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...

    def store(
        self: StoreT,
        ids: Sequence[str],
        embeddings: ArrayLike,
    ) -> StoreT:
        ...

    def retrieve(
        self,
        ids: Sequence[str],
    ) -> np.ndarray:
        ...

    def delete(
        self: StoreT,
        ids: Sequence[str],
    ) -> StoreT:
        ...

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        ...

    def __len__(self) -> int:
        ...


class Index(Protocol):
    @property
    def dim(self) -> int:
        ...

    def query(
        self,
        embeddings: ArrayLike,
        k: int,
    ) -> Tuple[Sequence[Sequence[str]], np.ndarray]:
        ...


ItemT = TypeVar("ItemT", contravariant=True)


class Embedder(Protocol, Generic[ItemT]):
    @property
    def dim(self) -> int:
        ...

    def __call__(self, items: Sequence[ItemT]) -> np.ndarray:
        ...
