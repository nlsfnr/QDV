import heapq
from itertools import tee
from typing import List, Tuple

import numpy as np

from qdv._src.types import ArrayLike, Index, Store

_DEFAULT_DTYPE = np.float32
_DEFAULT_METRIC = "euclidean"


class LinearSearchIndex(Index):
    def __init__(
        self,
        store: Store,
        metric: str = _DEFAULT_METRIC,
    ) -> None:
        self.store = store
        self.metric = metric

    @property
    def dim(self) -> int:
        return self.store.dim

    def query(
        self,
        embeddings: ArrayLike,
        k: int,
    ) -> Tuple[List[List[str]], np.ndarray]:
        all_ids = []
        all_distances = []
        for q in self._validate_embeddings(embeddings):
            kvs = iter(self.store)
            kvs1, kvs2 = tee(kvs)
            ids, vectors = (id for id, _ in kvs1), (v for _, v in kvs2)
            distances = (self._distance(v, q) for v in vectors)
            pairs = zip(ids, distances)
            top_k = heapq.nsmallest(k, pairs, key=lambda x: x[1])
            top_ids, top_distances = zip(*top_k)
            all_ids.append(list(top_ids))
            all_distances.append(top_distances)
        return all_ids, np.asarray(all_distances, dtype=np.float32)

    def _distance(
        self,
        x: np.ndarray,  # D
        y: np.ndarray,  # D
    ) -> float:  # 1
        if self.metric == "euclidean":
            return float(np.linalg.norm(x - y))
        if self.metric == "cosine":
            z = np.linalg.norm(x) * np.linalg.norm(y)
            return float(1.0 - np.dot(x, y.T) / z)
        raise ValueError(
            f"Expected metric to be 'euclidean' or 'cosine', got {self.metric}"
        )

    def _validate_embeddings(
        self,
        embeddings: ArrayLike,
    ) -> np.ndarray:
        if isinstance(embeddings, list) and len(embeddings) == 0:
            return np.empty(shape=(0, self.dim), dtype=_DEFAULT_DTYPE)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=_DEFAULT_DTYPE)
        if not np.issubdtype(embeddings.dtype, np.float32):
            raise TypeError(
                f"Expected embeddings to dtype np.float32, got dtype {embeddings.dtype}"
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
