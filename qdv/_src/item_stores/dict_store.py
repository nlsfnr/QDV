from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generic, Iterable, Iterator, Sequence, Tuple, TypeVar, Union

from qdv._src.types import ItemStore

JsonValue = Union[
    float, int, str, bool, None, Dict[str, "JsonValue"], Iterable["JsonValue"]
]
JsonItemT = TypeVar("JsonItemT", bound=JsonValue)


class JsonStore(ItemStore, Generic[JsonItemT]):
    def __init__(
        self,
        data: Dict[str, JsonItemT],
    ) -> None:
        self.data = data

    @classmethod
    def from_ids_and_items(
        cls,
        ids: Iterable[str],
        items: Iterable[JsonItemT],
    ) -> JsonStore:
        return cls(dict(zip(ids, items)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.data, f)

    @classmethod
    def load(cls, path: Path) -> JsonStore[JsonItemT]:
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dict, got {type(data)}")
        if not all(isinstance(k, str) for k in data.keys()):
            raise ValueError(f"Expected all keys to be strings, got {data.keys()}")
        return cls(data)

    @property
    def ids(self) -> Iterable[str]:
        return self.data.keys()

    def store(self, ids: Iterable[str], items: Sequence[JsonItemT]) -> None:
        self.data.update(zip(ids, items))

    def retrieve(self, ids: Sequence[str]) -> Sequence[JsonItemT]:
        return tuple(self.data[id] for id in ids)

    def delete(self, ids: Sequence[str]) -> None:
        for id in ids:
            del self.data[id]

    def __iter__(self) -> Iterator[Tuple[str, JsonItemT]]:
        return iter(self.data.items())

    def __len__(self) -> int:
        return len(self.data)
