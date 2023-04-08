import importlib
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Union

import numpy as np

from qdv._src.types import ArrayLike

_OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


def get_logger() -> logging.Logger:
    """Get the logger for the package.

    Returns:
        The logger.
    """
    return logging.getLogger("QDV")


class MissingDependency(ModuleType):
    """A placeholder for a missing dependency."""

    def __init__(
        self,
        name: str,
        pip_name: str,
    ) -> None:
        def raise_import_error() -> None:
            raise ImportError(
                f"Missing dependency: {name}. "
                f"Please install with `pip install {pip_name}`"
            )

        self.raise_import_error = raise_import_error

    def __getattr__(self, _: str) -> Any:
        return self.raise_import_error()


def try_import(
    import_str: str,
    name: str,
    pip_package: str,
) -> ModuleType:
    try:
        return importlib.import_module(import_str)
    except ImportError:
        return MissingDependency(name, pip_package)


def validate_embeddings(
    embeddings: ArrayLike,
    dim: int,
    dtype: np.dtype = np.float32,  # type: ignore
) -> np.ndarray:
    if isinstance(embeddings, list) and len(embeddings) == 0:
        return np.empty(shape=(0, dim), dtype=dtype)
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=dtype)
    if not np.issubdtype(embeddings.dtype, np.float32):
        raise TypeError(
            f"Expected embeddings to have dtype {dtype}, got dtype {embeddings.dtype}"
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings to be 2-dimensional, got shape {embeddings.shape}"
        )
    if embeddings.shape[1] != dim:
        raise ValueError(
            f"Expected embeddings to have dimension {dim}, got shape {embeddings.shape}"
        )
    return embeddings


def get_openai_key(api_key: Union[None, str, Path]) -> str:
    if api_key is None:
        if _OPENAI_API_KEY_ENV_VAR not in os.environ:
            raise ValueError(
                f"OpenAI API key not provided and environment variable "
                f"{_OPENAI_API_KEY_ENV_VAR} not set."
            )
        return str(os.environ[_OPENAI_API_KEY_ENV_VAR]).strip()
    if isinstance(api_key, Path):
        with api_key.open("r") as f:
            return f.read().strip()
    return api_key
