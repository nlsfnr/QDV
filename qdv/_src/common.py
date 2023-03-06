import importlib
import logging
from types import ModuleType
from typing import Any


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
