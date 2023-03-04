import logging
from typing import Any

import numpy as np


def get_logger() -> logging.Logger:
    """Get the logger for the package.

    Returns:
        The logger.
    """
    return logging.getLogger("VDB")


class MissingDependency:
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
