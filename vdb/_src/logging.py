import logging


def get_logger() -> logging.Logger:
    """Get the logger for the package.

    Returns:
        The logger.
    """
    return logging.getLogger("VDB")
