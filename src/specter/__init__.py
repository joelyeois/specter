import logging

from .jobs import Job as Job, JobDatabase as JobDatabase
from .random_seed import set_seed as seed  # noqa: F401

# Set up package-level logger
logger = logging.getLogger("specter")
logger.addHandler(logging.NullHandler())


def set_verbosity(level: int | str) -> None:
    """
    Set the logging verbosity for specter.

    Parameters
    ----------
    level : int or str
        Logging level, e.g., logging.INFO, logging.DEBUG, or 'INFO', 'DEBUG'.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Configure a handler if none exists to ensure output to stderr/notebook
    if not logger.handlers or isinstance(logger.handlers[0], logging.NullHandler):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: specter: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Remove NullHandler if it was the only one
        if isinstance(logger.handlers[0], logging.NullHandler):
            logger.removeHandler(logger.handlers[0])

    logger.setLevel(level)
