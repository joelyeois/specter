import logging

# Set up package-level logger
logger = logging.getLogger("cryosim")
logger.addHandler(logging.NullHandler())


def set_verbosity(level):
    """
    Set the logging verbosity for cryosim.

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
        formatter = logging.Formatter("%(levelname)s: cryosim: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Remove NullHandler if it was the only one
        if isinstance(logger.handlers[0], logging.NullHandler):
            logger.removeHandler(logger.handlers[0])

    logger.setLevel(level)
