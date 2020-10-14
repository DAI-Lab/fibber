import logging

G_FORMATTER = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")


def setup_custom_logger(name, level="INFO"):
    """Get a logger."""
    logger = logging.getLogger(name)

    handler = logging.StreamHandler()
    handler.setFormatter(G_FORMATTER)
    logger.addHandler(handler)

    logger.setLevel(getattr(logging, level))
    return logger


def add_file_handler(logger, filename):
    """Add file handler to a logger."""
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(G_FORMATTER)
    logger.root.addHandler(file_handler)


def remove_logger_tf_handler(logger):
    """Remove all handlers except file handler.

    This function can clean up the mess caused by tensorflow_hub.
    """
    logger.root.handlers = [item for item in logger.root.handlers
                            if isinstance(item, logging.FileHandler)]
