import logging

G_FORMATTER = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")


def setup_custom_logger(name, level="INFO"):
    logger = logging.getLogger(name)

    handler = logging.StreamHandler()
    handler.setFormatter(G_FORMATTER)
    logger.addHandler(handler)

    logger.setLevel(getattr(logging, level))
    return logger


def add_filehandler(logger, filename):
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(G_FORMATTER)
    logger.root.addHandler(file_handler)


def remove_logger_tf_handler(logger):
    logger.root.handlers = [item for item in logger.root.handlers
                            if isinstance(item, logging.FileHandler)]
