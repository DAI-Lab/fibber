import logging


def setup_custom_logger(name, filename=None, level="INFO"):
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        logger.addHandler(logging.FileHandler("log.txt"))

    logger.setLevel(getattr(logging, level))
    return logger
