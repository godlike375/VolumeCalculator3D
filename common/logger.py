import logging

def setup_logger(level):
    logger = logging.getLogger('default')
    logger.setLevel(level)
    _log_format = f"[%(levelname)s] %(filename)s %(funcName)s(%(lineno)d): %(message)s"
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(logging.Formatter(_log_format))
    logger.addHandler(handler)
    return logger

logger = setup_logger(logging.DEBUG)

