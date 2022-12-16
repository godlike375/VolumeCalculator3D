import logging
import os
from datetime import datetime
from pathlib import Path
from datetime import timedelta
from time import time

FOLDER = 'logs'

def setup_logger(level):
    logger = logging.getLogger('default')
    logger.setLevel(level)
    _log_format = f"[%(levelname)s] %(filename)s %(funcName)s(%(lineno)d): %(message)s"
    Path.mkdir(Path(FOLDER), exist_ok=True)
    logname = f'logs/log_{datetime.now().strftime("%d,%m,%Y_%H;%M;%S")}.txt'
    handler = logging.FileHandler(logname, mode='w')
    handler.setFormatter(logging.Formatter(_log_format))
    logger.addHandler(handler)
    return logger

logger = setup_logger(logging.DEBUG)


def cleanup_old_logs(dir=None):
    folder = dir or Path(FOLDER)
    if not Path.exists(folder):
        return
    for log in Path.iterdir(folder):
        if time() - log.stat().st_mtime > timedelta(days=30).total_seconds():
            os.remove(log)

