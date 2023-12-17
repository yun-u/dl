from __future__ import annotations

from loguru import _logger, logger
from tqdm import tqdm

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <level>{level: >8}</level> "
    "<dim>{file: >26}:{line: <4}</dim> <yellow>[{process}]</yellow> "
    "<level><normal>{message}</normal></level>"
)

__logger: _logger.Logger = None


def get_logger(level: str = "INFO") -> _logger.Logger:
    global __logger

    if __logger is None:
        config = {
            "handlers": [
                {
                    "level": level,
                    # print message with tqdm bar refer https://github.com/Delgan/loguru/issues/135#issuecomment-589025817
                    "sink": lambda message: tqdm.write(message, end=""),
                    "colorize": True,
                    "format": LOG_FORMAT,
                    "backtrace": True,
                },
            ],
            "levels": [
                {
                    "name": "TRACE",
                    "color": "<cyan><bold>",
                },
                {
                    "name": "DEBUG",
                    "color": "<white><bold>",
                },
                {
                    "name": "INFO",
                    "color": "<blue><bold>",
                },
                {
                    "name": "WARNING",
                    "color": "<yellow><bold>",
                },
                {
                    "name": "ERROR",
                    "color": "<red><bold>",
                },
                {
                    "name": "CRITICAL",
                    "color": "<RED><bold>",
                },
            ],
        }

        logger.configure(**config)

        __logger = logger

    return __logger
