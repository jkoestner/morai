"""
Custom logger with color formatter.

inspired by:
https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
"""

import logging
import sys
from typing import Any, Callable, Optional

from colorama import Back, Fore, Style


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[dict] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL
        record.default_color = Fore.WHITE

        return super().format(record)


def setup_logging(name: str = "morai") -> logging.Logger:
    """
    Set up the logging.

    Returns
    -------
    logger : logging.Logger
        The logger

    """
    formatter = ColoredFormatter(
        "{default_color} {asctime} {reset}|{default_color} {name} {reset}|"
        "{color} {levelname:8} {reset}|{color} {message} {reset}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        colors={
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
        },
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # root logger
    root_logger = logging.getLogger()
    root_logger.handlers[:] = []  # remove existing handlers
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)

    # morai logger
    morai_logger = logging.getLogger(name)
    morai_logger.setLevel(logging.INFO)

    return morai_logger


def set_log_level(new_level: str, module_prefix: str = "morai") -> None:
    """
    Set the log level.

    Parameters
    ----------
    new_level : str
        the new log level
    module_prefix : str
        the module logger prefix to set the log level for

    """
    options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if new_level not in options:
        raise ValueError(f"Log level must be one of {options}")
    numeric_level = logging.getLevelName(new_level)
    # update the log level for all project loggers
    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        # Check if the logger's name starts with the specified prefix
        if logger_name.startswith(module_prefix):
            if isinstance(logger, logging.Logger):
                logger.setLevel(numeric_level)


def get_log_level(module_prefix: str = "morai") -> str:
    """
    Get the log level.

    Parameters
    ----------
    module_prefix : str
        the module logger prefix to get the log level for

    Returns
    -------
    str
        the log level

    """
    log_level = None
    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        # Check if the logger's name starts with the specified prefix
        if logger_name.startswith(module_prefix):
            if isinstance(logger, logging.Logger):
                log_level = logging.getLevelName(logger.getEffectiveLevel())
                break
    return log_level


def suppress_logs(func: Callable) -> Callable:
    """
    Suppress the log of a function.

    Parameters
    ----------
    func : function
        the function to suppress the log for

    """

    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        current_level = logger.level
        logger.setLevel(logging.CRITICAL)
        try:
            return func(*args, **kwargs)
        finally:
            logger.setLevel(current_level)

    return wrapper


def test_logger() -> None:
    """Test the logger."""
    logger = setup_logging(__name__)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
