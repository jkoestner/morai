"""
Custom logger with color formatter.

inspired by:
https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
"""
import logging
import sys

from colorama import Back, Fore, Style


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors=None, **kwargs):
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record):
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL
        record.default_color = Fore.WHITE

        return super().format(record)


def setup_logging(name="morai"):
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


def set_log_level(new_level):
    """
    Set the log level.

    Parameters
    ----------
    new_level : int
        the new log level

    """
    options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if new_level not in options:
        raise ValueError(f"Log level must be one of {options}")
    # update the log level for all project loggers
    for logger_name, logger in logging.Logger.manager.loggerDict.items():
        # Check if the logger's name starts with the specified prefix
        if logger_name.startswith("morai"):
            if isinstance(logger, logging.Logger):
                logger.setLevel(new_level)


def test_logger():
    """Test the logger."""
    logger = setup_logging(__name__)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
