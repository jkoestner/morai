"""Custom logger with color formatter."""
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

        return super().format(record)


def setup_logging():
    """
    Set up the logging.

    Returns
    -------
    logger : logging.Logger
        The logger

    """
    formatter = ColoredFormatter(
        "{asctime} | {name} |{color} {levelname:8} {reset}| {message}",
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

    # xact logger
    xact_logger = logging.getLogger("xact")
    xact_logger.setLevel(logging.INFO)

    return xact_logger


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
        if logger_name.startswith("xact"):
            if isinstance(logger, logging.Logger):
                logger.setLevel(new_level)
