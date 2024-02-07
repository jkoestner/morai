"""Collection of helpers."""
import logging
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
TESTS_PATH = ROOT_PATH / "tests" / "files"


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


def clean_df(df, lowercase=True, underscore=True):
    """
    Clean the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    lowercase : bool, optional (default=True)
        Whether to lowercase the column names.
    underscore : bool, optional (default=True)
        Whether to replace spaces with underscores in the column names.

    Returns
    -------
    df : pd.DataFrame
        The cleaned DataFrame.
    """
    if lowercase:
        df.columns = df.columns.str.lower()
    if underscore:
        df.columns = df.columns.str.replace(" ", "_")
    return df
