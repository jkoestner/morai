"""Collection of helpers."""
import logging
import os
from pathlib import Path

import numpy as np

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
TESTS_PATH = ROOT_PATH / "tests" / "files"

logging.config.fileConfig(
    os.path.join(ROOT_PATH, "logging.ini"),
)
logger = logging.getLogger(__name__)


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
        logger.info("lowercasing the column names")
        df.columns = df.columns.str.lower()
    if underscore:
        logger.info("replacing spaces with underscores in the column names")
        df.columns = df.columns.str.replace(" ", "_")
    return df


def _weighted_mean(values, weights):
    """
    Calculate the weighted mean.

    Parameters
    ----------
    values : list, numpy array
        The values to use.
    weights : list, numpy array, or None
        The weights to use.

    Returns
    -------
    weighted_mean : float
        The weighted mean

    """
    if weights is None:
        return np.mean(values)
    if np.sum(weights) == 0:
        return np.nan
    else:
        return np.average(values, weights=weights)
