"""Collection of helpers."""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def memory_usage_df(df):
    """
    Calculate the memory usage of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the memory usage.

    """
    memory_usage_per_column = df.memory_usage(deep=True)
    most_memory_column = memory_usage_per_column.idxmax()
    total_memory_usage = memory_usage_per_column.sum() / 1048576
    print(f"Total memory usage: {total_memory_usage} mb")
    print(f"Column consuming the most memory: {most_memory_column}")
    print(f"Memory usage per column:\n{memory_usage_per_column}")


def memory_usage_jupyter(globals):
    """
    Calculate the memory usage of objects in the Jupyter notebook.

    Parameters
    ----------
    globals : dict
        The globals() dictionary. This needs to be passed in from the Jupyter notebook
        using "globals()".

    Returns
    -------
    object_sizes : pd.DataFrame
        The DataFrame with the object sizes in MB.

    """
    ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]

    variables = [
        x
        for x in globals
        if not x.startswith("_") and x not in sys.modules and x not in ipython_vars
    ]

    object_sizes = pd.DataFrame(
        [(x, sys.getsizeof(globals[x]) / (1024**2)) for x in variables],
        columns=["object", "size_mb"],
    )
    object_sizes = object_sizes.sort_values(by="size_mb", ascending=False).reset_index(
        drop=True
    )

    return object_sizes


def _weighted_mean(values, weights=None):
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
    if np.sum(weights) == 0:
        logger.warning("The sum of the weights is 0, returning NaN")
        return np.nan
    else:
        return np.average(values, weights=weights)
