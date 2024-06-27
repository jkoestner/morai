"""Collection of helpers."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from morai.utils import custom_logger

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
TESTS_PATH = ROOT_PATH / "tests" / "files"
FILES_PATH = (
    Path(os.getenv("MORAI_FILES_PATH"))
    if os.getenv("MORAI_FILES_PATH")
    else ROOT_PATH / "files"
)
CONFIG_PATH = FILES_PATH / "dashboard_config.yaml"

logger = custom_logger.setup_logging(__name__)


def clean_df(df, lowercase=True, underscore=True, update_cat=True):
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
    update_cat : bool, optional (default=False)
        Whether to remove unused categories.

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

    if update_cat:
        logger.info("removed unused categories and reorder")
        for column in df.select_dtypes(include=["category"]).columns:
            if df[column].isna().any():
                logger.info(f"{column} has missing values, filling with _NULL_")
                df[column] = df[column].cat.add_categories("_NULL_").fillna("_NULL_")

            df[column] = df[column].cat.remove_unused_categories()
            df[column] = df[column].cat.reorder_categories(
                sorted(df[column].unique()), ordered=True
            )

    logger.info("update index to int32")
    df.index = df.index.astype("int32")
    logger.info(f"dataFrame shape: {df.shape}")

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


def test_path(path):
    """
    Test the path with a few different options and return if it exists.

    Parameters
    ----------
    path : str
        The path to test.

    Returns
    -------
    path : pathlib.Path
        The path as a pathlib.Path.

    """
    paths_to_try = [
        path,
        os.path.join(FILES_PATH / "dataset", path),
        os.path.join(FILES_PATH / "result", path),
    ]
    for path_to_try in paths_to_try:
        try:
            with open(path_to_try):
                break
        except FileNotFoundError:
            continue
    else:
        paths_str = ", ".join(map(str, paths_to_try))
        raise FileNotFoundError(
            f"File not found at any of the following paths: {paths_str}"
        )
    path = path_to_try

    return path


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
    if weights is None or len(weights) == 0:
        return values.mean()

    if weights.sum() == 0:
        logger.warning("The sum of the weights is 0, returning NaN")
        return np.nan
    else:
        return np.average(values, weights=weights)


def _convert_object_to_category(df, column):
    """
    Convert the column to a category if it is an object.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.
    column : str
        The column to convert.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the column converted to a category.

    """
    if df[column].dtype == "object":
        df[column] = df[column].astype("category")
    return df
