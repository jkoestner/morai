"""Collection of helpers."""

import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

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
DASH_CONFIG_PATH = FILES_PATH / "dashboard_config.yaml"

logger = custom_logger.setup_logging(__name__)


def clean_df(
    data: Union[pd.DataFrame, dict],
    lowercase: bool = True,
    underscore: bool = True,
    update_cat: bool = True,
) -> Union[pd.DataFrame, dict]:
    """
    Clean the DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame, dict]
        The DataFrame or dict to clean.
        The dictionary is set up to be the mapping dictionary.
    lowercase : bool, optional (default=True)
        Whether to lowercase the column names.
    underscore : bool, optional (default=True)
        Whether to replace special characters with underscores in the column names.
    update_cat : bool, optional (default=False)
        Whether to remove unused categories.

    Returns
    -------
    data : Union[pd.DataFrame, dict]
        The cleaned DataFrame or dict.

    """
    # dataframe
    if isinstance(data, pd.DataFrame):
        if lowercase:
            logger.info("lowercasing the column names")
            data.columns = data.columns.str.lower()

        if underscore:
            logger.info(
                "replacing special characters with underscores in the column names"
            )
            data.columns = data.columns.str.replace("[^0-9a-zA-Z_]+", "_", regex=True)

        if update_cat:
            logger.info("removed unused categories and reorder")
            for column in data.select_dtypes(include=["category"]).columns:
                if data[column].isna().any():
                    logger.info(f"{column} has missing values, filling with _NULL_")
                    data[column] = (
                        data[column].cat.add_categories("_NULL_").fillna("_NULL_")
                    )

                data[column] = data[column].cat.remove_unused_categories()
                data[column] = data[column].cat.reorder_categories(
                    sorted(data[column].unique()), ordered=True
                )

        logger.info("update index to int32")
        data.index = data.index.astype("int32")
        logger.info(f"dataFrame shape: {data.shape}")

    # mapping dictionary
    elif isinstance(data, dict):
        for val in data.values():
            if isinstance(val, dict) and "values" in val:
                cleaned_values = {}
                for k, v in val["values"].items():
                    if not isinstance(v, str):
                        cleaned_values[k] = v
                        continue
                    new_v = v
                    if lowercase:
                        new_v = new_v.lower()
                    if underscore:
                        new_v = re.sub(r"[^0-9a-zA-Z_]+", "_", new_v)
                    cleaned_values[k] = new_v
                val["values"] = cleaned_values

    # type not accepted
    else:
        raise TypeError(
            "Input must be either a pandas DataFrame or a dictionary of lists."
        )

    return data


def memory_usage_df(df: pd.DataFrame) -> None:
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


def memory_usage_jupyter() -> pd.DataFrame:
    """
    Calculate the memory usage of objects in the Jupyter notebook.

    Returns
    -------
    object_sizes : pd.DataFrame
        The DataFrame with the object sizes in MB.

    """
    globals = sys.modules["__main__"].__dict__
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


def memory_usage_jupyter_cells(notebook: str, top_n: int = 10) -> pd.DataFrame:
    """
    Calculate the memory usage of cells in Jupyter notebook.

    Parameters
    ----------
    notebook : str
        The path to the Jupyter notebook.
    top_n : int, optional (default=10)
        The number of top cells to return.

    Returns
    -------
    top_cells : pd.DataFrame
        The DataFrame with the top cells in MB.

    """
    # get the notebook path
    nb_path = Path(notebook)
    if not nb_path.exists():
        matches = list(Path(ROOT_PATH / "notebooks").rglob(str(notebook)))
        if len(matches) == 1:
            nb_path = matches[0]
        else:
            raise FileNotFoundError(f"Notebook {notebook} not found.")

    logger.info(f"getting largest cells for {nb_path}")

    # read the notebook and get the cell memory usage
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    rows = []
    for i, cell in enumerate(nb["cells"]):
        raw = json.dumps(cell, ensure_ascii=False)
        size_kb = len(raw.encode("utf-8")) / 1024
        rows.append(
            {
                "cell_idx": i,
                "cell_type": cell.get("cell_type", "unknown"),
                "size_kb": round(size_kb, 1),
                "preview": "".join(cell.get("source", []))[:120].replace("\n", " "),
            }
        )

    top_cells = pd.DataFrame(rows).sort_values("size_kb", ascending=False)

    return top_cells.head(top_n)


def delete_jupyter_objects(objects: list) -> None:
    """
    Delete objects in the Jupyter notebook.

    Parameters
    ----------
    objects : list
        The objects to delete.

    """
    globals = sys.modules["__main__"].__dict__
    for obj in objects:
        del globals[obj]
    gc.collect()
    logger.info(f"deleted `{len(objects)}` objects in the Jupyter notebook")


def test_path(path: str) -> Path:
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


def check_merge(func: Callable) -> Callable:
    """
    Check the merge for a few common issues.

      - check if the merge column already exists.
      - check if there is a one-to-many or many-to-many relationship.

    Parameters
    ----------
    func : function
        the function that merges the DataFrames

    """

    def wrapper(*args, **kwargs) -> Any:
        # check func is 'pd.merge'
        if func.__name__ != "merge":
            raise ValueError("This check only works with the `pd.merge` function")

        # check there aren't going to be column conflicts
        left_df = kwargs.get("left", None)
        right_df = kwargs.get("right", None)
        left_on = kwargs.get("left_on", None)
        right_on = kwargs.get("right_on", None)
        how = kwargs.get("how", None)
        if (
            left_df is None
            or right_df is None
            or left_on is None
            or right_on is None
            or how is None
        ):
            raise ValueError(
                "The left, right, left_on, right_on, and how arguments are required"
            )
        right_columns = set(right_df.columns)
        left_columns = set(left_df.columns) - {left_on}
        common_columns = right_columns.intersection(left_columns)
        if common_columns:
            logger.warning(
                f"There are common columns between the DataFrames: {common_columns}"
            )
            return left_df

        # check if this is "left" merge
        if how != "left":
            logger.warning("This check only works with a `left` merge")
            return left_df

        # check if the right dataframe has multiple values for the right index
        right_unique = right_df[right_on].nunique()
        if right_unique != right_df.shape[0]:
            logger.warning(
                "The right DataFrame has multiple values for the right index"
            )
            return left_df

        # check if the left_on values are in the right_on values
        missing_values = set(left_df[left_on].unique()) - set(
            right_df[right_on].unique()
        )
        if missing_values:
            logger.warning(
                f"Not all left_on values are not in the "
                f"right_on values: {missing_values}"
            )
            return left_df

        # pass the function
        try:
            df = func(*args, **kwargs)
            # check if right columns are nan
            if df[list(right_columns)].isna().any().any():
                logger.warning("There are NaN values in the right columns")
            return df
        except Exception as e:
            raise e

    return wrapper


def _weighted_mean(
    values: Union[list, np.ndarray], weights: Optional[Union[list, np.ndarray]] = None
) -> float:
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


def _convert_object_to_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
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
