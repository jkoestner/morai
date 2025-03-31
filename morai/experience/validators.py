"""Validating module."""

import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import polars as pl
import yaml

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def get_checks(checks_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the checks from the checks_path.

    Parameters
    ----------
    checks_path : str
        The path to the checks file.

    Returns
    -------
    check_dict : dict
        The checks as a dictionary.

    """
    if checks_path is None:
        checks_path = os.path.join(helpers.ROOT_PATH, "files", "checks", "checks.yaml")
    try:
        with open(checks_path, "r") as file:
            logger.info(f"Loading checks from {checks_path}.")
            logger.info(
                "Ensure checks are reviewed and safe to run as they are "
                "evaluated with eval()."
            )
            check_dict = yaml.safe_load(file)
            check_dict = _replace_newlines_in_dict(check_dict, "\n", " ")
    except FileNotFoundError:
        (f"Config file not found at any of the following paths: {checks_path}")

    total_checks = len(check_dict)
    logger.info(f"Loaded {total_checks} checks.")

    return check_dict


def run_checks(
    lzdf: pl.LazyFrame, check_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run checks defined in check_dict on the dataframe df.

    The checks should be written in a way that they evaluate True if check fails.

    For instance, if you want to make sure that a circle is round, you would write a
    check: `circle != round`, to see how many circles are not round.

    Parameters
    ----------
    lzdf : pl.LazyFrame
        The LazyFrame to validate.
    check_dict : dict, optional (default=None)
        A dictionary of checks to run on the DataFrame.

    Returns
    -------
    check_output : dict
        The results of the checks.

    """
    # check to make sure lzdf is a LazyFrame
    if not isinstance(lzdf, pl.LazyFrame):
        raise TypeError(f"lzdf must be a LazyFrame, not {type(lzdf)}")

    # get the checks
    if check_dict is None:
        check_dict = get_checks()
    else:
        if not isinstance(check_dict, dict):
            raise TypeError(f"check_dict must be a dictionary, not {type(check_dict)}")
        logger.info("Using check_dict passed to function.")

    # get the total rows
    total_rows = lzdf.select(pl.len()).collect().item()

    # run the checks
    total_checks = len(check_dict)
    logger.info(f"Running {total_checks} checks")
    check_output = {}
    for idx, (check_name, check_logic_str) in enumerate(check_dict.items(), start=1):
        # add a running total of the checks in logger
        print(f"Running check {idx} of {total_checks}...", end="\r")
        try:
            check_logic = eval(check_logic_str)
            result = lzdf.filter(check_logic).select(pl.len()).collect().item()
            check_output[check_name] = result
        except Exception as e:
            logger.error(f"Error evaluating check '{check_name}': {e}")
            check_output[check_name] = None
    print(f"Completed checks {idx} of {total_checks}...")
    logger.info(f"Completed {total_checks} checks")

    # create a dataframe of the check results
    check_output = pd.DataFrame(
        list(check_output.items()), columns=["checks", "result"]
    )
    check_output["percent"] = (
        check_output["result"] / total_rows if total_rows else None
    )

    return check_output


def view_single_check(
    lzdf: pl.LazyFrame, check_dict: Dict[str, Any], check_name: str, limit: int = 10
) -> pd.DataFrame:
    """
    View the results of a specific check.

    Parameters
    ----------
    lzdf : pl.LazyFrame
        The LazyFrame to validate.
    check_dict : dict
        A dictionary of checks to run on the DataFrame.
    check_name : str
        The name of the check to view.
    limit : int, optional (default=10)
        The number of rows to return.

    Returns
    -------
    single_check : pd.DataFrame
        The results of the checks.

    """
    if check_name not in check_dict:
        raise ValueError(f"Check '{check_name}' not found in check_dict")
    check_logic_str = check_dict[check_name]
    check_logic = eval(check_logic_str)
    single_check = lzdf.filter(check_logic).limit(limit).collect()
    return single_check


def _replace_newlines_in_dict(
    d: Dict[str, Union[str, Dict[str, Any]]], old: str, new: str
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Recursively replace newline characters in all string values of a dictionary.

    This is useful for yaml files that have multiline strings.

    Parameters
    ----------
    d : dict
        The dictionary to update.
    old : str
        The string to replace.
    new : str
        The string to replace with.

    Returns
    -------
    d : dict
        The updated dictionary.

    """
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = v.replace(old, new)
        elif isinstance(v, dict):
            d[k] = _replace_newlines_in_dict(v, old, new)
        elif isinstance(v, list):
            d[k] = [
                _replace_newlines_in_dict(item, old, new)
                if isinstance(item, dict)
                else item
                for item in v
            ]
    return d
