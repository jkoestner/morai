"""Mortality Table Builder."""

import itertools

import pandas as pd
import pymort

from xact.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def build_table(table_list, extra_dims=None, extend=False):
    """
    Build a 1-d mortality dataframe from a list of tables.

    The table_list should be a list of mortality table id numbers from mort.soa.org.
    The list of tables should match the combinations of the extra dimensions and will
    be in the order of lexicographic combinations so that the leftmost elements
    vary the slowest.

    Example
    -------
    extra_dims={"gender":["F","M"], "underwriting":["NS","S"]}
    table_list=[3224, 3234, 3252, 3262]
    mort_table = build_table(table_list=table_list, extra_dims=extra_dims, extend=True)

    Parameters
    ----------
    table_list : list
        A list of mortality table id numbers from mort.soa.org.
    extra_dims : dict, optional (default=None)
        A dictionary of extra dimensions to add to the table.
        e.g. extra_dims={"gender":["F","M"], "underwriting":["NS","S"]}
    extend : bool, optional (default=False)
        Whether to extend the table to fill in missing values.

    Returns
    -------
    mort_table : pd.DataFrame
        The mortality table.

    """
    max_age = 121
    extra_dims = extra_dims or {}
    extra_dims_keys = list(extra_dims.keys())
    combinations = list(itertools.product(*extra_dims.values()))

    if len(table_list) != len(combinations):
        print(
            f"error: the tables {len(table_list)} do not match the "
            f"combinations {len(combinations)}"
        )

    # mortality grid
    dims = {
        "issue_age": range(max_age + 1),
        "duration": range(1, max_age + 2),
    } | extra_dims
    mort_table = _create_grid(dims, max_age)

    for table, combo in zip(table_list, combinations):
        extra_dims_zip = zip(extra_dims.keys(), combo)
        # get soa tables
        soa_xml = pymort.MortXML(table)
        # determine if select and ultimate
        is_select = len(soa_xml.Tables) == 2

        if is_select:
            # select table
            select_table, select_period, _ = _process_soa_table(soa_xml, 0, is_select)
            mort_table = _merge_tables(
                mort_table,
                select_table,
                ["issue_age", "duration", *extra_dims_keys],
                "vals_sel",
                extra_dims_zip,
            )
            # ult table
            ult_table, _, min_age = _process_soa_table(soa_xml, 1, is_select)
            # ult grid
            dims = {"attained_age": range(min_age, max_age + 1)} | extra_dims
            ult_grid = _create_grid(dims, max_age)
            ult_table = _merge_tables(ult_grid, ult_table, ["attained_age"], "vals_ult")
            mort_table = _merge_tables(
                mort_table,
                ult_table,
                ["attained_age", *extra_dims_keys],
                "vals_ult",
                extra_dims_zip,
            )
        else:
            # ult table
            ult_table, _, _ = _process_soa_table(soa_xml, 0, False)
            mort_table = _merge_tables(
                mort_table,
                ult_table,
                ["attained_age", *extra_dims_keys],
                "vals_ult",
                extra_dims_zip,
            )

    if extend:
        fill_key = ["issue_age"]
        mort_table["vals"] = mort_table.groupby(
            fill_key + extra_dims_keys, group_keys=False
        )["vals"].apply(lambda x: x.ffill().bfill())
        fill_key = ["attained_age"]
        mort_table["vals"] = mort_table.groupby(
            fill_key + extra_dims_keys, group_keys=False
        )["vals"].apply(lambda x: x.ffill().bfill())

    return mort_table


def _merge_tables(
    merge_table, source_table, merge_keys, column_rename, extra_dims_zip=None
):
    """
    Merge the source table into the merge table.

    Parameters
    ----------
    merge_table : pd.DataFrame
        The table to merge into.
    source_table : pd.DataFrame
        The table to merge from.
    merge_keys : list
        The keys to merge on.
    column_rename : str
        The column to rename.
    extra_dims_zip : list, optional (default=None)
        A list of tuples of extra dimensions to merge.

    Returns
    -------
    merge_table : pd.DataFrame
        The merged table.

    """
    if extra_dims_zip is None:
        extra_dims_zip = []
    source_table = source_table.rename(columns={"vals": column_rename})
    for dim_name, dim_value in extra_dims_zip:
        source_table[dim_name] = dim_value
    merge_table = merge_table.merge(source_table, on=merge_keys, how="left")
    merge_table["vals"] = merge_table["vals"].fillna(merge_table[column_rename])
    merge_table = merge_table.drop(columns=column_rename)
    return merge_table


def _process_soa_table(soa_xml, table_index, is_select):
    """
    Gather the metadata from the soa table.

    Parameters
    ----------
    soa_xml : pymort.MortXML
        The soa table.
    table_index : int
        The table index.
    is_select : bool
        Whether the table is a select and ultimate table.

    Returns
    -------
    soa_table : pd.DataFrame
        The table.
    select_period : int
        The select period.
    min_age : int
        The minimum age.

    """
    soa_table = soa_xml.Tables[table_index].Values.reset_index()
    soa_table.columns = soa_table.columns.str.lower()

    if table_index == 0 and is_select:
        soa_table = soa_table.rename(columns={"age": "issue_age"})
        select_period = soa_xml.Tables[table_index].MetaData.AxisDefs[1].MaxScaleValue
        min_age = None
    elif table_index == 1 and is_select:
        soa_table = soa_table.rename(columns={"age": "attained_age"})
        select_period = None
        min_age = soa_xml.Tables[1].MetaData.AxisDefs[0].MinScaleValue
    else:
        soa_table = soa_table.rename(columns={"age": "attained_age"})
        select_period = None
        min_age = None

    return soa_table, select_period, min_age


def _create_grid(dims, max_age=121):
    """
    Create a grid from the dimensions.

    Parameters
    ----------
    dims : dict
        The dimensions.
    max_age : int, optional (default=121)
        The maximum age.

    Returns
    -------
    mort_grid : pd.DataFrame
        The grid.

    """
    dimensions = list(dims.values())
    grid = list(itertools.product(*dimensions))
    column_names = list(dims.keys())
    mort_grid = pd.DataFrame(grid, columns=column_names)
    if "attained_age" not in mort_grid.columns:
        mort_grid["attained_age"] = mort_grid["issue_age"] + mort_grid["duration"] - 1
        mort_grid = mort_grid[mort_grid["attained_age"] <= max_age]
    mort_grid["vals"] = None
    return mort_grid
