"""Mortality Table Builder."""

import itertools

import numpy as np
import pandas as pd
import polars as pl
import pymort

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

# creating presets
presets = {
    "vbt15": {
        "table_list": [3224, 3234, 3252, 3262],
        "extra_dims": {"sex": ["F", "M"], "smoker_status": ["NS", "S"]},
        "juv_list": [3273, 3273, 3274, 3274],
        "extend": True,
    }
}


class MortTable:
    """A mortality table class that can be used to build a 1-d mortality table."""

    def __init__(
        self,
        preset=None,
    ):
        """
        Initialize the Table class.

        Parameters
        ----------
        preset : str, optional (default=None)
            A preset to use for the table. The preset can be "vbt15".

        """
        self.table_list = presets.get(preset, {}).get("table_list", None)
        self.extra_dims = presets.get(preset, {}).get("extra_dims", None)
        self.juv_list = presets.get(preset, {}).get("juv_list", None)
        self.extend = presets.get(preset, {}).get("extend", False)
        self.select_period = None
        self.df = None
        self.max_age = 121

        if preset:
            self.build_table(
                table_list=self.table_list,
                extra_dims=self.extra_dims,
                juv_list=self.juv_list,
                extend=self.extend,
            )

    def build_table(self, table_list, extra_dims=None, juv_list=None, extend=False):
        """
        Build a 1-d mortality dataframe from a list of tables.

        The 1-d table will have issue age and duration go to 121.

        The table_list should be a list of mortality table id numbers from mort.soa.org.
        The list of tables should match the combinations of the extra dimensions
        and will be in the order of lexicographic combinations so that the
        leftmost elements vary the slowest.

        Example
        -------
        extra_dims={"sex":["F","M"], "smoker_status":["NS","S"]}
        table_list=[3224, 3234, 3252, 3262]
        juv_list=[3273, 3273, 3274, 3274]
        mort_table = build_table(table_list=table_list,
                                 extra_dims=extra_dims,
                                 juv_list=juv_list,
                                 extend=True)

        Parameters
        ----------
        table_list : list
            A list of mortality table id numbers from mort.soa.org.
        extra_dims : dict, optional (default=None)
            A dictionary of extra dimensions to add to the table.
            e.g. extra_dims={"sex":["F","M"], "smoker_status":["NS","S"]}
        juv_list : list, optional (default=None)
            A list of juvenile select tables to merge into the table. The list should
            should have the same length as table_list and will only use issue ages 0-17.
        extend : bool, optional (default=False)
            Whether to extend the table to fill in missing values.

        Returns
        -------
        mort_table : DataFrame
            The 1-d mortality table.

        """
        max_age = self.max_age
        select_period = self.select_period

        extra_dims = extra_dims or {}
        extra_dims_keys = list(extra_dims.keys())
        combinations = list(itertools.product(*extra_dims.values()))
        juv_list = juv_list or [None] * len(table_list)

        if len(table_list) != len(combinations):
            raise ValueError(
                f"the tables length: {len(table_list)}, does not match the "
                f"combinations length: {len(combinations)}"
            )

        # mortality grid
        dims = {
            "issue_age": range(max_age + 1),
            "duration": range(1, max_age + 2),
        } | extra_dims
        mort_table = create_grid(dims=dims, max_age=max_age)
        if "attained_age" not in mort_table.columns:
            mort_table["attained_age"] = (
                mort_table["issue_age"] + mort_table["duration"] - 1
            )
            mort_table = mort_table[mort_table["attained_age"] <= max_age]

        for table, combo, juv_table_id in zip(table_list, combinations, juv_list):
            extra_dims_list = list(zip(extra_dims.keys(), combo))
            # get soa table
            soa_xml = self.get_soa_xml(table_id=table)
            # determine if select and ultimate
            num_tables = len(soa_xml.Tables)

            # select and ultimate
            if num_tables == 2:
                # select table
                select_table, select_period, _ = self._process_soa_table(
                    soa_xml=soa_xml, table_index=0, is_select=True
                )
                mort_table = self._merge_tables(
                    merge_table=mort_table,
                    source_table=select_table,
                    merge_keys=["issue_age", "duration", *extra_dims_keys],
                    column_rename="vals_sel",
                    extra_dims_list=extra_dims_list,
                )

                if juv_table_id:
                    juv_xml = self.get_soa_xml(table_id=juv_table_id)
                    juv_table, juv_select_period, _ = self._process_soa_table(
                        soa_xml=juv_xml, table_index=0, is_select=True
                    )
                    juv_table = juv_table[juv_table["issue_age"] <= 17]
                    if len(juv_xml.Tables) == 1:
                        logger.warning(
                            f"Juvenile table: {juv_table_id} has only one table "
                            f"and is not select and ultimate. Skipping."
                        )
                    elif juv_select_period != select_period:
                        logger.warning(
                            f"Juvenile table: {juv_table_id} has a different select "
                            f"period than the main table: {table}. Skipping."
                        )
                    else:
                        mort_table = self._merge_tables(
                            merge_table=mort_table,
                            source_table=juv_table,
                            merge_keys=["issue_age", "duration", *extra_dims_keys],
                            column_rename="vals_juv",
                            extra_dims_list=extra_dims_list,
                        )

                # ult table
                ult_table, _, min_age = self._process_soa_table(
                    soa_xml=soa_xml, table_index=1, is_select=True
                )
                mort_table = self._merge_tables(
                    merge_table=mort_table,
                    source_table=ult_table,
                    merge_keys=["attained_age", *extra_dims_keys],
                    column_rename="vals_ult",
                    extra_dims_list=extra_dims_list,
                )
            # aggregate or ultimate
            elif num_tables == 1:
                # ult table
                ult_table, _, _ = self._process_soa_table(
                    soa_xml=soa_xml, table_index=0, is_select=False
                )
                mort_table = self._merge_tables(
                    merge_table=mort_table,
                    source_table=ult_table,
                    merge_keys=["attained_age", *extra_dims_keys],
                    column_rename="vals_ult",
                    extra_dims_list=extra_dims_list,
                )
            else:
                raise ValueError(
                    f"Can't handle the number of tables: {num_tables} "
                    f"for table: {table}"
                )

        if extend:
            fill_keys = ["issue_age", "attained_age"]
            for key in fill_keys:
                missing = len(mort_table[mort_table["vals"].isnull()])
                grouped = mort_table.groupby([key, *extra_dims_keys], group_keys=False)
                mort_table["vals"] = grouped["vals"].apply(
                    lambda x: x.astype(float).ffill().bfill()
                )

        logger.info(f"Created table that has the following dims: {dims}")
        logger.info(f"Table has {len(mort_table)} cells.")
        logger.info(f"combinations: {combinations}")
        logger.info(f"tables: {table_list}")
        if juv_table_id:
            logger.info(f"juveniles: {juv_list}")
        if extend:
            logger.info(f"extend: True, filled in {missing} missing values.")

        self.df = mort_table
        self.table_list = table_list
        self.extra_dims = extra_dims
        self.juv_list = juv_list
        self.extend = extend
        self.select_period = select_period

        return mort_table

    def get_soa_xml(self, table_id):
        """
        Get the soa xml object.

        This is a wrapper for pymort.MortXML.from_id.

        Parameters
        ----------
        table_id : int
            The table id.

        Returns
        -------
        soa_xml : pymort.MortXML
            an xml object from pymort.

        """
        soa_xml = pymort.MortXML.from_id(table_id)
        return soa_xml

    def _merge_tables(
        self, merge_table, source_table, merge_keys, column_rename, extra_dims_list=None
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
        extra_dims_list : list, optional (default=None)
            A list of tuples of extra dimensions to merge.

        Returns
        -------
        merge_table : pd.DataFrame
            The merged table.

        """
        if extra_dims_list is None:
            extra_dims_list = []
        source_table = source_table.rename(columns={"vals": column_rename})
        for dim_name, dim_value in extra_dims_list:
            source_table[dim_name] = dim_value
        merge_table = merge_table.merge(source_table, on=merge_keys, how="left")
        merge_table["vals"] = (
            merge_table["vals"].astype(float).fillna(merge_table[column_rename])
        )
        merge_table = merge_table.drop(columns=column_rename)
        return merge_table

    def _process_soa_table(self, soa_xml, table_index, is_select):
        """
        Gather the metadata from the soa table.

        Parameters
        ----------
        soa_xml : xml object
            xml objects that comes from mort.soa.org.
        table_index : int
            The table index.
        is_select : bool
            Whether the table is a select and ultimate table.

        Returns
        -------
        soa_table : pymort.MortXML
            an xml object from pymort.
        select_period : int
            The select period.
        min_age : int
            The minimum age.

        """
        soa_table = soa_xml.Tables[table_index].Values.reset_index()
        soa_table.columns = soa_table.columns.str.lower()

        if table_index == 0 and is_select:
            soa_table = soa_table.rename(columns={"age": "issue_age"})
            select_period = (
                soa_xml.Tables[table_index].MetaData.AxisDefs[1].MaxScaleValue
            )
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


def create_grid(dims=None, mapping=None, max_age=121, max_grid_size=5000000):
    """
    Create a grid from the dimensions.

    Parameters
    ----------
    dims : dict
        The dimensions where it is structured as {dim_name: dim_values}.
    mapping : dict
        The mapping where it is structured as {dim_name: {"values": dim_values}}.
    max_age : int, optional (default=121)
        The maximum age.
    max_grid_size : int, optional (default=5,000,000)
        The maximum grid size.

    Returns
    -------
    mort_grid : pd.DataFrame
        The grid.

    """
    if (not dims and not mapping) or (dims and mapping):
        raise ValueError("Either dims or mapping must be provided.")
    if mapping:
        dims = {col: list(val["values"].keys()) for col, val in mapping.items()}
    dimensions = list(dims.values())

    # check the grid size before creating it
    grid_size = 1
    for dimension in dimensions:
        grid_size *= len(dimension)
    logger.info(f"Grid size: {grid_size} combinations.")
    if grid_size > max_grid_size:
        raise ValueError(
            f"Grid size too large: {grid_size} combinations. "
            f"Maximum allowed is {max_grid_size}."
        )

    grid = list(itertools.product(*dimensions))
    column_names = list(dims.keys())
    logger.info(f"Creating grid with dimensions: {column_names}")

    # create mort grid (polars is much quicker)
    mort_grid = pl.DataFrame(grid, schema=column_names)
    mort_grid = mort_grid.sort(by=mort_grid.columns)

    # convert objects to categorical
    mort_grid = mort_grid.with_columns(
        [
            pl.col(name).cast(pl.Categorical)
            for name in mort_grid.columns
            if mort_grid[name].dtype == pl.Utf8
        ]
    )
    mort_grid = mort_grid.to_pandas()
    mort_grid = check_aa_ia_dur_cols(mort_grid, max_age=max_age)

    mort_grid["vals"] = np.nan
    return mort_grid


def compare_tables(table_1, table_2, value_col="vals"):
    """
    Compare two tables.

    Table 1 is used as the source of the keys to compare on.

    Parameters
    ----------
    table_1 : pd.DataFrame
        The first table.
    table_2 : pd.DataFrame
        The second table.
    value_col : str, optional
        The column to compare.

    Returns
    -------
    compare_df : pd.DataFrame
        DataFrame of the comparison with the ratio of the table_1/table_2 values.

    """
    if type(table_1) != pd.DataFrame or type(table_2) != pd.DataFrame:
        raise ValueError("Both tables must be pandas DataFrames.")
    if value_col not in table_1.columns or value_col not in table_2.columns:
        raise ValueError(f"Value column: {value_col} not in both tables.")

    # get the common keys to compare on
    common_keys = list(set(table_1.columns) & set(table_2.columns) - {value_col})
    if not common_keys:
        raise ValueError("No common keys between the two tables.")
    logger.info(f"Comparing tables on keys: {common_keys}")

    # get the unique keys dict for each table
    unique_keys = {}
    for i, table in enumerate([table_1, table_2]):
        table_name = f"table_{i+1}"
        unique_keys[table_name] = list(
            set(table.columns) - set(common_keys) - {value_col}
        )
        if unique_keys[table_name]:
            unique_keys[table_name] = {
                key: len(table[key].unique()) for key in unique_keys[table_name]
            }
            if table_name == "table_1":
                logger.info(f"{table_name} has extra keys: {unique_keys[table_name]}.")
            # aggregate table_2 if it has extra keys
            elif table_name == "table_2":
                table_2 = table_2.groupby(common_keys, as_index=False).agg(
                    {value_col: "mean"}
                )
                logger.info(
                    f"{table_name} has extra keys: {unique_keys[table_name]}. "
                    f"Calculated mean for '{value_col}' column."
                )

    # compare
    compare_df = table_1.merge(
        table_2,
        on=common_keys,
        suffixes=("_1", "_2"),
    )
    compare_df = compare_df.rename(columns={"vals_1": "table_1", "vals_2": "table_2"})
    compare_df["ratio"] = compare_df["table_1"] / compare_df["table_2"]

    return compare_df


def check_aa_ia_dur_cols(df, max_age=121):
    """
    Check attained age, issue age, and duration columns.

    attained_age = issue_age + duration - 1

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    max_age : int, optional (default=121)
        The maximum age.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the columns checked.

    """
    initial_rows = len(df)

    # check for invalid attained age / duration / issue age combos
    if all(col in df.columns for col in ["attained_age", "issue_age", "duration"]):
        df = df[df["attained_age"] >= df["duration"] - 1]
        df = df[df["attained_age"] >= df["issue_age"]]
    elif all(col in df.columns for col in ["attained_age", "duration"]):
        df = df[df["attained_age"] >= df["duration"] - 1]
    elif all(col in df.columns for col in ["attained_age", "issue_age"]):
        df = df[df["attained_age"] >= df["issue_age"]]

    # cap the max attained age
    if "attained_age" in df.columns:
        df = df[df["attained_age"] <= max_age]
    elif all(col in df.columns for col in ["issue_age", "duration"]):
        df = df[(df["issue_age"] + df["duration"] - 1) <= max_age]

    removed_rows = initial_rows - len(df)
    if removed_rows:
        logger.info(
            f"Removed '{removed_rows}' rows where attained_age, issue_age, "
            f"or duration was invalid."
        )

    return df


def output_table(df, name="table.csv"):
    """
    Output the table to a csv file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    name : str, optional (default="table.csv")
        The name of the file.

    """
    path = helpers.FILES_PATH / "dataset" / "tables" / name
    df.to_csv(path, index=False)
    logger.info(f"Output table to: {path}")
