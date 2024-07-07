"""Mortality Table Builder."""

import copy
import itertools

import numpy as np
import pandas as pd
import polars as pl
import pymort
import yaml

from morai.forecast import models, preprocessors
from morai.utils import custom_logger, helpers
from morai.utils.custom_logger import suppress_logs

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
        mort_table = suppress_logs(create_grid)(dims=dims, max_age=max_age)
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
            missing = len(mort_table[mort_table["vals"].isnull()])
            for key in fill_keys:
                grouped = mort_table.groupby([key, *extra_dims_keys], group_keys=False)
                mort_table["vals"] = grouped["vals"].apply(
                    lambda x: x.astype(float).ffill().bfill()
                )

        logger.info(f"Created table that has the following dims: {dims}")
        logger.info(f"Table has {len(mort_table)} cells.")
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


def generate_table(
    model,
    preprocess_mapping,
    preprocess_feature_dict,
    preprocess_params,
    grid=None,
    mult_features=None,
):
    """
    Generate a 1-d mortality table based on model predictions.

    Parameters
    ----------
    model : model
        The model to use for generating the table
    preprocess_mapping : dict
        The mapping of the features to predict on with corresponding values
    preprocess_feature_dict : dict
        The dictionary of features to use for the model that were preprocessed
    preprocess_params : dict
        The parameters that were used for preprocessing
    grid : pd.DataFrame, optional
        The grid to use for the table
    mult_features : list, optional
        The features to use for the multiplier table

    Returns
    -------
    tuple
        rate_table : pd.DataFrame
            The 1-d mortality rate_table
        mult_table : pd.DataFrame
            The multiplier table

    """
    # initialize the variables
    logger.info(f"generating table for model {type(model).__name__}")
    models.ModelWrapper(model).check_predict()
    rate_mapping = preprocess_mapping
    rate_feature_dict = {
        k: v
        for k, v in preprocess_feature_dict.items()
        if k not in ["target", "weight"]
    }
    mult_table = None
    # remove the 'add_constant' parameter, due to already in mapping
    preprocess_params["add_constant"] = False

    # create separate mult_mapping and rate_mapping
    if mult_features:
        logger.warning(
            "the multipliers may not match the predictions from the model "
            "and will be the average prediction for the feature"
        )

        def _remove_mult_from_rate_mapping(mapping, mult_features):
            mapping = copy.deepcopy(mapping)
            for key, sub_dict in mapping.items():
                if key in mult_features and "values" in sub_dict:
                    first_key = next(iter(sub_dict["values"]))
                    sub_dict["values"] = {first_key: sub_dict["values"][first_key]}
            return mapping

        mult_mapping = {k: v for k, v in rate_mapping.items() if k in mult_features}
        rate_mapping = _remove_mult_from_rate_mapping(
            mapping=rate_mapping, mult_features=mult_features
        )

    # create the grid from the mapping
    if grid is None:
        grid = suppress_logs(create_grid)(mapping=rate_mapping)
        grid = grid.drop(columns=["vals"])
        grid = suppress_logs(remove_duplicates)(df=grid)

    # preprocess the data
    preprocess_dict = suppress_logs(preprocessors.preprocess_data)(
        model_data=grid,
        feature_dict=rate_feature_dict,
        **preprocess_params,
    )
    rate_table = preprocess_dict["md_encoded"]
    if mult_features:
        # add the mult_features to the predictions
        def _add_null_mult_features(df, mapping, mult_features):
            for feature in mult_features:
                type_ = mapping[feature]["type"]
                if type_ == "ohe":
                    ohe_dict = dict(list(mapping[feature]["values"].items())[1:])
                    for col in ohe_dict.values():
                        df[col] = 0
                else:
                    df[feature] = next(iter(mapping[feature]["values"].keys()))

            return df

        rate_table = _add_null_mult_features(
            df=rate_table, mapping=preprocess_mapping, mult_features=mult_features
        )
    # prediction needs to be in same order as model
    model_features = models.ModelWrapper(model).get_features()
    rate_table = rate_table.loc[:, model_features]

    # create the rate table
    try:
        rate_table["vals"] = model.predict(rate_table)
    except Exception as e:
        raise ValueError("Error during preprocessing or prediction") from e

    # create the multiplier table
    if mult_features:
        mult_list = []
        base = rate_table["vals"].mean()

        for feature, mapping in mult_mapping.items():
            mult_table = rate_table.copy()
            mult_table = mult_table.drop(columns=["vals"])
            feature_vals = list(mapping["values"].keys())
            feature_encoded = list(mapping["values"].values())
            feature_type = mapping["type"]

            if feature_type == "ohe":
                for i, value in enumerate(feature_encoded):
                    if i == 0:
                        vals = model.predict(mult_table)
                    else:
                        mult_table[value] = 1
                        vals = model.predict(mult_table)
                        mult_table[value] = 0

                    multiple = vals.mean() / base
                    mult_list.append(
                        {
                            "category": feature,
                            "subcategory": feature_vals[i],
                            "multiple": multiple,
                        }
                    )

            else:
                # extend table for all values of feature
                extended_tables = [
                    mult_table.assign(**{feature: value}) for value in feature_encoded
                ]
                extended_table = pd.concat(extended_tables, ignore_index=True)
                extended_table["vals"] = model.predict(extended_table)

                grouped = extended_table.groupby(feature)["vals"].mean()
                for i, value in enumerate(feature_encoded):
                    multiple = grouped.loc[value] / base
                    mult_list.append(
                        {
                            "category": feature,
                            "subcategory": feature_vals[i],
                            "multiple": multiple,
                        }
                    )

        mult_table = pd.DataFrame(mult_list)
        mult_table = mult_table.sort_values(by=["category", "subcategory"])
        logger.info(f"mult_table rows: {mult_table.shape[0]}")

    rate_table = preprocessors.remap_values(df=rate_table, mapping=preprocess_mapping)
    col_reorder = [col for col in rate_table.columns if col != "vals"] + ["vals"]
    rate_table = rate_table[col_reorder]
    if mult_features is not None:
        rate_table = rate_table.drop(columns=mult_features)
    logger.info(f"rate_table shape: {rate_table.shape}")

    return rate_table, mult_table


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
    mort_grid = pl.DataFrame(grid, schema=column_names, orient="row")
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

    Removes invalid rows for attained age, duration, and issue age. Will also
    capp the attained age at the max_age.

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
        df = df.reset_index(drop=True)

    return df


def add_aa_ia_dur_cols(df):
    """
    Add attained age, issue age, and duration columns.

    Removes invalid rows for attained age, duration, and issue age. Will also
    capp the attained age at the max_age.

    attained_age = issue_age + duration - 1

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the columns checked.

    """
    initial_rows = len(df)

    # check for invalid attained age / duration / issue age combos
    if all(col in df.columns for col in ["attained_age", "issue_age", "duration"]):
        pass
    elif all(col in df.columns for col in ["attained_age", "duration"]):
        df["issue_age"] = df["attained_age"] - df["duration"] + 1
    elif all(col in df.columns for col in ["attained_age", "issue_age"]):
        df["duration"] = df["attained_age"] - df["issue_age"] + 1
    elif all(col in df.columns for col in ["issue_age", "duration"]):
        df["attained_age"] = df["issue_age"] + df["duration"] - 1
    elif all(col in df.columns for col in ["issue_age"]):
        df_list = [df]
        for attained_age in range(122):
            df_temp = df.copy()
            df_temp["attained_age"] = attained_age
            df_temp["duration"] = df_temp["attained_age"] - df_temp["issue_age"] + 1
            df_list.append(df_temp)
        df = pd.concat(df_list, ignore_index=True)
    elif all(col in df.columns for col in ["attained_age"]):
        df_list = [df]
        for issue_age in range(122):
            df_temp = df.copy()
            df_temp["issue_age"] = issue_age
            df_temp["duration"] = df_temp["attained_age"] - df_temp["issue_age"] + 1
            df_list.append(df_temp)
        df = pd.concat(df_list, ignore_index=True)
    elif all(col in df.columns for col in ["duration"]):
        df_list = [df]
        for issue_age in range(122):
            df_temp = df.copy()
            df_temp["issue_age"] = issue_age
            df_temp["attained_age"] = df_temp["issue_age"] + df_temp["duration"] - 1
            df_list.append(df_temp)
        df = pd.concat(df_list, ignore_index=True)
    else:
        raise ValueError(
            "attained_age, issue_age, or duration columns must be provided."
        )

    df = check_aa_ia_dur_cols(df)

    added_rows = len(df) - initial_rows
    if added_rows:
        logger.info(
            f"Added '{added_rows}' rows for attained_age, issue_age, or duration."
        )

    return df


def remove_duplicates(df):
    """
    Remove duplicates from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame without duplicates.

    """
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    logger.info(f"Removed '{removed_rows}' duplicates.")

    return df


def output_table(rate_table, filename="table.csv", mult_table=None):
    """
    Output the table to a csv file.

    Parameters
    ----------
    rate_table : pd.DataFrame
        The DataFrame.
    filename : str, optional (default="table.csv")
        The name of the file.
    mult_table : pd.DataFrame, optional (default=None)
        The multiplier table.

    """
    path = helpers.FILES_PATH / "dataset" / "tables" / filename

    # check if path exists
    if not path.parent.exists():
        logger.error(f"directory does not exist: {path.parent}")
    else:
        if mult_table is None:
            # check if .csv if not change it to .csv
            if path.suffix != ".csv":
                logger.warning(
                    f"changing file extension to .csv as it was {path.suffix}"
                )
                path = path.with_suffix(".csv")
            rate_table.to_csv(path, index=False)
        else:
            if path.suffix != ".xlsx":
                logger.warning(
                    f"changing file extension to .xlsx as it was {path.suffix}"
                )
                path = path.with_suffix(".xlsx")
            with pd.ExcelWriter(path) as writer:
                rate_table.to_excel(writer, sheet_name="rate_table", index=False)
                if mult_table is not None:
                    mult_table.to_excel(writer, sheet_name="mult_table", index=False)
        logger.info(f"saving table to {path}")


def get_su_table(df, select_period):
    """
    Calculate the select and ultimate ratio.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    select_period : int
        The select period.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the ratio column.

    """
    # getting the ultimate period and table
    if isinstance(select_period, str):
        logger.debug(
            f"select period is: '{select_period}'. Defaulting to 'max duration'"
        )
        # max duration
        select_period = df["duration"].max()

    # the minimum issue age will have the longest duration values
    logger.debug(
        f"calculating select ultimate ratio for select period: '{select_period}'"
    )
    ult = df[df["issue_age"] == df["issue_age"].min()].rename(
        columns={"vals": "vals_ult"}
    )
    drop_cols = [
        col
        for col in ult.columns
        if any(keyword in col for keyword in ["duration", "issue_age"])
    ]
    if drop_cols:
        ult = ult.drop(columns=drop_cols)

    # merge the ultimate values and calculate the ratio
    merge_cols = [col for col in ult.columns if col != "vals_ult"]
    df = df.merge(ult, on=merge_cols, how="left")
    df["su_ratio"] = df["vals_ult"] / df["vals"]
    df = df[df["duration"] <= (select_period + 1)]

    return df


def map_rates(df, rate, key_dict=None, rate_map_location=None):
    """
    Map rates to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    rate : str
        The rate to map.
    key_dict : dict, optional
        The key dictionary. If no key dictionary is provided, the mapping will
        be based on the keys in the rate file mapping.
          - The keys are the rate map keys
          - The values are the dataframe keys
    rate_map_location : str, optional
        The location of the rate map file. If none this is assumed to
        be in the dataset/tables folder.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the mapped rates.

    """
    # load rate map file
    if rate_map_location is None:
        rate_map_location = helpers.FILES_PATH / "dataset" / "tables" / "rate_map.yaml"
    if not rate_map_location.exists():
        raise ValueError(f"Rate mapping file: {rate_map_location} does not exist.")
    with open(rate_map_location, "r") as file:
        rate_map = yaml.safe_load(file)

    # check if rate is in the rate mapping
    if rate not in rate_map:
        rates = list(rate_map.keys())
        raise ValueError(f"Rate: {rate} not in rate_mapping. Try one of: {rates}.")

    # get the key dictionary
    if key_dict is None:
        key_cols = rate_map[rate]["keys"]
        key_dict = {col: col for col in key_cols}

    # check the type of the rate table
    type = next(iter(rate_map[rate]["type"].keys()))
    rate_name = f"qx_{rate_map[rate]['rate']}"
    apply_mults = False
    logger.info(f"mapping rate: '{rate_name}' with type: '{type}'")

    # get the table rates based on type
    if type == "soa":
        soa_map = rate_map[rate]["type"]["soa"]
        mt = MortTable()
        rate_table = mt.build_table(
            table_list=soa_map["table_list"],
            juv_list=soa_map["juv_list"],
            extend=soa_map["extend"],
            extra_dims=soa_map["extra_dims"],
        )
        merge_keys = list(key_dict.values())
        table_rate_name = "vals"
    if type == "workbook":
        workbook_map = rate_map[rate]["type"]["workbook"]
        file_location = (
            helpers.FILES_PATH / "dataset" / "tables" / workbook_map["filename"]
        )

        # read in the rate_table
        try:
            rate_table = pd.read_excel(file_location, sheet_name="rate_table")
        except ValueError as ve:
            raise ValueError(
                f"Error reading workbook: {file_location}. "
                f"The Excel file should have a sheet named 'rate_table'. "
            ) from ve
        table_key_dict = {
            key: value for key, value in key_dict.items() if key in rate_table.columns
        }
        rate_table = rate_table.rename(columns=table_key_dict)
        for key in table_key_dict:
            rate_table[key] = rate_table[key].astype(df[key].dtype)
        merge_keys = list(table_key_dict.values())
        table_rate_name = "rate"

        # read in the mult_table
        if workbook_map["mult_table"]:
            apply_mults = True
            try:
                mult_table = pd.read_excel(file_location, sheet_name="mult_table")
            except ValueError as ve:
                raise ValueError(
                    f"Error reading workbook: {file_location}. "
                    f"The Excel file should have a sheet named 'mult_table'. "
                ) from ve

            # map the mult_table
            mult_key_dict = {
                key: value
                for key, value in key_dict.items()
                if key in list(mult_table["category"].unique())
            }
            for key in mult_key_dict:
                _pivot = (
                    mult_table[mult_table["category"] == key]
                    .pivot(index="subcategory", columns="category", values="multiple")
                    .reset_index()
                    .rename(columns={key: f"_mult_{key}", "subcategory": key})
                )
                _pivot[key] = _pivot[key].astype(df[key].dtype)
                df = df.merge(_pivot, on=key, how="left")
                # check for missing values in table
                missing_mult_values = set(df[key].unique()) - set(_pivot[key].unique())
                if missing_mult_values:
                    logger.warning(
                        f"Missing values in the mult_table for '{key}': "
                        f"{missing_mult_values}"
                    )
            _mult_cols = [col for col in df.columns if "_mult_" in col]
    if type == "csv":
        csv_map = rate_map[rate]["type"]["csv"]
        file_location = helpers.FILES_PATH / "dataset" / "tables" / csv_map["filename"]
        # read in the csv
        try:
            rate_table = pd.read_csv(file_location)
        except ValueError as ve:
            raise ValueError(f"Error reading csv: {file_location}. ") from ve
        merge_keys = list(key_dict.values())
        table_rate_name = "vals"

    # map the rates
    rate_table = rate_table.rename(columns=key_dict)
    rate_table = rate_table[[*merge_keys, table_rate_name]]
    rate_table = rate_table.rename(columns={table_rate_name: rate_name})
    if rate_name in df.columns:
        logger.warning(
            f"rate: '{rate_name}' already exists in the DataFrame. "
            f"Overwriting the rate."
        )
        df = df.drop(columns=[rate_name])
    df = df.merge(rate_table, on=merge_keys, how="left")
    if apply_mults:
        for _mult_col in _mult_cols:
            df[rate_name] = df[rate_name] * df[_mult_col]
        df = df.drop(columns=_mult_cols)
        merge_keys = merge_keys + list(mult_key_dict.keys())
    logger.info(f"the mapped rates are based on the following " f"keys: {merge_keys}")

    # check if there are any missing values
    missing = df[df[rate_name].isnull()]
    if not missing.empty:
        logger.warning(f" there are '{len(missing)}' missing values for '{rate_name}'.")

    return df
