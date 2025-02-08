"""Mortality Table Builder."""

import copy
import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import pymort
import yaml

from morai.forecast import models, preprocessors
from morai.utils import custom_logger, helpers
from morai.utils.custom_logger import suppress_logs

logger = custom_logger.setup_logging(__name__)

if TYPE_CHECKING:
    from pathlib import Path
    from xml.etree.ElementTree import Element


class MortTable:
    """
    A mortality table class that can be used to build a 1-d mortality table.

    There are a number of functions in the class including:
        - build_table: build a 1-d mortality table from a list of tables
        - get_soa_xml: get the soa xml object
    """

    def __init__(
        self,
        rate: Optional[pd.DataFrame] = None,
        rate_filename: Optional[Union[str, "Path"]] = None,
    ) -> None:
        """
        Initialize the Table class.

        Parameters
        ----------
        rate : str, optional (default=None)
            A rate to use for the table. The rate can be "vbt15".
        rate_filename : str, optional (default=None)
            The filename of the rate map. default name is rate_map.yaml.

        """
        self.rate_table = None
        self.mult_table = None
        self.rate_dict = None
        self.rate_name = None
        self.select_period = None
        self.max_age = 121
        if rate_filename is None:
            rate_filename = "rate_map.yaml"

        # building rate based on the rate file
        if rate:
            self.rate_dict = get_rate_dict(rate, rate_filename)
            rate_type = next(iter(self.rate_dict["type"].keys()))
            col_keys = self.rate_dict["keys"] + ["vals"]
            logger.info(f"building table for rate: '{rate}' with format: '{rate_type}'")
            self.rate_name = f"qx_{self.rate_dict['rate']}"
            if rate_type == "soa":
                self.rate_table = self.build_table_soa(
                    table_list=self.rate_dict["type"]["soa"]["table_list"],
                    extra_dims=self.rate_dict["type"]["soa"]["extra_dims"],
                    juv_list=self.rate_dict["type"]["soa"]["juv_list"],
                    extend=self.rate_dict["type"]["soa"]["extend"],
                )
            elif rate_type == "csv":
                csv_location = get_filepath(self.rate_dict["type"]["csv"]["filename"])
                # read in the csv
                try:
                    self.rate_table = pd.read_csv(csv_location, usecols=col_keys)
                except ValueError as ve:
                    raise ValueError(f"Error reading csv: {csv_location}. ") from ve
            elif rate_type == "workbook":
                workbook_location = get_filepath(
                    self.rate_dict["type"]["workbook"]["filename"]
                )
                self.rate_table, self.mult_table = self.build_table_workbook(
                    file_location=workbook_location,
                    has_mults=self.rate_dict["type"]["workbook"]["mult_table"],
                )

    def build_table_workbook(
        self, file_location: Union[str, "Path"], has_mults: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a 1-d mortality table from a workbook.

        A 1-d table is where there is a 'vals' column and all the other columns are the
        features. If the workbook has a multiplier table, then the multiplier table will
        be returned as well.

        Parameters
        ----------
        file_location : str
            The location of the workbook.
        has_mults : bool, optional (default=False)
            Whether the workbook has a multiplier table.

        Returns
        -------
        rate_table : pd.DataFrame
            The rate table.
        mult_table : pd.DataFrame
            The multiplier table.

        """
        mult_table = None

        # read in the rate_table
        try:
            rate_table = pd.read_excel(file_location, sheet_name="rate_table")
        except ValueError as ve:
            raise ValueError(
                f"Error reading workbook: {file_location}. "
                f"The Excel file should have a sheet named 'rate_table'. "
            ) from ve

        # read in the mult_table
        if has_mults:
            try:
                mult_table = pd.read_excel(file_location, sheet_name="mult_table")
            except ValueError as ve:
                raise ValueError(
                    f"Error reading workbook: {file_location}. "
                    f"The Excel file should have a sheet named 'mult_table'. "
                ) from ve

        self.rate_table = rate_table
        self.mult_table = mult_table

        return rate_table, mult_table

    def build_table_soa(
        self,
        table_list: List[int],
        extra_dims: Optional[Dict[str, List[str]]] = None,
        juv_list: Optional[List[int]] = None,
        extend: bool = False,
    ) -> pd.DataFrame:
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
        mort_table["vals"] = np.nan
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

        self.rate_table = mort_table
        self.table_list = table_list
        self.extra_dims = extra_dims
        self.juv_list = juv_list
        self.extend = extend
        self.select_period = select_period

        return mort_table

    def get_soa_xml(self, table_id: int) -> Any:
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

    def calc_derived_table_from_mults(
        self,
        selected_dict: Optional[dict[str, list]] = None,
    ) -> pd.DataFrame:
        """
        Calculate a derived rate table from the rate table and multiplier table.

        The derived table is calculated by multiplying the rate table by the
        multiplier table given the selected multiplier columns.

        Parameters
        ----------
        selected_dict : dict, optional (default=None)
            The selected multiplier columns.
            If None, then the first subcategory multiplier of each category
            will be used.

        Returns
        -------
        derived_table : pd.DataFrame
            The derived table.

        """
        if self.rate_table is None or self.mult_table is None:
            raise ValueError(
                "calc_derived_table_from_mults requires the rate table and "
                "multiplier table to be set."
            )

        # get subcategory multipliers if not provided
        if selected_dict is None:
            first_mults = self.mult_table.groupby("category").first().reset_index()
            selected_dict = (
                first_mults.set_index("category")["subcategory"]
                .apply(lambda x: [x])
                .to_dict()
            )

        # check that subcategory and category are in the multiplier table
        # if not set(selected_dict.keys()).issubset(self.mult_table["category"].unique()):
        #     raise ValueError(
        #         "selected_dict keys must be in the multiplier table `category` column"
        #     )
        # if not set(selected_dict.values()).issubset(
        #     self.mult_table["subcategory"].unique()
        # ):
        #     raise ValueError(
        #         "selected_dict values must be in the multiplier table "
        #         "`subcategory` column."
        #     )

        # select the rows in mult_table that match the selected mults
        selected_mults = self.mult_table[
            self.mult_table.apply(
                lambda row: row["subcategory"]
                in selected_dict.get(row["category"], []),
                axis=1,
            )
        ]

        # calculate the multiplier
        mean_category_mult = selected_mults.groupby("category")["multiple"].mean()
        product_of_means = np.prod(mean_category_mult)
        logger.info(f"derived table using multiplier: `{product_of_means:.2f}`")
        logger.info(
            f"used the following subcategories: " f"`{list(selected_dict.values())}`"
        )

        # apply the multiplier
        derived_table = self.rate_table.copy()
        derived_table["vals"] = derived_table["vals"] * product_of_means

        return derived_table

    def _merge_tables(
        self,
        merge_table: pd.DataFrame,
        source_table: pd.DataFrame,
        merge_keys: List[str],
        column_rename: str,
        extra_dims_list: Optional[List[Tuple[str, Any]]] = None,
    ) -> pd.DataFrame:
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

    def _process_soa_table(
        self, soa_xml: "Element", table_index: int, is_select: bool
    ) -> Tuple[Any, int, int]:
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
    model: Any,
    mapping: Dict[str, Any],
    preprocess_feature_dict: Dict[str, Any],
    preprocess_params: Dict[str, Any],
    grid: Optional[pd.DataFrame] = None,
    mult_features: Optional[List[str]] = None,
    mult_method: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a 1-d mortality table based on model predictions.

    A 1-d table is where there is a 'vals' column and all the other columns are the
    features.

    Parameters
    ----------
    model : model
        The model to use for generating the table
    mapping : dict
        The mapping dictionary. This is used for a number of processes
          - creating the grid if it is not provided
          - identifying the mult_features and then predicting based on type
          - remapping the encoded values back to the original values
    preprocess_feature_dict : dict
        The preprocess feature dictionary that was used for the model that will
        encode the features
    preprocess_params : dict
        The parameters that were used for preprocessing that will also be used to
        encode the features
    grid : pd.DataFrame, optional
        The grid to use for the table.
    mult_features : list, optional
        The features to use for the multiplier table. This is based on the method
        of `mult_method`.
    mult_method : str, optional
        The method to use for the multiplier table. The options are:
          - "mean": the mean prediction for the feature
          - "glm": uses the initial log ratio of the glm model


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
    rate_mapping = mapping
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
            "THIS IS EXPERIMENTAL: "
            "the multipliers most likely not match the predictions exactly from "
            "the model and is used to simplify the output."
        )
        mult_mapping = {k: v for k, v in rate_mapping.items() if k in mult_features}
        rate_mapping = _remove_mult_from_rate_mapping(
            mapping=rate_mapping, mult_features=mult_features
        )

    # create the grid from the mapping
    if grid is None:
        grid = suppress_logs(create_grid)(mapping=rate_mapping)
        grid = suppress_logs(remove_duplicates)(df=grid)

    # preprocess the data
    preprocess_dict = suppress_logs(preprocessors.preprocess_data)(
        model_data=grid,
        feature_dict=rate_feature_dict,
        **preprocess_params,
    )
    rate_table = preprocess_dict["md_encoded"]

    # add the mult_features to the predictions
    if mult_features:
        rate_table = _add_null_mult_features(
            df=rate_table, mapping=mapping, mult_features=mult_features
        )

    # prediction needs to be in same order as model
    model_features = models.ModelWrapper(model).get_features()
    rate_table = rate_table.loc[:, model_features]

    # make predictions
    try:
        rate_table["vals"] = model.predict(rate_table)
    except Exception as e:
        raise ValueError("Error during preprocessing or prediction") from e

    # create the multiplier table
    if mult_features:
        mult_list = []
        if mult_method == "mean":
            logger.info("creating multiplier table based on the 'mean'")
            base = rate_table["vals"].mean()

            for feature, feature_map in mult_mapping.items():
                mult_table = rate_table.copy()
                mult_table = mult_table.drop(columns=["vals"])
                feature_vals = list(feature_map["values"].keys())
                feature_encoded = list(feature_map["values"].values())
                feature_type = feature_map["type"]

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

                # add all feature values to the table
                else:
                    extended_tables = [
                        mult_table.assign(**{feature: value})
                        for value in feature_encoded
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
        elif mult_method == "glm":
            logger.info("creating multiplier table based on the 'glm'")
            for feature, feature_map in mult_mapping.items():
                if not hasattr(model, "params"):
                    raise ValueError(
                        f"model: {model}, does not have 'params' attribute"
                    )
                odds = np.exp(model.params)
                feature_vals = list(feature_map["values"].keys())
                if feature_map["type"] == "ohe":
                    for feature_val in feature_vals:
                        feature_lookup = f"{feature}_{feature_val}"
                        multiple = odds.get(feature_lookup, 1)
                        mult_list.append(
                            {
                                "category": feature,
                                "subcategory": feature_val,
                                "multiple": multiple,
                            }
                        )
                else:
                    for i, feature_val in enumerate(feature_vals):
                        multiple = np.exp(model.params.get(feature, 0) * i)
                        mult_list.append(
                            {
                                "category": feature,
                                "subcategory": feature_val,
                                "multiple": multiple,
                            }
                        )
        else:
            raise ValueError(f"mult_method: {mult_method} not recognized")

        mult_table = pd.DataFrame(mult_list)
        mult_table = mult_table.sort_values(by=["category", "subcategory"])
        logger.info(f"mult_table rows: {mult_table.shape[0]}")

    rate_table = preprocessors.remap_values(df=rate_table, mapping=mapping)
    col_reorder = [col for col in rate_table.columns if col != "vals"] + ["vals"]
    rate_table = rate_table[col_reorder]
    if mult_features is not None:
        rate_table = rate_table.drop(columns=mult_features)
    logger.info(f"rate_table shape: {rate_table.shape}")

    return rate_table, mult_table


def map_rates(
    df: pd.DataFrame,
    rate: str,
    rate_to_df_map: Optional[Dict[str, str]] = None,
    rate_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map rates to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to map the rates to.
    rate : str
        The rate to map to be looked up in the rate mapping file.
    rate_to_df_map : dict, optional
        The key dictionary. If no key dictionary is provided, the mapping will
        be based on the key list in the rate file mapping.
          - The keys are the rate map keys
          - The values are the dataframe keys
    rate_filename : str, optional
        The location of the rate map file. If none this is assumed to
        be in the dataset/tables folder.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with the mapped rates.

    """
    # get the table
    mt = suppress_logs(MortTable)(rate=rate, rate_filename=rate_filename)
    rate_table = mt.rate_table
    mult_table = mt.mult_table
    rate_dict = mt.rate_dict
    rate_type = next(iter(rate_dict["type"].keys()))
    rate_name = mt.rate_name
    logger.info(f"mapping rate: '{rate_name}' with format: '{rate_type}'")

    # create rate_to_df_map if not provided
    if rate_to_df_map is None:
        logger.debug(
            "create 'rate_to_df_map' which assumes the keys are the same "
            "between the rate and df."
        )
        rate_cols = rate_dict["keys"]
        rate_to_df_map = {col: col for col in rate_cols}
    # check if columns are in the DataFrame
    for df_col in rate_to_df_map.values():
        if df_col not in df.columns:
            raise ValueError(f"column: '{df_col}' not in the DataFrame.")
    # update the mapping if there is a mult_table
    if mult_table is not None:
        mult_to_df_map = {
            mult_col: df_col
            for mult_col, df_col in rate_to_df_map.items()
            if mult_col in list(mult_table["category"].unique())
        }
        rate_to_df_map = {
            rate_col: df_col
            for rate_col, df_col in rate_to_df_map.items()
            if rate_col in rate_table.columns
        }

    # update the rate_table to merge with the df
    merge_keys = list(rate_to_df_map.values())
    table_rate_name = "vals"
    rate_table = rate_table.rename(columns=rate_to_df_map)
    for df_col in rate_to_df_map.values():
        rate_table[df_col] = rate_table[df_col].astype(df[df_col].dtype)
    rate_table = rate_table[[*merge_keys, table_rate_name]]
    rate_table = rate_table.rename(columns={table_rate_name: rate_name})
    if rate_name in df.columns:
        logger.warning(
            f"rate: '{rate_name}' already exists in the DataFrame. "
            f"Overwriting the rate."
        )
        df = df.drop(columns=[rate_name])

    # update the mult_table to merge with the df
    if mult_table is not None:
        for mult_col in mult_to_df_map:
            _pivot = (
                mult_table[mult_table["category"] == mult_col]
                .pivot(index="subcategory", columns="category", values="multiple")
                .reset_index()
                .rename(
                    columns={mult_col: f"_mult_{mult_col}", "subcategory": mult_col}
                )
            )
            _pivot[mult_col] = _pivot[mult_col].astype(df[mult_col].dtype)
            df = df.merge(_pivot, on=mult_col, how="left")
            # check for missing values in table
            missing_mult_values = set(df[mult_col].unique()) - set(
                _pivot[mult_col].unique()
            )
            if missing_mult_values:
                logger.warning(
                    f"Missing values in the mult_table for '{mult_col}' but in df: "
                    f"{missing_mult_values}"
                )
        mult_cols = [col for col in df.columns if "_mult_" in col]

    # merge in the rates
    df = df.merge(rate_table, on=merge_keys, how="left")
    if mult_table is not None:
        for mult_col in mult_cols:
            df[rate_name] = df[rate_name] * df[mult_col]
        df = df.drop(columns=mult_cols)
        merge_keys = merge_keys + list(mult_to_df_map.keys())
    logger.info(f"the mapped rates are based on the following " f"keys: {merge_keys}")

    # check if there are any missing rates
    missing_rates = df[df[rate_name].isnull()]
    if not missing_rates.empty:
        logger.warning(
            f" there are '{len(missing_rates)}' missing values for '{rate_name}'."
        )

    return df


def create_grid(
    dims: Optional[Dict[str, Union[List[Any], np.ndarray]]] = None,
    mapping: Optional[Dict[str, Dict[str, Union[List[Any], np.ndarray]]]] = None,
    max_age: int = 121,
    max_grid_size: int = 5_000_000,
    mult_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create an empty grid from the dimensions.

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
    mult_features : list, optional
        The features to use for the multiplier table.

    Returns
    -------
    mort_grid : pd.DataFrame
        The grid.

    """
    if (not dims and not mapping) or (dims and mapping):
        raise ValueError("Either dims or mapping must be provided.")
    if mapping:
        if mult_features:
            mapping = _remove_mult_from_rate_mapping(mapping, mult_features)
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

    return mort_grid


def compare_tables(
    table_1: pd.DataFrame, table_2: pd.DataFrame, value_col: str = "vals"
) -> pd.DataFrame:
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


def check_aa_ia_dur_cols(df: pd.DataFrame, max_age: int = 121) -> pd.DataFrame:
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


def add_aa_ia_dur_cols(df: pd.DataFrame) -> pd.DataFrame:
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


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
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


def output_table(
    rate_table: pd.DataFrame,
    filename: str = "table.csv",
    mult_table: Optional[pd.DataFrame] = None,
) -> None:
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
    path = helpers.FILES_PATH / "rates" / filename

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


def get_su_table(df: pd.DataFrame, select_period: int) -> pd.DataFrame:
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


def get_rates(rate_filename: Optional[str] = None) -> List[str]:
    """
    Get the possible rates in rate mapping file.

    Parameters
    ----------
    rate_filename : str, optional
        The filename of the rate map file. If none this is assumed to
        be in the dataset/tables folder.

    Returns
    -------
    rates : list
        The rates in the rate mapping file.

    """
    # load rate map file
    if rate_filename is None:
        rate_filename = "rate_map.yaml"
    rate_map_location = get_filepath(rate_filename)
    with open(rate_map_location, "r") as file:
        rate_map = yaml.safe_load(file)

    rates = list(rate_map.keys())

    return rates


def get_rate_dict(rate: str, rate_filename: Optional[str] = None) -> Dict[str, str]:
    """
    Process the rate file.

    Parameters
    ----------
    rate : str
        The rate to be looked up in the rate mapping file.
    rate_filename : str, optional
        The filename of the rate map file. If none this is assumed to
        be in the dataset/tables folder.

    Returns
    -------
    rate_dict : dict
        The rate dictionary

    """
    # load rate map file
    if rate_filename is None:
        rate_filename = "rate_map.yaml"
    rate_map_location = get_filepath(rate_filename)
    with open(rate_map_location, "r") as file:
        rate_map = yaml.safe_load(file)

    # check if rate is in the rate mapping
    if rate not in rate_map:
        rates = list(rate_map.keys())
        raise ValueError(f"Rate: {rate} not in rate_mapping. Try one of: {rates}.")

    # get the rate dictionary
    logger.info(f"loading '{rate}' from mapping file: {rate_map_location}")
    rate_dict = rate_map[rate]

    return rate_dict


def get_filepath(filename: str) -> "Path":
    """
    Get the file path based on a number of paths.

    Parameters
    ----------
    filename : str
        The file location.

    Returns
    -------
    filepath : file
        The file.

    """
    filepaths = [
        helpers.FILES_PATH / "rates" / filename,
        helpers.ROOT_PATH / "tests" / "files" / "experience" / "tables" / filename,
        filename,
    ]
    for filepath in filepaths:
        if filepath.exists():
            break
    if not filepath.exists():
        raise ValueError(f"File: {filename} not found in any of the paths.")
    return filepath


def _add_null_mult_features(
    df: pd.DataFrame, mapping: Dict[str, Dict[str, Any]], mult_features: List[str]
) -> pd.DataFrame:
    logger.debug("adding initial value for multiplier features")
    for feature in mult_features:
        type_ = mapping[feature]["type"]
        if type_ == "ohe":
            ohe_dict = dict(list(mapping[feature]["values"].items())[1:])
            for col in ohe_dict.values():
                df[col] = 0
        else:
            df[feature] = next(iter(mapping[feature]["values"].values()))

    return df


def _remove_mult_from_rate_mapping(
    mapping: Dict[str, Dict[str, Any]], mult_features: List[str]
) -> Dict[str, Dict[str, Any]]:
    logger.debug("removing multiplier features from rate mapping")
    mapping = copy.deepcopy(mapping)
    for key, sub_dict in mapping.items():
        if key in mult_features and "values" in sub_dict:
            first_key = next(iter(sub_dict["values"]))
            sub_dict["values"] = {first_key: sub_dict["values"][first_key]}
    return mapping
