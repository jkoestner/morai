"""Mortality Table Builder."""

import itertools

import pandas as pd
import pymort

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

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
        mort_table = self._create_grid(dims=dims)

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

    def _create_grid(self, dims):
        """
        Create a grid from the dimensions.

        Parameters
        ----------
        dims : dict
            The dimensions.

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
            mort_grid["attained_age"] = (
                mort_grid["issue_age"] + mort_grid["duration"] - 1
            )
            mort_grid = mort_grid[mort_grid["attained_age"] <= self.max_age]
        mort_grid["vals"] = None
        return mort_grid
