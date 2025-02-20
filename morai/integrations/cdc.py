"""
CDC Data.

This module contains functions to work with CDC data. The CDC data is a
collection of data from the CDC Wonder database.

Link: https://wonder.cdc.gov/

There are data limitations from the CDC web service listed here
https://wonder.cdc.gov/wonder/help/WONDER-API.html

For instance the web service does not provide support for location based filters.

If wanting to further develop integration the following repository is a good
reference.

Link: https://github.com/alipphardt/cdc-wonder-api
"""

import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests

from morai.forecast import graduation
from morai.utils import custom_logger, helpers, sql
from morai.utils.custom_logger import suppress_logs
from morai.utils.helpers import check_merge

logger = custom_logger.setup_logging(__name__)


def get_cdc_data_xml(
    xml_filename: str,
    parse_date_col: Optional[str] = None,
    convert_dtypes: bool = True,
    clean_df: bool = True,
) -> pd.DataFrame:
    """
    Get CDC data from an XML file.

    Parameters
    ----------
    xml_filename : str
        Path to the XML file that contains the request parameters.
    parse_date_col : str, optional
        Column name to parse dates.
    convert_dtypes : bool, optional
        Convert data types to the correct type.
    clean_df : bool, optional
        Clean the DataFrame.

    Returns
    -------
    cdc_df : pd.DataFrame
        DataFrame object.

    """
    # read the xml file
    xml_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "xml" / xml_filename
    if not xml_filepath.exists():
        logger.error(f"File not found: {xml_filepath}")
        return None
    with open(xml_filepath, "r") as file:
        xml_request = file.read()

    # query cdc wonder
    data_id = _xml_parse_dataid(xml_request)
    url = f"https://wonder.cdc.gov/controller/datarequest/{data_id}"
    logger.debug(f"requesting data from CDC Wonder: {url}")

    # https://wonder.cdc.gov/datause.html
    response = requests.post(
        url, data={"request_xml": xml_request, "accept_datause_restrictions": "true"}
    )

    if response.status_code == 200:
        xml_response = response.text
    else:
        raise Exception(f"Error: {response.status_code, response.text}")

    # create the dataframe from the response
    logger.debug("creating dataframe from response")
    cdc_df = _xml_create_df(xml_response=xml_response)

    # parse the month column
    if parse_date_col:
        if parse_date_col not in cdc_df.columns:
            logger.warning(f"Column not found: {parse_date_col}")
            return cdc_df
        cdc_df[parse_date_col] = _parse_date_col(df=cdc_df, col=parse_date_col)

    # clean the dataframe
    if clean_df:
        cdc_df = suppress_logs(helpers.clean_df)(cdc_df, update_cat=False)
        if "year" in cdc_df.columns:
            cdc_df["year"] = cdc_df["year"].str[:4]

    if convert_dtypes:
        cdc_df = _infer_dtypes(cdc_df)

    return cdc_df


def get_cdc_data_txt(
    txt_filename: str, convert_dtypes: bool = True, clean_df: bool = True
) -> pd.DataFrame:
    """
    Get CDC data from a TXT file.

    Parameters
    ----------
    txt_filename : str
        Path to the TXT file that contains the data.
    convert_dtypes : bool, optional
        Convert data types to the correct type.
    clean_df : bool, optional
        Clean the DataFrame.

    Returns
    -------
    cdc_df : pd.DataFrame
        DataFrame object.

    """
    # read the txt file
    txt_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "txt" / txt_filename
    if not txt_filepath.exists():
        logger.error(f"File not found: {txt_filepath}")
        return None
    cdc_df = pd.read_csv(txt_filepath, sep="\t")

    # clean the dataframe
    if clean_df:
        cdc_df = suppress_logs(helpers.clean_df)(cdc_df, update_cat=False)
        if "notes" in cdc_df.columns:
            cdc_df = cdc_df.drop(columns=["notes"])
        # delete rows with all null values
        cdc_df = cdc_df.dropna(how="all")

    if convert_dtypes:
        cdc_df = _infer_dtypes(cdc_df)

    return cdc_df


def get_cdc_data_sql(db_filepath: str, table_name: str) -> pd.DataFrame:
    """
    Get CDC data from a SQLite database.

    Parameters
    ----------
    db_filepath : str
        Database file path.
    table_name : str
        Table name.

    Returns
    -------
    cdc_df : pd.DataFrame
        DataFrame object.

    """
    # read the data from the database
    query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY added_at
            DESC LIMIT 1
            """
    last_date_row = sql.read_sql(db_filepath, query)
    last_date = last_date_row["added_at"].iloc[0]

    query = f"""
            SELECT *
            FROM {table_name}
            """
    cdc_df = sql.read_sql(db_filepath, query)

    # correctly order the age groups
    if "age_groups" in cdc_df.columns:
        age_group_order = [
            "Not Stated",
            "< 1 year",
            "1-4 years",
            "5-9 years",
            "10-14 years",
            "5-14 years",
            "15-19 years",
            "20-24 years",
            "15-24 years",
            "25-34 years",
            "35-44 years",
            "45-54 years",
            "55-64 years",
            "65-74 years",
            "75-84 years",
            "85+ years",
            "total",
        ]
        cdc_df["age_groups"] = pd.Categorical(
            cdc_df["age_groups"], categories=age_group_order, ordered=True
        )
        cdc_df["age_groups"] = cdc_df["age_groups"].cat.remove_unused_categories()

    logger.debug(
        f"table `{table_name}` last updated at: {last_date}, rows: {len(cdc_df)}"
    )

    return cdc_df


def get_last_updated(table_name: Optional[str] = None) -> str:
    """Get the last updated date of a table."""
    if table_name is None:
        table_name = "mcd18_cod"
    db_filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc.sql"
    tables = sql.get_tables(db_filepath=db_filepath)
    if table_name not in tables:
        return ""
    df = get_cdc_data_sql(db_filepath=db_filepath, table_name=table_name)
    return df["added_at"].max()


def get_cdc_reference(sheet_name: str) -> pd.DataFrame:
    """
    Get CDC reference data.

    Parameters
    ----------
    sheet_name : str
        Sheet name.

    Returns
    -------
    reference_df : pd.DataFrame
        DataFrame object.

    """
    # read the data from the database
    filepath = helpers.FILES_PATH / "integrations" / "cdc" / "cdc_reference.xlsx"
    if not filepath.exists():
        logger.error(f"The reference file was not found here: {filepath}")
        return None
    reference_df = pd.read_excel(filepath, sheet_name=sheet_name)

    return reference_df


def map_reference(
    df: pd.DataFrame,
    col: str,
    on_dict: Optional[dict] = None,
    sheet_name: str = "cod",
    category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map a column from the CDC reference to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to map the column onto.
    col : str
        The column in the reference to map.
    on_dict : dict, optional
        Dictionary of the columns to match on.
        e.g. {"df_column": "reference_column"}
        If not provided, the default is {"icd-10_113_cause_list": "wonder_cause"}.
    sheet_name : str, optional
        The sheet name to use.
    category : str, optional
        The category to use.

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the column mapped.

    """
    logger.debug(f"mapping reference column: {col}")
    # get the mapping sheet with appropiate reference and col
    mapping = get_cdc_reference(sheet_name=sheet_name)
    if category is not None:
        if category not in mapping["category"].unique():
            logger.error(f"The category {category} does not exist.")
            return df
        mapping = mapping[mapping["category"] == category]
    if on_dict is None:
        on_dict = {"icd-10_113_cause_list": "cause_wonder"}
    df_on = next(iter(on_dict.keys()))
    reference_on = next(iter(on_dict.values()))
    mapping = mapping[[reference_on, col]]
    mapping = mapping.drop_duplicates()

    # merge the col into the df
    df = check_merge(pd.merge)(
        left=df,
        right=mapping[[reference_on, col]],
        how="left",
        left_on=df_on,
        right_on=reference_on,
    )

    # drop the reference column
    if reference_on != df_on and reference_on in df.columns:
        df = df.drop(columns=[reference_on])

    return df


def calc_mi(df: pd.DataFrame, rolling: int = 10) -> pd.DataFrame:
    """
    Calulate the crude mortality improvment using a standardized population.

    The dataframe will create a "1_year_mi" column and also an average mi column based
    on the rolling parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the deaths and the population by time period.
    rolling : int, optional
        The rolling average time period.

    Returns
    -------
    mi_df : pd.DataFrame
        DataFrame object.

    """
    # check if the columns exist
    columns_needed = ["year", "age_groups", "deaths", "population"]
    if not all(col in df.columns for col in columns_needed):
        logger.error(f"Columns not found: {columns_needed}")
        return None

    logger.info(
        "calculating mortality improvement by using a `2000 age adjusted` crude "
        "mortality rate"
    )
    # group and calculate crude 2000 adjusted mortality rate
    mi_df = (
        df.groupby(["year", "age_groups"])[["deaths", "population"]].sum().reset_index()
    )
    mi_df = map_reference(
        df=mi_df,
        col="population_%",
        on_dict={"age_groups": "age_bucket"},
        sheet_name="age_std_2000",
    )
    mi_df["crude_adj"] = (
        mi_df["deaths"] / mi_df["population"] * 100000 * mi_df["population_%"]
    )

    # calculate mortality improvement
    mi_df = mi_df.groupby(["year"])[["crude_adj", "deaths"]].sum().reset_index()
    mi_df["1_year_mi"] = 1 - (mi_df["crude_adj"] / mi_df["crude_adj"].shift(1))
    mi_df[f"{rolling}_year_mi"] = mi_df["1_year_mi"].rolling(window=rolling).mean()

    # calculate whl
    mi_df = mi_df[mi_df["1_year_mi"].notna()]
    mi_df["whl_3"] = graduation.whl(
        rates=mi_df["1_year_mi"], horizontal_order=3, horizontal_lambda=400
    )

    return mi_df


def compare_dfs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    compare_col_dict: Optional[dict] = None,
    compare_value_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compare two DataFrames.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left DataFrame.
    right_df : pd.DataFrame
        Right DataFrame.
    compare_col_dict : dict, optional
        Dictionary of the columns to compare.
    compare_value_dict : dict, optional
        Dictionary of the values to compare.

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame object.

    """
    # set the default values
    if compare_value_dict is None:
        compare_value_dict = {"deaths": "deaths"}
    if compare_col_dict is None:
        compare_col_dict = {"year": "year"}

    # get the columns and values to compare
    left_col = next(iter(compare_col_dict.keys()))
    left_value = next(iter(compare_value_dict.keys()))
    right_col = next(iter(compare_col_dict.values()))
    right_value = next(iter(compare_value_dict.values()))

    # compare the dfs
    compare_df = left_df.groupby(left_col)[left_value].sum().reset_index()
    compare_df = compare_df.merge(
        right_df.groupby(right_col)[right_value].sum().reset_index(),
        how="outer",
        left_on=left_col,
        right_on=right_col,
        suffixes=("_left", "_right"),
    )
    compare_df[f"{left_value}_diff"] = (
        compare_df[f"{left_value}_left"] - compare_df[f"{right_value}_right"]
    )

    return compare_df


def _xml_parse_dataid(xml_string: str) -> str:
    """
    Parse the data-id from an XML string object.

    Parameters
    ----------
    xml_string : str
        XML string object.

    Returns
    -------
    data_id : str
        Data ID.

    """
    root = ET.fromstring(xml_string)
    value = root.find(".//parameter[name='B_1']/value").text
    data_id = value.split(".")[0]

    return data_id


def get_top_deaths_by_age_group(
    df: pd.DataFrame, year: int, cod_col: str = "simple_grouping"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the top deaths by age group for a given year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object.
    year : int
        Year to get the top deaths by age group.
    cod_col : str, optional
        Column name to get the top deaths by.

    Returns
    -------
    (deaths_pivot, names_pivot) : tuple
        Tuple of the deaths pivot and the names pivot

    """
    # filter to year
    df_year = df[df["year"] == year].copy()

    # drop column if exists and filter out look up column
    if cod_col in df_year.columns:
        df_year = df_year.drop(columns=[cod_col])
    df_year = df_year[df_year["icd_-_sub-chapter"].notna()]

    # map the cod column
    df_year = map_reference(
        df=df_year,
        col=cod_col,
        on_dict={"icd_-_sub-chapter": "wonder_sub_chapter"},
    )

    # group the data for top 10 deaths in each age_group
    grouped = (
        df_year.groupby(["age_groups", cod_col], observed=False)["deaths"]
        .sum()
        .reset_index()
    )
    grouped = grouped.sort_values(by=["age_groups", "deaths"], ascending=[False, False])
    top_deaths = grouped.groupby("age_groups", observed=False).head(10).copy()
    top_deaths["rank"] = top_deaths.groupby("age_groups", observed=False)[
        "deaths"
    ].rank(method="first", ascending=False)

    # create the pivots
    deaths_pivot = top_deaths.pivot(index="rank", columns="age_groups", values="deaths")
    names_pivot = top_deaths.pivot(index="rank", columns="age_groups", values=cod_col)

    return (deaths_pivot, names_pivot)


def _xml_create_df(xml_response: str) -> pd.DataFrame:
    """
    Create a DataFrame from an XML string object.

    Parameters
    ----------
    xml_response : str
        XML string object that is the response

    Returns
    -------
    df : pd.DataFrame
        DataFrame object.

    """
    root = ET.fromstring(xml_response)
    # get the data-table
    data_table = root.find(".//data-table")

    # check if show-totals is true
    show_totals = root.find(".//parameter[@code='O_show_totals']/value").text
    if show_totals == "true":
        logger.warning(
            "show totals is true, this should be false with option `O_show_totals`"
        )

    # get the cdc mapping for columns
    cdc_mapping = get_cdc_reference(sheet_name="mapping")
    cdc_mapping = cdc_mapping[cdc_mapping["category"] == "mortality"]
    cdc_mapping.set_index("key", inplace=True)
    cdc_mapping = cdc_mapping["value"].to_dict()

    # get the column names
    byvariables = [
        var.get("code").split(".")[1] for var in root.findall(".//byvariables/variable")
    ]
    measure_selections = [
        measure.get("code").split(".")[1]
        for measure in root.find(".//measure-selections").findall(".//measure")
    ]
    columns = byvariables + measure_selections
    columns = [cdc_mapping.get(key, key) for key in columns]
    num_columns = len(columns)

    # create the data table
    rows = []

    # initialize row-span counts and values for each column
    row_span_counts = [0] * num_columns
    column_values = [None] * num_columns

    # iterate over the rows
    for row in data_table.findall("r"):
        row_cells = row.findall("c")
        cell_idx = 0
        cells = []

        for idx in range(num_columns):
            if row_span_counts[idx] > 0:
                row_span_counts[idx] -= 1
            else:  # noqa: PLR5501
                if cell_idx < len(row_cells):
                    cell = row_cells[cell_idx]
                    cell_idx += 1
                    if "r" in cell.attrib:
                        # row-spanning cell
                        value = cell.get("l") or cell.get("v") or cell.text
                        row_span_counts[idx] = int(cell.get("r")) - 1
                        column_values[idx] = value
                    else:
                        # regular cell
                        value = cell.get("v") or cell.get("l") or cell.text
                        column_values[idx] = value
                else:
                    # no more cells
                    column_values[idx] = None
            cells.append(column_values[idx])
        rows.append(cells)

    df = pd.DataFrame(rows, columns=columns)

    return df


def _parse_date_col(df: pd.DataFrame, col: str = "Month") -> pd.Series:
    """
    Parse the date column to a datetime object.

    CDC has the "Month" column as a string with the format "Month. (Year)"
    when grouping by month and year.
    This function parses the month column to a datetime object.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object.
    col : str
        Column name to parse dates.

    Returns
    -------
    parsed_dates : pd.Series
        Series of parsed dates.

    """
    parsed_dates = [date.split(" (")[0].strip() for date in df[col]]
    parsed_dates = [date.replace(".", "") for date in parsed_dates]
    parsed_dates = pd.to_datetime(parsed_dates, format="%b, %Y")

    return parsed_dates


def _infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer data types from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object.

    Returns
    -------
    df : pd.DataFrame
        DataFrame object with inferred data types.

    """
    comma_cols = [col for col in ["deaths", "population"] if col in df.columns]
    for col in comma_cols:
        if not hasattr(df[col], "str"):
            continue
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")

    float_cols = [col for col in [] if col in df.columns]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols = [col for col in ["year"] if col in df.columns]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    return df
