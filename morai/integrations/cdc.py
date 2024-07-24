"""
CDC Data.

This module contains functions to work with CDC data. The CDC data is a
collection of data from the CDC Wonder database.

Link: https://wonder.cdc.gov/

If wanting to further develop integration the following repository is a good
reference.

Link: https://github.com/alipphardt/cdc-wonder-api
"""

import xml.etree.ElementTree as ET

import pandas as pd
import requests

from morai.utils import custom_logger, helpers, sql
from morai.utils.custom_logger import suppress_logs
from morai.utils.helpers import check_merge

logger = custom_logger.setup_logging(__name__)


def get_cdc_data_xml(
    xml_filename, parse_date_col=None, convert_dtypes=True, clean_df=True
):
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
    xml_filepath = helpers.FILES_PATH / "integrations" / "cdc" / xml_filename
    if not xml_filepath.exists():
        logger.error(f"File not found: {xml_filepath}")
        return None
    with open(xml_filepath, "r") as file:
        xml_request = file.read()

    # query cdc wonder
    data_id = _xml_parse_dataid(xml_request)
    url = f"https://wonder.cdc.gov/controller/datarequest/{data_id}"
    logger.debug(f"requesting data from CDC Wonder: {url}")
    response = requests.post(
        url, data={"request_xml": xml_request, "accept_datause_restrictions": "true"}
    )

    if response.status_code == 200:
        xml_response = response.text
    else:
        raise Exception(f"Error: {response.status_code}")

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


def get_cdc_data_sql(db_filepath, table_name):
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
            WHERE added_at = '{last_date}'
            """
    cdc_df = sql.read_sql(db_filepath, query)
    logger.info(f"table `{table_name}` last updated at: {last_date}")

    return cdc_df


def get_cdc_reference(sheet_name):
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


def map_reference(df, col, index_dict=None, sheet_name="cod"):
    """
    Map a column from the CDC reference to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to map the column onto.
    col : str
        The column to map.
    index_dict : dict, optional
        Dictionary of the indices to match.
        If not provided, the default is {"icd-10_113_cause_list": "wonder_cause"}.
    sheet_name : str, optional
        The sheet name to use.

    """
    logger.info(f"mapping column: {col}")
    if index_dict is None:
        index_dict = {"icd-10_113_cause_list": "cause_wonder"}
    mapping = get_cdc_reference(sheet_name=sheet_name)
    df_idx = next(iter(index_dict.keys()))
    reference_idx = next(iter(index_dict.values()))
    mapping = mapping[[reference_idx, col]]
    mapping = mapping.drop_duplicates()

    df = check_merge(pd.merge)(
        left=df,
        right=mapping[[reference_idx, col]],
        how="left",
        left_on=df_idx,
        right_on=reference_idx,
    )

    if reference_idx != df_idx and reference_idx in df.columns:
        df = df.drop(columns=[reference_idx])

    return df


def _xml_parse_dataid(xml_string):
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


def _xml_create_df(xml_response):
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

    # create the data table
    rows = []

    # row-spanning cells
    row_span_value = None
    row_span_count = 0

    # iterate over the rows
    for row in data_table.findall("r"):
        cells = []
        if row_span_count > 0:
            cells.append(row_span_value)
            row_span_count -= 1
        for cell in row.findall("c"):
            if "r" in cell.attrib:
                row_span_value = cell.get("l")
                row_span_count = int(cell.get("r")) - 1
                cells.append(row_span_value)
            else:
                cell_value = (
                    cell.get("v")
                    or cell.get("dt")
                    or cell.get("l")
                    or cell.get("cd")
                    or cell.get("cf")
                    or cell.get("c")
                )
                cells.append(cell_value)
        rows.append(cells)

    df = pd.DataFrame(rows)
    df.columns = columns

    return df


def _parse_date_col(df, col="Month"):
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


def _infer_dtypes(df):
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
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")

    float_cols = [col for col in [] if col in df.columns]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols = [col for col in ["year"] if col in df.columns]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    return df
