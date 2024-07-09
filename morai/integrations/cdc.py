"""
CDC Data.

This module contains functions to work with CDC data. The CDC data is a
collection of data from the CDC Wonder database.

Link: https://wonder.cdc.gov/

If wanting to further develop integration the following repository is a good
reference.

Link: https://github.com/diegomtzg/CDC-WonderPy
"""

import xml.etree.ElementTree as ET

import pandas as pd
import requests

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def get_cdc_data(xml_filename):
    """
    Get CDC data from an XML file.

    Parameters
    ----------
    xml_filename : str
        Path to the XML file that contains the request parameters.

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
        logger.error(f"Error: {response.status_code}")

    # create the dataframe from the response
    logger.debug("creating dataframe from response")
    cdc_df = _xml_create_df(xml_response=xml_response)

    return cdc_df


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

    # create the data table
    rows = []
    for row in data_table.findall("r"):
        cells = []
        for cell in row.findall("c"):
            # Get the value of the cell
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

    return df
