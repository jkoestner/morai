"""Tests the cdc."""

from unittest.mock import patch

import pandas as pd

from morai.integrations import cdc
from morai.utils import helpers

test_sql_path = helpers.TESTS_PATH / "integrations" / "cdc" / "cdc.sql"


def test_get_cdc_data_txt():
    """
    Tests getting cdc data from txt.

    Patched the files path to the tests path.
    """
    with patch("morai.utils.helpers.FILES_PATH", helpers.TESTS_PATH):
        df = cdc.get_cdc_data_txt(
            txt_filename="mcd99_mi_q1.txt", convert_dtypes=True, clean_df=True
        )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (255, 9)
    assert df.columns.str.islower().all()
    assert df.columns.str.contains(" ").sum() == 0
    assert "notes" not in df.columns
    assert df["year"].dtype == "int16"
    assert df["deaths"].dtype == "float64"
    assert df["population"].dtype == "float64"


def test_get_cdc_data_sql():
    """Tests getting cdc data from sql."""
    df = cdc.get_cdc_data_sql(db_filepath=test_sql_path, table_name="mcd18_monthly")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (92, 7)


def test_get_last_updated():
    """Tests getting the last updated date."""
    assert cdc.get_last_updated(table_name="mcd18_monthly") == "2025-08-17 00:41:38"


def test_get_cdc_reference():
    """Tests getting the cdc reference."""
    df = cdc.get_cdc_reference(sheet_name="mapping")
    assert isinstance(df, pd.DataFrame)


def test_map_reference():
    """Tests mapping a column from the CDC reference to the DataFrame."""
    # create test data
    test_data = {
        "year": [2021, 2021, 2021, 2021],
        "icd_sub_chapter": [
            "Intestinal infectious diseases",
            "Tuberculosis",
            "Other bacterial diseases",
            "Viral infections of the central nervous system",
        ],
        "age_groups": ["45-54 years", "45-54 years", "45-54 years", "45-54 years"],
        "deaths": [266, 52, 2718, 43],
    }
    df = pd.DataFrame(test_data)

    # map the reference
    mapped_df = cdc.map_reference(
        df=df, col="simple_grouping", on_dict={"icd_sub_chapter": "wonder_sub_chapter"}
    )
    assert isinstance(mapped_df, pd.DataFrame)
    assert mapped_df["simple_grouping"].unique().tolist() == ["infectious"]


def test_xml_parse_dataid():
    """Tests parsing the data-id from an XML string object."""
    xml_filepath = (
        helpers.FILES_PATH / "integrations" / "cdc" / "xml" / "mcd18_monthly.xml"
    )
    with open(xml_filepath, "r") as file:
        xml_request = file.read()

    data_id = cdc._xml_parse_dataid(xml_request)
    assert data_id == "D176"


def test_infer_dtypes():
    """Tests inferring the data types from a DataFrame."""
    # create test data
    test_data = {
        "year": ["1979", "1980", "1981"],
        "deaths": ["19685", "19722", "18853"],
        "population": ["1,703,131.00", "1,759,642.00", "1,768,966.00"],
    }
    df = pd.DataFrame(test_data)
    df = df.astype(str)

    # infer the data types
    df = cdc._infer_dtypes(df)
    assert pd.api.types.is_numeric_dtype(df["year"])
    assert pd.api.types.is_numeric_dtype(df["deaths"])
    assert pd.api.types.is_numeric_dtype(df["population"])
