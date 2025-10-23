"""Tests the cdc."""

import shutil
from unittest.mock import MagicMock, patch

import pandas as pd

from morai.integrations import cdc
from morai.utils import helpers

test_sql_path = helpers.TESTS_PATH / "integrations" / "cdc" / "cdc.sql"


@patch("requests.post")
def test_get_cdc_data_xml(mock_post, tmp_path):
    """
    Tests getting cdc data from xml.

    MagicMock is used to return the sample xml instead of making an api call.
    The xml can be retrieved using:

    ```
        url = f"https://wonder.cdc.gov/controller/datarequest/D176"
        response = requests.post(
            url,
            data={
                "request_xml": xml_request,
                "accept_datause_restrictions": "true",
            },
        )
    ```
    """
    # read static xml
    xml_filepath = helpers.TESTS_PATH / "integrations" / "cdc" / "cdc_d176.xml"
    with open(xml_filepath, "r") as file:
        xml_request = file.read()

    # create temporary files to mock the xml output and avoid api calls
    # cdc_d176 corresponds to mcd18_monthly
    test_xml_dir = tmp_path / "integrations" / "cdc" / "xml"
    test_xml_dir.mkdir(parents=True)
    test_xml_file = test_xml_dir / "cdc_d176.xml"
    test_xml_file.write_text(xml_request)
    reference_src = helpers.FILES_PATH / "integrations" / "cdc" / "cdc_reference.xlsx"
    reference_dst = tmp_path / "integrations" / "cdc" / "cdc_reference.xlsx"
    reference_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(reference_src, reference_dst)

    # mock the post request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = xml_request
    mock_post.return_value = mock_response

    # patch the files_path to use tmp_path
    with patch("morai.utils.helpers.FILES_PATH", tmp_path):
        df = cdc.get_cdc_data_xml(xml_filename="cdc_d176.xml", parse_date_col="Month")

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (93, 6)
    assert df.iloc[-1]["month"] == pd.Timestamp("2025-09-01")
    assert df.iloc[-1]["year"] == 2025


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
    """
    Tests getting the last updated date.

    Patched the files path to the tests path.
    """
    with patch("morai.utils.helpers.FILES_PATH", helpers.TESTS_PATH):
        last_updated = cdc.get_last_updated(table_name="mcd18_monthly")
    assert last_updated == "2025-10-14 23:13:03"


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


def test_calc_mi():
    """Tests calculating mortality improvement."""
    # create test data
    test_data = {
        "year": [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        "age_groups": [
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
            "45-54 years",
        ],
        "deaths": [
            187568,
            183207,
            183247,
            179463,
            177724,
            175917,
            174494,
            173516,
            170142,
            164837,
            160393,
        ],
        "population": [
            44867088,
            45006716,
            44718203,
            44268738,
            43767532,
            43458851,
            43188161,
            42786679,
            42374952,
            41631699,
            40874902,
        ],
    }

    df = pd.DataFrame(test_data)

    # calculate mortality improvement
    mi_df = cdc.calc_mi(df=df)

    # test calculations
    mi_2019 = mi_df.loc[mi_df["year"] == 2019, "crude_adj"].iloc[0]
    mi_2018 = mi_df.loc[mi_df["year"] == 2018, "crude_adj"].iloc[0]
    mi_1_year = 1 - (mi_2019 / mi_2018)
    mi_10_year = mi_df["1_year_mi"].rolling(window=10).mean().iloc[-1]
    assert isinstance(mi_df, pd.DataFrame)
    assert all(
        col in mi_df.columns
        for col in ["year", "crude_adj", "deaths", "1_year_mi", "10_year_mi", "whl_3"]
    )
    assert mi_df["1_year_mi"].iloc[-1] == mi_1_year
    assert mi_df["10_year_mi"].iloc[-1] == mi_10_year


def test_compare_df():
    """Tests the compare df function."""
    # create test data
    left_df = pd.DataFrame({"calendar": [2020, 2020, 2021], "fatalities": [1, 2, 3]})
    right_df = pd.DataFrame({"calendar": [2020, 2021, 2021], "fatalities": [1, 1, 1]})
    compare_col_dict = {"calendar": "calendar"}
    compare_value_dict = {"fatalities": "fatalities"}

    # expected result
    expected_df = pd.DataFrame(
        {
            "calendar": [2020, 2021],
            "fatalities_left": [3, 3],
            "fatalities_right": [1, 2],
            "fatalities_diff": [2, 1],
        }
    )

    # test compare_df
    result_df = cdc.compare_dfs(left_df, right_df, compare_col_dict, compare_value_dict)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)


def test_get_top_deaths_by_age_group():
    """Tests getting the top deaths by age group."""
    # create test data
    data = {
        "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023],
        "age_groups": [
            "< 1 year",
            "45-54 years",
            "85+ years",
            "< 1 year",
            "45-54 years",
            "85+ years",
            "< 1 year",
            "45-54 years",
            "85+ years",
        ],
        "deaths": [9000, 0, 0, 500, 9000, 10, 10, 500, 9000],
        "icd_sub_chapter": [
            "Birth trauma",
            "Birth trauma",
            "Birth trauma",
            "Malignant neoplasms",
            "Malignant neoplasms",
            "Malignant neoplasms",
            "Ischaemic heart diseases",
            "Ischaemic heart diseases",
            "Ischaemic heart diseases",
        ],
    }
    df = pd.DataFrame(data)

    top_deaths, top_names = cdc.get_top_deaths_by_age_group(df, year=2023)
    assert isinstance(top_deaths, pd.DataFrame)
    assert top_deaths.loc[1.0, "< 1 year"] == 9000
    assert top_deaths.loc[3.0, "< 1 year"] == 10
    assert top_deaths.loc[1.0, "45-54 years"] == 9000
    assert top_deaths.loc[1.0, "85+ years"] == 9000
    assert isinstance(top_names, pd.DataFrame)
    assert top_names.loc[1.0, "< 1 year"] == "perinatal"
    assert top_names.loc[3.0, "< 1 year"] == "circulatory"
    assert top_names.loc[1.0, "45-54 years"] == "neoplasms"
    assert top_names.loc[1.0, "85+ years"] == "circulatory"


def test_xml_parse_dataid():
    """Tests parsing the data-id from an XML string object."""
    xml_filepath = (
        helpers.FILES_PATH / "integrations" / "cdc" / "xml" / "mcd18_monthly.xml"
    )
    with open(xml_filepath, "r") as file:
        xml_request = file.read()

    data_id = cdc._xml_parse_dataid(xml_request)
    assert data_id == "D176"


def test_xml_create_df():
    """Tests creating a DataFrame from an XML string object."""
    xml_filepath = helpers.TESTS_PATH / "integrations" / "cdc" / "cdc_d176.xml"
    with open(xml_filepath, "r") as file:
        xml_request = file.read()

    df = cdc._xml_create_df(xml_request)
    assert isinstance(df, pd.DataFrame)
    assert all(
        col in df.columns
        for col in [
            "Year",
            "Month",
            "Deaths",
            "Population",
            "Crude Rate",
            "data_through",
        ]
    )
    assert df.iloc[0]["Deaths"] == "286,744"
    assert df.iloc[0]["Population"] == "Not Applicable"
    assert df.iloc[0]["Crude Rate"] == "Not Applicable"


def test_parse_date_col():
    """Tests parsing the date column to a datetime object."""
    df = pd.DataFrame({"Month": ["Jan., 2023", "Feb., 2023", "Mar., 2023"]})
    parsed = cdc._parse_date_col(df, col="Month")
    expected = pd.Series(
        [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-03-01"),
        ]
    )

    pd.testing.assert_series_equal(parsed, expected, check_dtype=False)


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
