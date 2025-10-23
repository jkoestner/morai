"""Tests the hmd module."""

from unittest.mock import MagicMock, patch

import pandas as pd

from morai.integrations import hmd
from morai.utils import helpers

test_sql_path = helpers.TESTS_PATH / "integrations" / "hmd" / "hmd.sql"


@patch("requests.Session")
def test_get_hmd_data(mock_session_class):
    """
    Tests getting hmd data using the hmd website.

    MagicMock is used to spoof the request to the hmd website and returning
    the sample hmd text file.
    """
    # read static xml
    txt_filepath = helpers.TESTS_PATH / "integrations" / "hmd" / "hmd.txt"
    with open(txt_filepath, "r") as file:
        hmd_txt = file.read()
    # mock the session
    mock_session = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_session
    # due to having 3 get requests, we need to mock them separately
    # 1st get request
    login_response = MagicMock()
    login_response.text = (
        '<html><input name="__RequestVerificationToken" value="mock_token"/></html>'
    )
    # 2nd get request
    logged_in_response = MagicMock()
    logged_in_response.text = "<html>Logout</html>"
    # 3rd get request
    data_response = MagicMock()
    data_response.text = hmd_txt
    data_response.raise_for_status = MagicMock()
    mock_session.get.side_effect = [login_response, logged_in_response, data_response]
    mock_session.post.return_value = MagicMock()

    # get the data
    hmd_df = hmd.get_hmd_data(
        email="test@example.com", password="test_password", clean_df=True
    )
    assert isinstance(hmd_df, pd.DataFrame)
    assert hmd_df.shape == (6, 6)
    assert hmd_df["year"].unique().tolist() == [1933, 1959, 2020]
    assert hmd_df["age"].unique().tolist() == [0, 1, 110]
    assert hmd_df["last_modified"].iloc[0] == pd.Timestamp("2025-05-08")


def test_get_hmd_data_sql():
    """Tests getting hmd data from sql."""
    hmd_df = hmd.get_hmd_data_sql(db_filepath=test_sql_path, table_name="hmd")
    assert isinstance(hmd_df, pd.DataFrame)
    assert hmd_df.shape == (100, 12)
