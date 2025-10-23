"""Tests the helper utils."""

import pandas as pd
from pytest import approx

from morai.utils import helpers


def test_clean_df():
    """Tests the clean_df function for a DataFrame."""
    df = pd.DataFrame(
        {
            "UPPERCASED": [1, 2, 3],
            "spaced COLUMN": [4, 5, 6],
            "unused categories": pd.Categorical(
                ["a", "b", "a"],  # actual values
                categories=["a", "b", "c", "d"],  # define unused categories too
            ),
        }
    )
    clean_df = helpers.clean_df(df)
    assert clean_df.columns.tolist() == [
        "uppercased",
        "spaced_column",
        "unused_categories",
    ], "Column names not cleaned correctly"
    assert clean_df["unused_categories"].cat.categories.tolist() == ["a", "b"], (
        "Unused categories not removed"
    )


def test_clean_df_dict():
    """Tests the clean_df function for a dictionary."""
    test_dict = {
        "dummy": {
            "values": {
                "val1": "UPPERCASE",
                "val2": "spaced column",
                "val3": "MixedCase",
            }
        },
    }
    clean_dict = helpers.clean_df(test_dict)
    assert clean_dict == {
        "dummy": {
            "values": {
                "val1": "uppercase",
                "val2": "spaced_column",
                "val3": "mixedcase",
            }
        },
    }


def test_weighted_mean():
    """Tests the weighted_mean function."""
    values = [1, 2, 3]
    weights = [0.1, 0.2, 0.7]
    assert helpers._weighted_mean(values, weights) == approx(2.6, abs=1e-4)


def test_convert_object_to_category():
    """Tests the convert_object_to_category function."""
    df = pd.DataFrame({"cat_column": ["a", "b", "a"]})
    df = helpers._convert_object_to_category(df, "cat_column")
    assert df["cat_column"].dtype == "category"


def test_to_list():
    """Tests the to_list function."""
    # string
    assert helpers._to_list("a") == ["a"]
    # dict
    assert helpers._to_list({"a": 1, "b": 2}) == ["a", "b"]
    # list
    assert helpers._to_list(["a", "b", "c"]) == ["a", "b", "c"]
    # None
    assert helpers._to_list(None) == []


def test_check_merge_common_column(caplog):
    """Tests the check_merge function for common columns."""
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3, 1], "b": [1, 2, 3, 4]})
    with caplog.at_level("WARNING", logger="morai.utils.helpers"):
        helpers.check_merge(pd.merge)(left=df1, right=df2, how="left", on="a")
    assert any("common columns" in msg for msg in caplog.messages)


def test_check_merge_x_to_many(caplog):
    """Tests the check_merge function for x-to-many relationship."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3, 1], "b": [1, 2, 3, 4]})
    with caplog.at_level("WARNING", logger="morai.utils.helpers"):
        helpers.check_merge(pd.merge)(left=df1, right=df2, how="left", on="a")
    assert any("multiple values" in msg for msg in caplog.messages)
