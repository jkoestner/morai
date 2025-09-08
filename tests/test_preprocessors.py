"""Tests the preprocessors."""

import numpy as np
import pandas as pd
import polars as pl

from morai.forecast import preprocessors


def test_preprocess_passthrough():
    """Tests the preprocess passthrough function."""
    # create df
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [1.1, 2.2, 3.3],
            "feature3": ["a", "b", "c"],
        }
    )
    feature_dict = {"passthrough": ["feature1", "feature2", "feature3"]}

    # preprocess
    preprocess_dict = preprocessors.preprocess_data(
        model_data=df, feature_dict=feature_dict
    )
    X = preprocess_dict["X"]

    # test values and names
    pd.testing.assert_frame_equal(
        X.reset_index(drop=True),
        df.reset_index(drop=True),
        check_names=True,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )


def test_preprocess_ordinal():
    """Tests the preprocess ordinal function."""
    # create df
    df = pd.DataFrame(
        {
            "ordinal1": ["low", "medium", "high", "medium", "low"],
            "ordinal2": ["cold", "warm", "hot", "warm", "cold"],
        }
    )
    feature_dict = {"ordinal": ["ordinal1", "ordinal2"]}

    # preprocess
    preprocess_dict = preprocessors.preprocess_data(
        model_data=df, feature_dict=feature_dict
    )
    X = preprocess_dict["X"]

    # expected dataset
    expected_df = pd.DataFrame(
        {
            "ordinal1": [1, 2, 0, 2, 1],
            "ordinal2": [0, 2, 1, 2, 0],
        }
    )

    # test values and names
    pd.testing.assert_frame_equal(
        X.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_names=True,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )


def test_preprocess_ohe():
    """Tests the preprocess ohe function."""
    # create df
    df = pd.DataFrame(
        {
            "ohe1": ["low", "medium", "high", "medium", "low"],
            "ohe2": ["cold", "warm", "hot", "warm", "cold"],
        }
    )
    feature_dict = {"ohe": ["ohe1", "ohe2"]}

    # preprocess
    preprocess_dict = preprocessors.preprocess_data(
        model_data=df, feature_dict=feature_dict
    )
    X = preprocess_dict["X"]

    # expected dataset
    expected_df = pd.DataFrame(
        {
            "ohe1_low": [1, 0, 0, 0, 1],
            "ohe1_medium": [0, 1, 0, 1, 0],
            "ohe2_hot": [0, 0, 1, 0, 0],
            "ohe2_warm": [0, 1, 0, 1, 0],
        }
    )

    # test values and names
    pd.testing.assert_frame_equal(
        X.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_names=True,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )


def test_preprocess_nominal():
    """Tests the preprocess nominal function."""
    # create df
    df = pd.DataFrame(
        {
            "nominal1": ["low", "medium", "high", "medium", "low"],
            "nominal2": ["cold", "warm", "hot", "warm", "cold"],
            "rate": [0.1, 0.2, 0.3, 0.2, 0.2],
        }
    )
    feature_dict = {"target": ["rate"], "nominal": ["nominal1", "nominal2"]}

    # preprocess
    preprocess_dict = preprocessors.preprocess_data(
        model_data=df, feature_dict=feature_dict
    )
    X = preprocess_dict["X"]

    # expected dataset
    expected_df = pd.DataFrame(
        {
            "nominal1": [0.15, 0.2, 0.3, 0.2, 0.15],
            "nominal2": [0.15, 0.2, 0.3, 0.2, 0.15],
        }
    )

    # test values and names
    pd.testing.assert_frame_equal(
        X.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_names=True,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )


def test_preprocess_standardize():
    """Tests the preprocess standardize function."""
    # create df
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [10, 20, 30],
        }
    )
    feature_dict = {"passthrough": ["feature1", "feature2"]}

    # preprocess
    preprocess_dict = preprocessors.preprocess_data(
        model_data=df, feature_dict=feature_dict, standardize=True
    )
    X = preprocess_dict["X"]

    # expected dataset
    expected_df = pd.DataFrame(
        {
            "feature1": [-1.22474487, 0, 1.22474487],
            "feature2": [-1.22474487, 0, 1.22474487],
        }
    )

    # test values and names
    pd.testing.assert_frame_equal(
        X.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_names=True,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )


def test_bin_feature():
    """Tests the bin feature function."""
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    binned = preprocessors.bin_feature(feature=s, bins=2)
    assert binned.unique().tolist() == ["01~05", "06~10"]


def test_lazy_bin_feature():
    """Tests the lazy bin feature function."""
    lf = pl.LazyFrame({"foo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    binned = preprocessors.lazy_bin_feature(lf=lf, feature="foo", bins=2)
    assert binned.collect()["foo_binned"].unique().to_list() == ["01~05", "06~10"]


def test_lazy_groupby():
    """Tests the lazy groupby function."""
    lf = pl.LazyFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "value1": [1, 2, 3, 4, 5],
            "value2": [10, 20, 30, 40, 50],
        }
    )
    grouped = preprocessors.lazy_groupby(
        df=lf, groupby_cols=["group"], agg_cols=["value1"], aggs="sum"
    )
    assert grouped.collect()["value1"].unique().to_list() == [3, 12]


def test_time_based_split():
    """Tests the time based split function."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.standard_normal(20),
            "feature2": rng.integers(0, 3, size=20),
            "cal_year": np.repeat([2020, 2021, 2022, 2023], 5),
        }
    )
    y = pd.Series(rng.random(20), name="target")
    w = pd.Series(rng.integers(1, 5, size=20), name="weights")

    X_train, X_test, y_train, y_test, w_train, w_test = preprocessors.time_based_split(
        df, y, w, time_col="cal_year", cutoff=2021, test_size=0.5, random_state=42
    )
    assert X_train.shape == (5, 3)
    assert X_test.shape == (15, 3)
    assert y_train.shape == (5,)
    assert y_test.shape == (15,)
    assert w_train.shape == (5,)
    assert w_test.shape == (15,)
    assert X_train["cal_year"].max() == 2021
