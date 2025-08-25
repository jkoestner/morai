"""Tests the preprocessors."""

import numpy as np
import pandas as pd
import polars as pl

from morai.forecast import preprocessors


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
