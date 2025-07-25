"""Tests the preprocessors."""

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
        df=lf, groupby_cols=["group"], agg_cols=["value1"], agg="sum"
    )
    assert grouped.collect()["value1"].unique().to_list() == [3, 12]
