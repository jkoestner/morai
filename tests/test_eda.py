"""Tests the eda."""

import pandas as pd
from pytest import approx

from morai.experience import eda


def test_correlation():
    """Tests the correlation function."""
    df_numeric = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [1, 1, 2, 9, 11],
        }
    )
    df_categorical = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "blue", "red"],
            "shape": ["circle", "square", "circle", "square", "circle"],
        }
    )
    pearson_corr = eda.correlation(df_numeric)
    kendall_corr = eda.correlation(df_numeric, method="kendall")
    spearman_corr = eda.correlation(df_numeric, method="spearman")
    cramers_v_corr = eda.correlation(df_categorical, numeric=False)
    assert pearson_corr["x"]["y"] == approx(0.92, abs=0.01)
    assert kendall_corr["x"]["y"] == approx(0.95, abs=0.01)
    assert spearman_corr["x"]["y"] == approx(0.98, abs=0.01)
    assert cramers_v_corr["shape"]["color"] == approx(1.0, abs=0.01)


def test_mutual_info():
    """Tests the mutual_info function."""
    df_mixed = pd.DataFrame(
        {
            "x": [1, 1, 1, 2, 2],
            "y": [1, 1, 2, 9, 11],
            "color": ["red", "red", "red", "blue", "blue"],
        }
    )
    mutual_info = eda.mutual_info(df_mixed, random_state=42)
    assert mutual_info["x"]["y"] == 0
    assert mutual_info["x"]["color"] == approx(0.15, abs=0.01)


def test_cramers_v():
    """Tests the cramers_v function."""
    s1 = pd.Series(["A", "A", "B", "B"])
    s2 = pd.Series(["X", "X", "Y", "Y"])
    confusion = pd.crosstab(s1, s2)
    cramers_v = eda.cramers_v(confusion)
    assert cramers_v == 1.0


def test_gvif():
    """Tests the gvif function."""
    # numeric only
    df_num = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # perfectly correlated with x1
            "x3": [5, 3, 6, 2, 1],
        }
    )
    gvif_num = eda.gvif(df_num)
    # mixed
    df_mix = pd.DataFrame({"num": [1, 2, 3, 4], "cat": ["A", "B", "A", "B"]})
    gvif_mix = eda.gvif(df_mix, numeric_only=False)
    assert gvif_num.iloc[0]["gvif"] > 5
    assert gvif_mix.iloc[0]["gvif"] == approx(1.25, abs=0.01)
