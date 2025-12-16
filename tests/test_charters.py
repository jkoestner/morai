"""Tests the charters."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest
from pytest import approx
from sklearn import linear_model

from morai.experience import charters, experience
from morai.forecast import preprocessors
from morai.utils import helpers

# Test data
df = pd.DataFrame(
    {
        "x_axis": [1, 2, 3, 4, 5],
        "y_axis": [10, 20, 30, 40, 50],
        "color": ["red", "blue", "green", "red", "blue"],
        "numerator": [5, 10, 15, 20, 25],
        "denominator": [50, 100, 150, 200, 250],
    }
)
lazy_df = pl.DataFrame(df).lazy()
test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
experience_df = pd.read_csv(test_experience_path / "simple_experience.csv")
experience_df["actual"] = experience_df["rate"] * experience_df["exposed"]
lazy_experience_df = pl.DataFrame(experience_df).lazy()


def test_chart_line():
    """Tests the line chart."""
    # pandas
    fig = charters.chart(
        df, "x_axis", "y_axis", color="color", type="line", y_log=True, add_line=True
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == len(df["color"].unique()) + 1
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"
    assert fig.layout.yaxis.type == "log"
    assert fig.data[0].mode == "lines"
    # check if line was added
    assert fig.data[-1].y == (1, 1)

    # polars
    fig = charters.chart(
        lazy_df,
        "x_axis",
        "y_axis",
        color="color",
        type="line",
        y_log=True,
        add_line=True,
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == len(df["color"].unique()) + 1
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"
    assert fig.layout.yaxis.type == "log"
    assert fig.data[0].mode == "lines"
    # check if line was added
    assert fig.data[-1].y == (1, 1)


def test_chart_bar():
    """Tests the bar chart."""
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color="color", type="bar")
    assert isinstance(fig.data[0], go.Bar)
    assert len(fig.data) == len(df["color"].unique())
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color="color", type="bar")
    assert isinstance(fig.data[0], go.Bar)
    assert len(fig.data) == len(df["color"].unique())
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"


def test_chart_heatmap():
    """Tests the heatmap chart."""
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color="numerator", type="heatmap")
    assert isinstance(fig.data[0], go.Heatmap)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Heatmap of 'numerator' by 'x_axis' and 'y_axis'"

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color="numerator", type="heatmap")
    assert isinstance(fig.data[0], go.Heatmap)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Heatmap of 'numerator' by 'x_axis' and 'y_axis'"


def test_chart_histogram():
    """Tests the histogram chart."""
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color="color", type="histogram")
    assert isinstance(fig.data[0], go.Histogram)
    assert len(fig.data) == len(df["color"].unique())
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color="color", type="histogram")
    assert isinstance(fig.data[0], go.Histogram)
    assert len(fig.data) == len(df["color"].unique())
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"


def test_chart_area():
    """Tests the area chart."""
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color="color", type="area")
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == len(df["color"].unique())
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"
    assert fig.data[0].stackgroup == "1"

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color="color", type="area")
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == len(df["color"].unique())
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'color'"
    assert fig.data[0].stackgroup == "1"


def test_chart_contour():
    """Tests the contour chart."""
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color="numerator", type="contour")
    assert isinstance(fig.data[0], go.Contour)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Contour of 'numerator' by 'x_axis' and 'y_axis'"

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color="numerator", type="contour")
    assert isinstance(fig.data[0], go.Contour)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Contour of 'numerator' by 'x_axis' and 'y_axis'"


def test_chart_ratio():
    """Tests the ratio chart."""
    # pandas
    fig = charters.chart(
        df, "x_axis", y_axis="ratio", numerator="numerator", denominator="denominator"
    )
    ratio = df["numerator"] / df["denominator"]
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "'ratio' by 'x_axis' and 'None'"
    npt.assert_allclose(fig.data[0].y, ratio.values, atol=1e-3)

    # polars
    fig = charters.chart(
        lazy_df,
        "x_axis",
        y_axis="ratio",
        numerator="numerator",
        denominator="denominator",
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "'ratio' by 'x_axis' and 'None'"
    npt.assert_allclose(fig.data[0].y, ratio.values, atol=1e-3)


def test_chart_risk():
    """Tests the risk chart."""
    # pandas
    fig = charters.chart(
        df, "x_axis", y_axis="risk", numerator="numerator", denominator="denominator"
    )
    total_risk = df["numerator"].sum() / df["denominator"].sum()
    risk = (df["numerator"] / df["denominator"]) / total_risk
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "'risk' by 'x_axis' and 'None'"
    npt.assert_allclose(fig.data[0].y, risk.values, atol=1e-3)

    # polars
    fig = charters.chart(
        lazy_df,
        "x_axis",
        y_axis="risk",
        numerator="numerator",
        denominator="denominator",
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "'risk' by 'x_axis' and 'None'"
    npt.assert_allclose(fig.data[0].y, risk.values, atol=1e-3)


def test_chart_sort():
    """
    Tests the sorting of the chart.

    When using y_sort=True, the chart should be sorted by the y_axis greatest to least.
    """
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color=None, y_sort=True)
    y_sort_values = df.sort_values(by="y_axis", ascending=False)["y_axis"].values
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'None'"
    assert list(fig.data[0].y) == list(y_sort_values)

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color=None, y_sort=True)
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'None'"
    assert list(fig.data[0].y) == list(y_sort_values)


def test_chart_bins():
    """
    Tests the binning of the chart.

    When using x_bins, the chart should be binned by the x_axis in the
    number of bins specified.
    """
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color=None, x_bins=2)
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'None'"
    assert list(fig.data[0].x) == ["1~2", "3~5"]
    assert list(fig.data[0].y) == [30, 120]

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color=None, x_bins=2)
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.title.text == "'y_axis' by 'x_axis' and 'None'"
    assert list(fig.data[0].x) == ["1~2", "3~5"]
    assert list(fig.data[0].y) == [30, 120]


def test_chart_display():
    """
    Tests the display of the chart.

    When using display=False, the chart should be returned as a pandas DataFrame.
    """
    # pandas
    fig = charters.chart(df, "x_axis", "y_axis", color="color", display=False)
    assert isinstance(fig, pd.DataFrame)
    assert len(fig) == len(df)

    # polars
    fig = charters.chart(lazy_df, "x_axis", "y_axis", color="color", display=False)
    assert isinstance(fig, pd.DataFrame)
    assert len(fig) == len(df)


def test_chart_invalid_type():
    """Tests error when invalid chart type."""
    with pytest.raises(ValueError):
        charters.chart(df, "x_axis", "y_axis", color="color", type="invalid_type")


def test_chart_missing_color():
    """Tests error when missing color and when using heatmap."""
    with pytest.raises(ValueError):
        charters.chart(df, "x_axis", "y_axis", type="heatmap")


def test_relative_risk_aggregate():
    """Tests the relative risk functionality - aggregate."""
    # pandas
    fig = charters.relative_risk(
        df=experience_df,
        y_axis="rate",
        features=["sex"],
        display=False,
    )
    relative_risk_values = experience.calc_relative_risk(
        df=experience_df,
        features=["sex"],
        risk_col=["rate"],
    )["relative_risk"].unique()
    npt.assert_allclose(
        np.sort(fig["relative_risk"].values), np.sort(relative_risk_values), atol=1e-3
    )

    # polars
    fig = charters.relative_risk(
        df=lazy_experience_df,
        y_axis="rate",
        features=["sex"],
        display=False,
    )
    npt.assert_allclose(
        np.sort(fig["relative_risk"].values), np.sort(relative_risk_values), atol=1e-3
    )


def test_relative_risk_reference():
    """Tests the relative risk functionality - reference."""
    # pandas
    fig = charters.relative_risk(
        df=experience_df,
        y_axis="rate",
        features=["sex"],
        relative_to="reference",
        display=False,
    )
    relative_risk_values = experience.calc_relative_risk(
        df=experience_df,
        features=["sex"],
        risk_col=["rate"],
        relative_to="reference",
    )["relative_risk"].unique()
    npt.assert_allclose(
        np.sort(fig["relative_risk"].values), np.sort(relative_risk_values), atol=1e-3
    )

    # polars
    fig = charters.relative_risk(
        df=lazy_experience_df,
        y_axis="rate",
        features=["sex"],
        relative_to="reference",
        display=False,
    )
    npt.assert_allclose(
        np.sort(fig["relative_risk"].values), np.sort(relative_risk_values), atol=1e-3
    )


def test_relative_risk_flip():
    """Tests the relative risk functionality - aggregate."""
    # pandas - non-flip
    fig = charters.relative_risk(
        df=experience_df,
        y_axis="rate",
        features=["sex"],
        relative_cols=["smoker_status"],
    )
    name = experience_df["smoker_status"].unique()[0]
    assert fig.data[0].name == name

    # pandas - flip
    fig = charters.relative_risk(
        df=experience_df,
        y_axis="rate",
        features=["sex"],
        relative_cols=["smoker_status"],
        flip_x_color=True,
    )
    name = experience_df["sex"].unique()[0]
    assert fig.data[0].name == name


def test_compare_rates_data():
    """Tests the compare rates functionality."""
    expected_secondary = experience_df.groupby("smoker_status")["exposed"].sum()
    expected_rate_sex = experience_df.groupby("smoker_status").apply(
        lambda x: (x["sex_rate"] * x["exposed"]).sum() / x["exposed"].sum(),
        include_groups=False,
    )
    expected_rate_smoker = experience_df.groupby("smoker_status").apply(
        lambda x: (x["smoker_rate"] * x["exposed"]).sum() / x["exposed"].sum(),
        include_groups=False,
    )

    # pandas
    fig = charters.compare_rates(
        df=experience_df,
        x_axis="smoker_status",
        rates=["sex_rate", "smoker_rate"],
        weights=["exposed"],
        secondary="exposed",
        display=False,
    )
    assert np.allclose(fig["exposed"].values, expected_secondary.values)
    assert np.allclose(fig["sex_rate"].values, expected_rate_sex.values)
    assert np.allclose(fig["smoker_rate"].values, expected_rate_smoker.values)

    # polars
    fig = charters.compare_rates(
        df=lazy_experience_df,
        x_axis="smoker_status",
        rates=["sex_rate", "smoker_rate"],
        weights=["exposed"],
        secondary="exposed",
        display=False,
    )
    assert np.allclose(fig["exposed"].values, expected_secondary.values)
    assert np.allclose(fig["sex_rate"].values, expected_rate_sex.values)
    assert np.allclose(fig["smoker_rate"].values, expected_rate_smoker.values)


def test_compare_rates_chart():
    """Tests the compare rates chart functionality."""
    expected_plots = (
        experience_df["sex"].nunique() + experience_df["smoker_status"].nunique()
    )

    # pandas
    fig = charters.compare_rates(
        df=experience_df,
        x_axis="smoker_status",
        rates=["sex_rate", "smoker_rate"],
        line_feature="sex",
        weights=["exposed"],
        y_log=True,
        display=True,
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.yaxis.type == "log"
    assert len(fig.data) == expected_plots

    # polars
    fig = charters.compare_rates(
        df=lazy_experience_df,
        x_axis="smoker_status",
        rates=["sex_rate", "smoker_rate"],
        line_feature="sex",
        weights=["exposed"],
        y_log=True,
        display=True,
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.yaxis.type == "log"
    assert len(fig.data) == expected_plots


def test_frequency():
    """Tests the frequency functionality."""
    expected_sex_sum = experience_df.groupby("sex")["exposed"].sum()

    # pandas
    fig = charters.frequency(
        df=experience_df,
        cols=3,
        features=["sex", "smoker_status", "year"],
        sum_var="exposed",
    )
    assert isinstance(fig.data[0], go.Bar)
    assert len(fig.data) == 3
    assert np.allclose(fig.data[0].y, expected_sex_sum.values)

    # polars
    fig = charters.frequency(
        df=lazy_experience_df,
        cols=3,
        features=["sex", "smoker_status", "year"],
        sum_var="exposed",
    )
    assert isinstance(fig.data[0], go.Bar)
    assert len(fig.data) == 3
    assert np.allclose(fig.data[0].y, expected_sex_sum.values)


def test_pdp():
    """Tests the pdp functionality."""
    # expected df
    expected_df = pd.DataFrame(
        {
            "sex": ["F", "F", "M", "M"],
            "smoker_status": ["NS", "S", "NS", "S"],
            "pred": [0.07, 0.09, 0.11, 0.13],
            "%_diff": [0.7, 0.9, 1.1, 1.3],
            "exposed": [100, 100, 200, 200],
        }
    )

    # setup data
    feature_dict = {
        "target": ["rate"],
        "passthrough": ["year"],
        "ordinal": [
            "sex",
            "smoker_status",
        ],
        "ohe": [],
        "nominal": [],
    }
    preprocess_dict = preprocessors.preprocess_data(
        experience_df,
        feature_dict=feature_dict,
        standardize=False,
    )
    X = preprocess_dict["X"]
    y = preprocess_dict["y"]
    weights = preprocess_dict["weights"]
    mapping = preprocess_dict["mapping"]
    md_encoded = preprocess_dict["md_encoded"]

    # setup model
    model = linear_model.LinearRegression()
    model.fit(X=X, y=y)

    # test pdp
    pdp = charters.pdp(
        model=model,
        df=md_encoded,
        x_axis="sex",
        line_color="smoker_status",
        weight=weights,
        mapping=mapping,
        secondary="exposed",
        center="global",
        display=False,
    )
    pd.testing.assert_frame_equal(
        pdp.sort_values(by=["sex", "smoker_status"]).reset_index(drop=True),
        expected_df.sort_values(by=["sex", "smoker_status"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_scatter():
    """Tests the scatter functionality."""
    # pandas
    fig = charters.scatter(
        experience_df,
        target="rate",
        features=["sex", "smoker_status", "year"],
        sample_nbr=None,
        cols=3,
    )
    assert isinstance(fig.data[0], go.Scattergl)
    assert len(fig.data) == 3
    assert len(fig.data[0].y) == len(experience_df)

    fig = charters.scatter(
        experience_df,
        target="rate",
        features=["sex", "smoker_status", "year"],
        sample_nbr=5,
        cols=3,
    )
    assert len(fig.data[0].y) == 5


def test_matrix():
    """Tests the matrix functionality."""
    matrix_df = pd.DataFrame(
        [[1.0, 0.8, 0.3], [0.8, 1.0, 0.6], [0.3, 0.6, 1.0]],
        columns=list("ABC"),
        index=list("ABC"),
    )
    expected_z = np.array(
        [[np.nan, np.nan, np.nan], [0.8, np.nan, np.nan], [0.3, 0.6, np.nan]]
    )

    # pandas
    fig = charters.matrix(matrix_df, threshold=0.5)
    assert isinstance(fig.data[0], go.Heatmap)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Matrix Heatmap"
    # bottom left triangle will be populated while top right will be empty
    assert np.allclose(fig.data[0].z, expected_z, equal_nan=True)
    # values should be red if values are above threshold or below -threshold
    assert fig.layout.annotations[0].font.color == "red"
    assert fig.layout.annotations[1].font.color is None
    assert fig.layout.annotations[2].font.color == "red"


def test_target():
    """Tests the target functionality."""
    expected_target_values = (
        experience_df.groupby("smoker_status", observed=True)[
            ["rate", "exposed"]
        ].apply(lambda x: helpers._weighted_mean(x.iloc[:, 0], weights=x.iloc[:, 1]))
    ).values

    # pandas
    fig = charters.target(
        experience_df,
        target="rate",
        features=["sex", "smoker_status", "year"],
        cols=3,
        weights="exposed",
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 4
    assert np.allclose(fig.data[1].y, expected_target_values)

    # polars
    fig = charters.target(
        lazy_experience_df,
        target="rate",
        features=["sex", "smoker_status", "year"],
        cols=3,
        weights="exposed",
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 4
    assert np.allclose(fig.data[1].y, expected_target_values)


def test_target_risk():
    """Tests the target functionality - risk ratio."""
    relative_risk_values = experience.calc_relative_risk(
        df=experience_df,
        features=["sex"],
        risk_col=["rate"],
        weight_col=["exposed"],
    )["relative_risk"].unique()

    # pandas
    fig = charters.target(
        experience_df,
        target="risk",
        features=["sex", "smoker_status", "year"],
        numerator="actual",
        denominator="exposed",
        cols=3,
        add_line=True,
    )
    assert len(fig.data) == 8
    assert np.allclose(fig.data[0].y, relative_risk_values)

    # polars
    fig = charters.target(
        lazy_experience_df,
        target="risk",
        features=["sex", "smoker_status", "year"],
        numerator="actual",
        denominator="exposed",
        cols=3,
        add_line=True,
    )
    assert len(fig.data) == 8
    assert np.allclose(fig.data[0].y, relative_risk_values)


def test_target_generate_pairwise():
    """Tests the target functionality - generate pairwise."""
    expected_target_values = (
        experience_df[experience_df["sex"] == "F"]
        .groupby("smoker_status", observed=True)[["rate"]]
        .mean()
    ).values.flatten()

    # pandas
    fig = charters.target(
        experience_df,
        target="rate",
        features=["sex", "smoker_status", "year"],
        cols=3,
        weights="exposed",
        generate_pairwise=True,
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 4
    assert np.allclose(fig.data[0].y, expected_target_values)

    # polars
    fig = charters.target(
        lazy_experience_df,
        target="rate",
        features=["sex", "smoker_status", "year"],
        cols=3,
        weights="exposed",
        generate_pairwise=True,
    )
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data) == 4
    assert np.allclose(fig.data[0].y, expected_target_values)


def test_get_stats():
    """
    Tests the get_stats functionality.

    Only provides stats for numeric columns.
    """
    # pandas
    stats = charters.get_stats(experience_df, features=["sex", "smoker_status", "year"])
    assert isinstance(stats, pd.DataFrame)
    assert stats["feature"].unique() == ["year"]
    assert (
        all(col in stats.columns for col in ["null_count", "null_pct", "zero_pct"])
        == True
    )

    # polars
    stats = charters.get_stats(lazy_experience_df, features=["year"])
    assert isinstance(stats, pd.DataFrame)
    assert stats["feature"].unique() == ["year"]
    assert (
        all(col in stats.columns for col in ["null_count", "null_pct", "zero_pct"])
        == True
    )


def test_get_category_orders():
    """Tests the get_category_orders functionality."""
    expected_category_orders = list(
        experience_df.groupby("sex")["actual"]
        .agg("sum")
        .sort_values(ascending=False)
        .index.values
    )

    # pandas
    category_orders = charters.get_category_orders(
        df=experience_df, category="sex", measure="rate"
    )
    assert isinstance(category_orders, dict)
    assert category_orders["sex"] == expected_category_orders
