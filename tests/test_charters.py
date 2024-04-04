"""Tests the charters."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from morai.experience import charters

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


def test_chart_line():
    """Tests the line chart."""
    fig = charters.chart(df, "x_axis", "y_axis", color="color", type="line")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # three lines (red, blue, and green)
    assert fig.layout.title.text == "y_axis by x_axis and color"


def test_chart_bar():
    """Tests the bar chart."""
    fig = charters.chart(df, "x_axis", "y_axis", color="color", type="bar")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # three lines (red, blue, and green)
    assert fig.layout.title.text == "y_axis by x_axis and color"


def test_chart_heatmap():
    """Tests the heatmap chart."""
    fig = charters.chart(df, "x_axis", "y_axis", color="color", type="heatmap")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Heatmap of color by x_axis and y_axis"


def test_chart_ratio():
    """Tests the ratio chart."""
    fig = charters.chart(
        df, "x_axis", y_axis="ratio", numerator="numerator", denominator="denominator"
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # One line for the ratio
    assert fig.layout.title.text == "ratio by x_axis and None"


def test_chart_sort():
    """Tests the sorting of the chart."""
    fig = charters.chart(df, "x_axis", "y_axis", color=None, sort_y=True)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "y_axis by x_axis and None"
    # assert fig.data[0].y == [50, 40, 30, 20, 10]


def test_chart_bins():
    """Tests the binning of the chart."""
    fig = charters.chart(df, "x_axis", "y_axis", color=None, x_bins=2)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "y_axis by x_axis and None"
    # assert fig.data[0].x == [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]


def test_chart_display():
    """Tests the display of the chart."""
    fig = charters.chart(df, "x_axis", "y_axis", color="color", display=False)
    assert isinstance(fig, pd.DataFrame)
    assert len(fig) == 5


def test_chart_invalid_type():
    """Tests error when invalid chart type."""
    with pytest.raises(ValueError):
        charters.chart(df, "x_axis", "y_axis", color="color", type="invalid_type")


def test_chart_missing_color():
    """Tests error when missing color and when using heatmap."""
    with pytest.raises(ValueError):
        charters.chart(df, "x_axis", "y_axis", type="heatmap")
