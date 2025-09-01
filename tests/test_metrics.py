"""Tests the metrics."""

import pandas as pd
from pytest import approx

from morai.forecast import metrics
from morai.utils import helpers

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast"
metric_df = pd.read_csv(test_forecast_path / "metrics" / "metric_data.csv")


def test_smape():
    """Tests the sMAPE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]
    epsilon = 1e-10

    test_smape = ((y_true - y_pred).abs() / ((y_true + y_pred) / 2 + epsilon)).mean()
    assert metrics.smape(y_true, y_pred, epsilon) == test_smape, "sMAPE not matching"


def test_smape_weighted():
    """Tests the sMAPE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]
    epsilon = 1e-10
    sample_weight = metric_df["weight"]
    test_smape = (
        (y_true - y_pred).abs() / ((y_true + y_pred) / 2 + epsilon)
    ) * sample_weight
    test_smape = test_smape.sum() / sample_weight.sum()

    assert metrics.smape(y_true, y_pred, epsilon, sample_weight) == test_smape, (
        "sMAPE weighted not matching"
    )


def test_mape():
    """Tests the MAPE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]
    test_mape = ((y_true - y_pred).abs() / y_true.abs()).mean()
    mape = metrics.calculate_metrics(
        y_true, y_pred, metrics=["mean_absolute_percentage_error"]
    )["_mean_absolute_percentage_error"]
    assert mape == test_mape, "MAPE not matching"


def test_ae():
    """Tests the AE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    test_ae = y_true.sum() / y_pred.sum()
    assert metrics.ae(y_true, y_pred) == test_ae, "ae not matching"


def test_mse():
    """Tests the MSE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    test_mse = ((y_true - y_pred) ** 2).mean()
    mse = metrics.calculate_metrics(y_true, y_pred, metrics=["mean_squared_error"])[
        "_mean_squared_error"
    ]
    assert mse == test_mse, "MSE not matching"


def test_mae():
    """Tests the MAE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    test_mae = ((y_true - y_pred).abs()).mean()
    mae = metrics.calculate_metrics(y_true, y_pred, metrics=["mean_absolute_error"])[
        "_mean_absolute_error"
    ]
    assert mae == test_mae, "MAE not matching"


def test_r2():
    """Tests the R2 calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    rss = ((y_true - y_pred) ** 2).sum()
    tss = ((y_true - y_true.mean()) ** 2).sum()

    test_r2 = 1 - rss / tss
    r2 = metrics.calculate_metrics(y_true, y_pred, metrics=["r2_score"])["_r2_score"]
    assert r2 == test_r2, "R2 not matching"
