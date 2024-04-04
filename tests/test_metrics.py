"""Tests the metrics."""

import pandas as pd

from morai.forecast import metrics


def test_smape():
    """Tests the sMAPE calculation."""
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([2, 3, 4, 5, 6])
    epsilon = 1e-10

    test_smape = ((y_true - y_pred).abs() / ((y_true + y_pred) / 2 + epsilon)).mean()
    assert metrics.smape(y_true, y_pred, epsilon) == test_smape, "sMAPE not matching"


def test_ae():
    """Tests the AE calculation."""
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([2, 3, 4, 5, 6])

    test_ae = y_true.sum() / y_pred.sum()
    assert metrics.ae(y_true, y_pred) == test_ae, "ae not matching"
