"""Tests the neural models."""

import pandas as pd
import pytest
from pytest import approx

from morai.forecast import preprocessors
from morai.utils import helpers

torch = pytest.importorskip("torch", reason="torch required for neural tests")

from morai.models import neural  # noqa: E402

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast" / "models"
seed = 42


def test_neural_poisson():
    """Test the Neural network model - poisson."""
    # create model
    sigmoid_data = pd.read_csv(test_forecast_path / "sigmoid_data.csv")
    feature_dict = {
        "target": ["rate"],
        "weight": ["weight"],
        "passthrough": ["age", "gender", "faceband"],
    }
    preprocess_dict = preprocessors.preprocess_data(
        sigmoid_data,
        feature_dict=feature_dict,
        standardize=True,  # neural networks do better with standardization
        add_constant=False,
    )
    X = preprocess_dict["X"]
    y = preprocess_dict["y"]
    weights = preprocess_dict["weights"]

    # test setup
    model = neural.Neural(
        task="poisson",
    )
    assert model.task == "poisson"

    # test fit
    model.fit(
        X=X,
        y=y,
        weights=weights,
        X_test=X,
        y_test=y,
        weights_test=weights,
        epochs=100,
        lr=0.01,
    )
    assert model.dropout1.p == 0.0
    assert model.dropout2.p == 0.0
    assert model.dropout3.p == 0.0

    # test predict
    predictions = model.predict(X)
    predictions_mean = helpers._weighted_mean(predictions, weights)
    y_mean = helpers._weighted_mean(y, weights)
    assert predictions_mean == approx(y_mean, abs=0.02)


def test_neural_binomial():
    """
    Test the Neural network model - binomial.

    Additional tests:
      - embedding dimensions
      - dropout
    """
    # create model
    sigmoid_data = pd.read_csv(test_forecast_path / "sigmoid_data.csv")
    feature_dict = {
        "target": ["rate"],
        "weight": ["weight"],
        "passthrough": ["age", "gender", "faceband"],
    }
    preprocess_dict = preprocessors.preprocess_data(
        sigmoid_data,
        feature_dict=feature_dict,
        standardize=True,  # neural networks do better with standardization
        add_constant=False,
    )
    X = preprocess_dict["X"]
    y = preprocess_dict["y"]
    weights = preprocess_dict["weights"]

    # test setup
    model = neural.Neural(
        task="binomial",
    )
    assert model.task == "binomial"

    # test fit
    model.fit(
        X=X,
        y=y,
        weights=weights,
        X_test=X,
        y_test=y,
        weights_test=weights,
        epochs=100,
        lr=0.01,
        dropout=0.01,
    )
    assert model.dropout1.p == 0.01
    assert model.dropout2.p == 0.01
    assert model.dropout3.p == 0.01

    # test predict
    predictions = model.predict(X)
    predictions_mean = helpers._weighted_mean(predictions, weights)
    y_mean = helpers._weighted_mean(y, weights)
    assert predictions_mean == approx(y_mean, abs=0.2)
