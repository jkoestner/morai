"""Tests the neural models."""

import pandas as pd
from pytest import approx

from morai.forecast import preprocessors
from morai.models import neural
from morai.utils import helpers

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
        cat_cols=[
            "gender",
            "faceband",
        ],
    )
    assert model.task == "poisson"

    # test fit
    model.fit(X=X, y=y, weights=weights, epochs=100, lr=0.001)
    assert set(model.embeddings.keys()) == {"gender", "faceband"}
    assert model.embeddings["gender"].num_embeddings == 3
    assert model.embeddings["gender"].embedding_dim == 2
    assert model.dropout1.p == 0.0
    assert model.dropout2.p == 0.0
    assert model.dropout3.p == 0.0

    # test predict
    predictions = model.predict(X)
    predictions_mean = helpers._weighted_mean(predictions, weights)
    y_mean = helpers._weighted_mean(y, weights)
    assert predictions_mean == approx(y_mean, abs=0.02), (
        "neural mean is off and should be close to .438"
    )


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
        cat_cols=[
            "gender",
            "faceband",
        ],
        embedding_dims={"gender": 5, "faceband": 10},
    )
    assert model.task == "binomial"

    # test fit
    model.fit(X=X, y=y, weights=weights, epochs=100, lr=0.001, dropout=0.1)
    assert set(model.embeddings.keys()) == {"gender", "faceband"}
    assert model.embeddings["gender"].num_embeddings == 3
    assert model.embeddings["gender"].embedding_dim == 5
    assert model.dropout1.p == 0.1
    assert model.dropout2.p == 0.1
    assert model.dropout3.p == 0.1

    # test predict
    predictions = model.predict(X)
    predictions_mean = helpers._weighted_mean(predictions, weights)
    y_mean = helpers._weighted_mean(y, weights)
    assert predictions_mean == approx(y_mean, abs=0.05), (
        "neural mean is off and should be close to .438"
    )
