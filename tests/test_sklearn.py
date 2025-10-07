"""
Tests the sklearn models to test if they are working as expected.

These tests are meant to ensure the models are producing
results that are similar to what has been provided in the past. Libraries
may change logic so it's helpful to know if these results are changing.
"""

import pandas as pd
from pytest import approx
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from morai.forecast import preprocessors
from morai.utils import helpers

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast" / "models"
simple_data = pd.read_csv(test_forecast_path / "simple_data.csv")
seed = 42

# preprocess metadata for models
feature_dict = {
    "target": ["rate"],
    "weight": [],
    "passthrough": ["year", "gender"],
}
preprocess_dict = preprocessors.preprocess_data(
    simple_data,
    feature_dict=feature_dict,
    standardize=False,
)
X = preprocess_dict["X"]
y = preprocess_dict["y"]
weights = preprocess_dict["weights"]


def test_tree():
    """Test the Decision Tree."""
    clf = DecisionTreeRegressor(max_depth=6)
    clf.fit(X, y, sample_weight=weights)
    predictions = clf.predict(X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), "decision tree mean is off"
    assert predictions[0] == approx(0.1000, abs=1e-4), (
        "decision tree first value is off"
    )


def test_rf():
    """
    Test the Random Forest.

    https://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.RandomForestRegressor.html
    """
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
        oob_score=True,
        random_state=seed,
    )
    rf_fit = rf.fit(X, y, sample_weight=weights)
    predictions = rf_fit.predict(X)

    assert predictions.mean() == approx(0.0523, abs=1e-4), "random forest mean is off"
    assert predictions[0] == approx(0.0645, abs=1e-4), (
        "random forest first value is off"
    )


def test_lr():
    """Test the Linear Regression."""
    clf = linear_model.LinearRegression()
    clf.fit(X, y, sample_weight=weights)
    predictions = clf.predict(X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), (
        "linear regression mean is off"
    )
    assert predictions[0] == approx(0.0775, abs=1e-4), (
        "linear regression first value is off"
    )
