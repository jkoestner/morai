"""
Tests the experience.

These tests are meant to ensure the models are producing
results that are similar to what has been provided in the past. Libraries
may change logic so it's helpful to know if these results are changing.
"""

import lightgbm as lgb
import pandas as pd
from pytest import approx
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from morai.forecast import models, preprocessors
from morai.utils import helpers

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast"
simple_data = pd.read_csv(test_forecast_path / "simple_data.csv")

# preprocess the data
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


def test_glm():
    """Test the Generalized Linear Model."""
    GLM = models.GLM(X, y, weights)
    GLM.fit_model()
    predictions = GLM.model.predict(GLM.X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), "glm mean is off"
    assert predictions[0] == approx(0.0815, abs=1e-4), "glm first value is off"


def test_lr():
    """Test the Linear Regression."""
    clf = linear_model.LinearRegression()
    clf.fit(X, y, sample_weight=weights)
    predictions = clf.predict(X)

    assert predictions.mean() == approx(
        0.0525, abs=1e-4
    ), "linear regression mean is off"
    assert predictions[0] == approx(
        0.0775, abs=1e-4
    ), "linear regression first value is off"


def test_tree():
    """Test the Decision Tree."""
    clf = DecisionTreeRegressor(max_depth=6)
    clf.fit(X, y, sample_weight=weights)
    predictions = clf.predict(X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), "decision tree mean is off"
    assert predictions[0] == approx(
        0.1000, abs=1e-4
    ), "decision tree first value is off"


def test_rf():
    """Test the Random Forest."""
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
        oob_score=True,
        random_state=42,
    )
    rf_fit = rf.fit(X, y, sample_weight=weights)
    predictions = rf_fit.predict(X)

    assert predictions.mean() == approx(0.0523, abs=1e-4), "random forest mean is off"
    assert predictions[0] == approx(
        0.0645, abs=1e-4
    ), "random forest first value is off"


def test_lgb():
    """Test the LightGBM."""
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }
    train_data = lgb.Dataset(X, label=y, weight=weights, categorical_feature="auto")
    bst = lgb.train(params, train_data)
    predictions = bst.predict(X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), "lightgbm mean is off"
    assert predictions[0] == approx(0.0525, abs=1e-4), "lightgbm first value is off"


def test_xg():
    """Test the xg boost."""
    bst = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        enable_categorical=True,
    )
    bst.fit(X, y, sample_weight=weights)
    predictions = bst.predict(X)

    assert predictions.mean() == approx(0.0527, abs=1e-4), "xgboost mean is off"
    assert predictions[0] == approx(0.0767, abs=1e-4), "xgboost first value is off"
