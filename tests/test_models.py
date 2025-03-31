"""
Tests the models.

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

# loading in usa mortality data for lee-carter and cbd
# https://www.mortality.org/Country/Country?cntr=USA
colspecs = [(2, 7), (15, 20), (31, 40), (46, 56), (62, 72)]
colnames = ["observation_year", "attained_age", "female", "male", "qx_raw"]
hmd_usa_df = pd.read_fwf(
    test_forecast_path / "hmd_usa_qx.txt",
    skiprows=3,
    colspecs=colspecs,
    names=colnames,
)
hmd_usa_df = hmd_usa_df[["observation_year", "attained_age", "qx_raw"]]
excluded_ages = [str(age) for age in range(101, 111)] + ["110+"]
hmd_usa_df = hmd_usa_df[~hmd_usa_df["attained_age"].isin(excluded_ages)]
hmd_usa_df["attained_age"] = hmd_usa_df["attained_age"].astype(int)
hmd_usa_df["qx_raw"] = hmd_usa_df["qx_raw"].astype(float)


def test_glm():
    """Test the Generalized Linear Model."""
    preprocess_dict = preprocessors.preprocess_data(
        simple_data,
        feature_dict=feature_dict,
        standardize=False,
        add_constant=True,
    )
    X = preprocess_dict["X"]
    y = preprocess_dict["y"]
    weights = preprocess_dict["weights"]
    GLM = models.GLM()
    GLM.fit(X, y, weights)
    predictions = GLM.model.predict(X)

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
    assert predictions[0] == approx(
        0.0645, abs=1e-4
    ), "random forest first value is off"


def test_lgb():
    """
    Test the LightGBM.

    https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
    """
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": seed,
    }
    train_data = lgb.Dataset(X, label=y, weight=weights, categorical_feature="auto")
    bst = lgb.train(params, train_data)
    predictions = bst.predict(X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), "lightgbm mean is off"
    assert predictions[0] == approx(0.0525, abs=1e-4), "lightgbm first value is off"


def test_xg():
    """
    Test the xg boost.

    https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """
    bst = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        eta=0.05,
        enable_categorical=True,
        random_state=seed,
    )
    bst.fit(X, y, sample_weight=weights)
    predictions = bst.predict(X)

    assert predictions.mean() == approx(0.0525, abs=1e-4), "xgboost mean is off"
    # xgboost has sampling which is difficult to replicate
    # loosen the tolerance
    assert predictions[0] == approx(0.0962, abs=1e-3), "xgboost first value is off"


def test_lee_carter():
    """Test the Lee-Carter model."""
    lc_df = hmd_usa_df.copy()
    # creating the model
    lc = models.LeeCarter()
    # qx values for historical
    lc_df = lc.fit(lc_df)
    # qx values for projected
    lcf_df = lc.forecast(years=5, seed=seed)

    assert lc_df["qx_lc"].mean() == approx(
        0.0526, abs=1e-4
    ), "lee-carter historical mean is off"
    assert lcf_df["qx_lc"].mean() == approx(
        0.0464, abs=1e-4
    ), "lee-carter forecasted mean is off"


def test_cbd():
    """Test the CBD model."""
    cbd_df = hmd_usa_df.copy()
    # creating the model
    cbd = models.CBD()
    # qx values for historical
    cbd_df = cbd.fit(cbd_df)
    # qx values for projected
    cbdf_df = cbd.forecast(years=5, seed=seed)

    assert cbd_df["qx_cbd"].mean() == approx(
        0.0423, abs=1e-4
    ), "cbd historical mean is off"
    assert cbdf_df["qx_cbd"].mean() == approx(
        0.0370, abs=1e-4
    ), "cbd forecasted mean is off"
