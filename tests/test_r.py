"""Tests the r models."""

import pandas as pd
from pytest import approx

from morai.forecast import preprocessors
from morai.models import r
from morai.utils import helpers

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast" / "models"
seed = 42


def test_gam_r():
    """Test the Generalized Additive Model - R (mgcv)."""
    # create model
    sigmoid_data = pd.read_csv(test_forecast_path / "sigmoid_data.csv")
    feature_dict = {
        "target": ["rate"],
        "weight": ["weight"],
        "passthrough": ["age", "gender"],
    }
    preprocess_dict = preprocessors.preprocess_data(
        sigmoid_data,
        feature_dict=feature_dict,
        standardize=False,
        add_constant=True,
    )
    X = preprocess_dict["X"]
    y = preprocess_dict["y"]
    weights = preprocess_dict["weights"]
    spline_dict = {
        "age": {"df": 5, "degree": 3, "bs": "ps"},
    }

    # test get_formula
    GAM = r.GAMR()
    formula = GAM.get_formula(X=X, y=y, spline_dict=spline_dict)
    assert formula == "rate ~ s(`age`, k=5, m=3, bs='ps')+`constant`+`gender`", (
        "formula is not correct"
    )

    # test fit
    GAM.setup_model(X=X, y=y, weights=weights, spline_dict=spline_dict)
    assert GAM.coefs is not None, "gam model is not fit"

    # test predict
    predictions = GAM.predict(X)
    assert predictions.mean() == approx(0.5813, abs=1e-4), "gam mean is off"
    assert predictions[0] == approx(0.0005, abs=1e-4), "gam first value is off"
