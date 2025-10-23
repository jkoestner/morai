"""Tests the core models."""

import pandas as pd
from pytest import approx

from morai.forecast import preprocessors
from morai.models import core
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
    add_constant=True,
)
X = preprocess_dict["X"]
y = preprocess_dict["y"]
weights = preprocess_dict["weights"]
mapping = preprocess_dict["mapping"]

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

expected_age_columns = [
    "0-8",
    "9-17",
    "18-26",
    "27-35",
    "36-44",
    "45-53",
    "54-62",
    "63-71",
    "72-80",
    "81-89",
    "90-98",
    "99-100",
]


def test_model_wrapper():
    """Test the model wrapper."""
    # create model
    GLM = core.GLM()
    GLM.fit(X, y, weights)

    # test get_features
    features = core.ModelWrapper(GLM.model).get_features()
    assert features == ["constant", "gender", "year"]

    # test get_importance
    importance = core.ModelWrapper(GLM.model).get_importance()
    assert importance is not None

    # test check_predict
    assert core.ModelWrapper(GLM.model).check_predict() is None


def test_glm():
    """Test the Generalized Linear Model."""
    # test fit
    GLM = core.GLM()
    GLM.fit(X, y, weights)
    assert GLM.model.params is not None, "GLM model is not fit"

    # test predict
    predictions = GLM.predict(X)
    assert predictions.mean() == approx(0.0525, abs=1e-4), "glm mean is off"
    assert predictions[0] == approx(0.0815, abs=1e-4), "glm first value is off"

    # test predict - manual
    manual_predictions = GLM.predict(X, manual=True)
    assert manual_predictions.mean() == approx(0.0525, abs=1e-4), "glm mean is off"
    assert manual_predictions[0] == approx(0.0815, abs=1e-4), "glm first value is off"

    # test get_odds
    odds = GLM.get_odds()
    assert odds["gender"] == approx(1.10, abs=1e-2)
    assert odds["year"] == approx(0.90, abs=1e-2)

    # test get_feature_contributions
    feature_contributions = GLM.get_feature_contributions(X, y)
    assert feature_contributions.loc[
        feature_contributions["features"] == "year", "contribution"
    ].iloc[0] == approx(0.97, abs=1e-2)
    assert feature_contributions.loc[
        feature_contributions["features"] == "gender", "contribution"
    ].iloc[0] == approx(0.03, abs=1e-2)

    # test get_feature_contributions - with a base feature
    feature_contributions = GLM.get_feature_contributions(
        X, y, base_features=["gender"]
    )
    assert feature_contributions.loc[
        feature_contributions["features"] == "year", "contribution"
    ].iloc[0] == approx(1.00, abs=1e-2)


def test_glm_r_style():
    """Test the Generalized Linear Model with r_style."""
    GLM = core.GLM()

    # test fit
    GLM.fit(X, y, weights, r_style=True, mapping=mapping)
    assert GLM.model.params is not None, "GLM model is not fit"

    # test get_formula
    formula = GLM.get_formula(X, y)
    assert formula == "rate ~ constant + gender + year"

    # test predict
    predictions = GLM.predict(X)
    assert predictions.mean() == approx(0.0525, abs=1e-4), "glm mean is off"
    assert predictions[0] == approx(0.0815, abs=1e-4), "glm first value is off"


def test_lee_carter():
    """Test the Lee-Carter model."""
    # create data
    lc_df = hmd_usa_df.copy()
    lc_df["exposure"] = 1000
    lc_df["actuals"] = lc_df["exposure"] * lc_df["qx_raw"]
    lc_df.drop(columns=["qx_raw"], inplace=True)

    # creating the model
    lc = core.LeeCarter(
        age_col="attained_age",
        year_col="observation_year",
        actual_col="actuals",
        expose_col="exposure",
        interval=9,
    )

    # test structure_df
    lc_df = lc.structure_df(lc_df)
    assert "qx_raw" in lc_df.columns

    # test fit
    lc_df = lc.fit(lc_df)
    assert list(lc.age_columns.keys()) == expected_age_columns
    assert lc_df["qx_lc"].mean() == approx(0.0526, abs=1e-4), (
        "lee-carter historical mean is off"
    )

    # test forecast
    lcf_df = lc.forecast(years=5, seed=seed)
    assert lcf_df["qx_lc"].mean() == approx(0.0445, abs=1e-4), (
        "lee-carter forecasted mean is off"
    )

    # test map
    df = pd.DataFrame(
        {"observation_year": [2020, 2020, 2021, 2021], "attained_age": [20, 30, 20, 30]}
    )
    mapped_df = lc.map(df)
    assert mapped_df.iloc[0]["qx_lc"] == approx(0.001017, abs=1e-6), (
        "lee-carter historical mean is off for age 20 and year 2020"
    )


def test_cbd():
    """Test the CBD model."""
    # create data
    cbd_df = hmd_usa_df.copy()
    cbd_df["exposure"] = 1000
    cbd_df["actuals"] = cbd_df["exposure"] * cbd_df["qx_raw"]
    cbd_df.drop(columns=["qx_raw"], inplace=True)

    # creating the model
    cbd = core.CBD(
        age_col="attained_age",
        year_col="observation_year",
        actual_col="actuals",
        expose_col="exposure",
        interval=9,
    )

    # test structure_df
    cbd_df = cbd.structure_df(cbd_df)
    assert "qx_raw" in cbd_df.columns

    # test fit
    cbd_df = cbd.fit(cbd_df)
    assert list(cbd.age_columns.keys()) == expected_age_columns
    assert cbd_df["qx_cbd"].mean() == approx(0.0525, abs=1e-4), (
        "cbd historical mean is off"
    )

    # test forecast
    cbd_df = cbd.forecast(years=5, seed=seed)
    assert cbd_df["qx_cbd"].mean() == approx(0.0446, abs=1e-4), (
        "cbd forecasted mean is off"
    )

    # test map
    df = pd.DataFrame(
        {"observation_year": [2020, 2020, 2021, 2021], "attained_age": [20, 30, 20, 30]}
    )
    mapped_df = cbd.map(df)
    assert mapped_df.iloc[0]["qx_cbd"] == approx(0.000922, abs=1e-6), (
        "cbd historical mean is off for age 20 and year 2020"
    )


def test_gam_stats():
    """Test the Generalized Additive Model - statsmodels."""
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
        "age": {"df": 5, "degree": 3},
    }

    # test smoother
    smoother, X_model = core.GAMStats().create_smoother(X=X, spline_dict=spline_dict)
    assert smoother.col_names == ["age_s0", "age_s1", "age_s2", "age_s3"]
    assert smoother.basis.shape == (480, 4)
    assert smoother.degrees == [3]

    # test fit
    GAM = core.GAMStats()
    GAM.setup_model(X=X, y=y, weights=weights, spline_dict=spline_dict)
    GAM.fit()
    assert GAM.model.params is not None, "gam model is not fit"

    # test predict
    predictions = GAM.predict(X)
    assert predictions.mean() == approx(0.4381, abs=1e-4), "gam mean is off"
    assert predictions[0] == approx(2.9221e-04, abs=1e-4), "gam first value is off"

    # test search_alpha
    alpha = GAM.search_alpha()
    assert isinstance(alpha, float), "alpha is not a float"

    # test get_formula
    formula = GAM.get_formula(X=X, y=y, smoother=smoother)
    assert formula == "rate ~ constant + gender"


def test_gam_pygam():
    """Test the Generalized Additive Model - pygam."""
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
        "age": {"df": 5, "degree": 3},
    }

    # test get_terms
    GAM = core.GAMPy()
    terms = GAM.get_terms(X=X, spline_dict=spline_dict)
    assert str(terms) == "s(0) + l(1) + l(2)", "terms are not correct"

    # test fit
    GAM.setup_model(X=X, y=y, weights=weights, spline_dict=spline_dict)
    GAM.fit()
    assert GAM.model.get_params() is not None, "gam model is not fit"

    # test predict
    predictions = GAM.predict(X)
    assert predictions.mean() == approx(0.5813, abs=1e-4), "gam mean is off"
    assert predictions[0] == approx(0.0028, abs=1e-4), "gam first value is off"


def test_calc_likelihood_ratio():
    """Test the calc_likelihood_ratio function."""
    # create full_model
    full_model = core.GLM()
    full_model.fit(X, y, weights)

    # create reduced_model
    feature_dict_reduced = {
        "target": ["rate"],
        "weight": [],
        "passthrough": ["gender"],
    }
    preprocess_dict_reduced = preprocessors.preprocess_data(
        simple_data,
        feature_dict=feature_dict_reduced,
        standardize=False,
        add_constant=True,
    )
    X_reduced = preprocess_dict_reduced["X"]
    y_reduced = preprocess_dict_reduced["y"]
    weights_reduced = preprocess_dict_reduced["weights"]
    reduced_model = core.GLM()
    reduced_model.fit(X_reduced, y_reduced, weights_reduced)

    # test calc_likelihood_ratio
    likelihood_dict = core.calc_likelihood_ratio(full_model.model, reduced_model.model)
    assert likelihood_dict["likelihood_ratio"] == approx(0.0836, abs=1e-4)
    assert likelihood_dict["degrees_of_freedom_diff"] == 1
    assert likelihood_dict["p_value"] == approx(0.7725, abs=1e-4)
