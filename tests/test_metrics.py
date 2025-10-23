"""Tests the metrics."""

import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

from morai.forecast import metrics
from morai.utils import helpers

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast"
metric_df = pd.read_csv(test_forecast_path / "metrics" / "metric_data.csv")
ae_rank_df = pd.read_csv(test_forecast_path / "metrics" / "ae_rank_data.csv")


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


def test_ae():
    """Tests the AE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    test_ae = y_true.sum() / y_pred.sum()
    assert metrics.ae(y_true, y_pred) == test_ae, "ae not matching"


def test_ae_rank():
    """Tests the AE rank calculation."""
    features = ["feature_issue", "feature_driver"]
    actuals = "actuals"
    expecteds = "expecteds"
    exposures = "exposures"
    ae_rank = metrics.ae_rank(ae_rank_df, features, actuals, expecteds, exposures)
    ae_rank = ae_rank.sort_values(["rank_combined", "rank_issue"]).reset_index(
        drop=True
    )

    # manual calculation of rank_issue
    df = ae_rank_df.copy()
    total_exposure = df["exposures"].sum()
    melted = []
    for feature in features:
        melted.append(
            df.groupby(feature)
            .agg(
                actuals=("actuals", "sum"),
                expecteds=("expecteds", "sum"),
                exposures=("exposures", "sum"),
            )
            .reset_index()
            .rename(columns={feature: "attribute_value"})
        )
        melted[-1]["attribute"] = feature
    manual = pd.concat(melted, ignore_index=True)

    # add formulas
    manual["ae"] = manual["actuals"] / manual["expecteds"]
    manual["a-e"] = manual["actuals"] - manual["expecteds"]
    manual["exposure_pct"] = manual["exposures"] / total_exposure
    manual["issue_value"] = (manual["actuals"] - manual["expecteds"]).abs() * (
        manual["ae"] - 1
    )
    manual["driver_value"] = (manual["actuals"] - manual["expecteds"]).abs() * (
        1 - manual["exposure_pct"]
    )

    manual["rank_issue"] = (
        manual["issue_value"].rank(ascending=False, method="dense").astype(int)
    )
    manual["rank_driver"] = (
        manual["driver_value"].rank(ascending=False, method="dense").astype(int)
    )
    manual["rank_combined"] = manual["rank_issue"] + manual["rank_driver"]

    manual = manual.sort_values(["rank_combined", "rank_issue"]).reset_index(drop=True)
    assert ae_rank["rank_issue"].equals(manual["rank_issue"])
    assert ae_rank["rank_driver"].equals(manual["rank_driver"])
    assert ae_rank["rank_combined"].equals(manual["rank_combined"])

    # test polars
    pl_ae_rank = metrics.ae_rank(
        pl.from_pandas(ae_rank_df).lazy(), features, "actuals", "expecteds", "exposures"
    )
    pl_ae_rank = pl_ae_rank.sort_values(["rank_combined", "rank_issue"]).reset_index(
        drop=True
    )
    assert pl_ae_rank.equals(ae_rank)


def test_calc_metrics_mape():
    """Tests the MAPE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]
    test_mape = ((y_true - y_pred).abs() / y_true.abs()).mean()
    mape = metrics.calculate_metrics(
        y_true, y_pred, metrics=["mean_absolute_percentage_error"]
    )["_mean_absolute_percentage_error"]
    assert mape == test_mape, "MAPE not matching"


def test_calc_metrics_mse():
    """Tests the MSE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    test_mse = ((y_true - y_pred) ** 2).mean()
    mse = metrics.calculate_metrics(y_true, y_pred, metrics=["mean_squared_error"])[
        "_mean_squared_error"
    ]
    assert mse == test_mse, "MSE not matching"


def test_calc_metrics_mae():
    """Tests the MAE calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    test_mae = ((y_true - y_pred).abs()).mean()
    mae = metrics.calculate_metrics(y_true, y_pred, metrics=["mean_absolute_error"])[
        "_mean_absolute_error"
    ]
    assert mae == test_mae, "MAE not matching"


def test_calc_metrics_r2():
    """Tests the R2 calculation."""
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]

    rss = ((y_true - y_pred) ** 2).sum()
    tss = ((y_true - y_true.mean()) ** 2).sum()

    test_r2 = 1 - rss / tss
    r2 = metrics.calculate_metrics(y_true, y_pred, metrics=["r2_score"])["_r2_score"]
    assert r2 == test_r2, "R2 not matching"


def test_calc_metrics_others():
    """
    Tests the other metrics calculation.

    Other includes the following metrics:
    - smape
    - shape
    - ae
    """
    y_true = metric_df["actual"]
    y_pred = metric_df["pred"]
    test_smape = metrics.smape(y_true, y_pred)
    test_ae = metrics.ae(y_true, y_pred)
    test_shape = y_true.shape[0]
    smape = metrics.calculate_metrics(y_true, y_pred, metrics=["smape"])["_smape"]
    ae = metrics.calculate_metrics(y_true, y_pred, metrics=["ae"])["_ae"]
    shape = metrics.calculate_metrics(y_true, y_pred, metrics=["shape"])["_shape"]
    assert smape == test_smape, "sMAPE not matching"
    assert ae == test_ae, "AE not matching"
    assert shape == test_shape, "Shape not matching"


def test_model_results():
    """
    Tests the model_results class.

    The tests include:
    - get_scorecard
    - add_model
    - remove_model
    - save_model
    - load_model
    """
    data_path = test_forecast_path / "metrics" / "ae_rank_data.csv"

    metric_cols = ["ae", "smape", "r2_score", "root_mean_squared_error", "aic", "shape"]
    model_results = metrics.ModelResults(metrics=metric_cols)

    # get scorecard
    scorecard = model_results.get_scorecard(
        y_true_train=ae_rank_df["actuals"],
        y_pred_train=ae_rank_df["expecteds"],
        weights_train=None,
        model=None,
        metrics=metric_cols,
    )
    assert scorecard.shape == (1, len(metric_cols))
    assert scorecard.columns.get_level_values(1).tolist() == metric_cols

    # add model
    model_results.add_model(
        model_name="test_model",
        data_path=data_path,
        data_shape=ae_rank_df.shape,
        preprocess_dict=None,
        model_params=None,
        scorecard=scorecard,
        importance=None,
    )

    assert "test_model" in model_results.model["model_name"].values
    assert "test_model" in model_results.scorecard["model_name"].values

    # check duplicate model does not add
    model_results.add_model(
        model_name="test_model",
        data_path=data_path,
        data_shape=ae_rank_df.shape,
        preprocess_dict=None,
        model_params=None,
        scorecard=scorecard,
        importance=None,
    )
    assert len(model_results.model) == 1
    assert len(model_results.scorecard) == 1

    # save model
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_model_results.json"
        model_results.save_model(filepath=str(tmp_path))
        assert tmp_path.exists()

        # load model
        model_results = metrics.ModelResults(filepath=str(tmp_path))
        assert "test_model" in model_results.model["model_name"].values
        assert "test_model" in model_results.scorecard["model_name"].values

    # remove model
    model_results.remove_model("test_model")
    assert model_results.model.empty
    assert model_results.scorecard.empty
