"""Metrics used in the models."""

import json
from io import StringIO
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import sklearn.metrics as skm

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def smape(y_true: pd.Series, y_pred: pd.Series, epsilon: float = 1e-10) -> float:
    """
    Calculate the Symetric Mean Absolute Percentage Error (sMAPE).

    source: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    y_true : series
        The actual column name
    y_pred : series
        The predicted column name
    epsilon : float, optional (default=1e-10)
        The epsilon value

    Returns
    -------
    smape : float
        The sMAPE

    """
    denominator = (y_true.abs() + y_pred.abs()) / 2 + epsilon
    return np.mean((y_true - y_pred).abs() / denominator)


def ae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the actual/expected.

    Parameters
    ----------
    y_true : series
        The actual column name
    y_pred : series
        The predicted column name

    Returns
    -------
    ae : float
        The ae

    """
    return y_true.sum() / y_pred.sum()


def ae_rank(
    df: Union[pd.DataFrame, pl.LazyFrame],
    features: list,
    actuals: str,
    expecteds: str,
    exposures: str,
) -> pd.DataFrame:
    """
    Calculate the actual/expected ranking.

    There are 2 types of ranks
      - rank_issue = abs((actuals - expected) * (actuals/expected - 1))
         - high loss value with high loss percentage will rank higher, which identifies
           issues with data better
      - rank_driver = abs((actuals - expected) * (1 - exposure/total_exposure))
         - high loss value with small exposure will rank higher, which identifies
           drivers better

    Parameters
    ----------
    df : pd.DataFrame or pl.LazyFrame
        The DataFrame.
    features : list
        The features.
    actuals : str
        The actuals.
    expecteds : str
        The expecteds.
    exposures : str
        The exposures.

    Returns
    -------
    rank_df : pd.DataFrame
        The DataFrame with the actual/expected ranking.

    """
    is_lazy = isinstance(df, pl.LazyFrame)
    if not is_lazy:
        df = pl.from_pandas(df).lazy()

    df = df.select([*features, actuals, expecteds, exposures])
    agg_result = df.select(
        [
            pl.sum(exposures).alias("total_exposure"),
            pl.sum(actuals).alias("total_actuals"),
            pl.sum(expecteds).alias("total_expecteds"),
        ]
    ).collect()
    total_exposure = agg_result["total_exposure"][0]
    ae = agg_result["total_actuals"][0] / agg_result["total_expecteds"][0]
    logger.info(f"The total AE with {expecteds} as E for dataset is {ae * 100:.1f}%")

    # dataframe with attributes and attribute values
    melted_dfs = []
    for feature in features:
        feature_df = df.select(
            [
                pl.lit(feature).alias("attribute"),
                pl.col(feature).cast(pl.Utf8).alias("attribute_value"),
                pl.col(actuals),
                pl.col(expecteds),
                pl.col(exposures),
            ]
        )
        melted_dfs.append(feature_df)
    melted_df = pl.concat(melted_dfs)

    rank_df = melted_df.group_by(["attribute", "attribute_value"]).agg(
        [
            pl.sum(actuals).alias(actuals),
            pl.sum(expecteds).alias(expecteds),
            pl.sum(exposures).alias(exposures),
        ]
    )

    # calculate inputs to rank formula
    rank_df = rank_df.with_columns(
        [
            (pl.col(actuals) / pl.col(expecteds)).alias("ae"),
            (pl.col(actuals) - pl.col(expecteds)).alias("a-e"),
            (pl.col(exposures) / total_exposure).alias("exposure_pct"),
        ]
    )
    rank_df = rank_df.with_columns(
        [
            ((pl.col(actuals) - pl.col(expecteds)).abs() * (pl.col("ae") - 1)).alias(
                "issue_value"
            )
        ]
    )
    rank_df = rank_df.with_columns(
        [
            (
                (pl.col(actuals) - pl.col(expecteds)).abs()
                * (1 - pl.col("exposure_pct"))
            ).alias("driver_value")
        ]
    )
    rank_df = rank_df.collect().to_pandas()

    # calculate ranks and sort by rank_combined
    rank_df["rank_issue"] = (
        rank_df["issue_value"].rank(ascending=False, method="dense").astype(int)
    )
    rank_df["rank_driver"] = (
        rank_df["driver_value"].rank(ascending=False, method="dense").astype(int)
    )
    rank_df["rank_combined"] = rank_df["rank_issue"] + rank_df["rank_driver"]
    rank_df = rank_df.drop(columns=["issue_value", "driver_value"])
    rank_df = rank_df.sort_values(by="rank_combined")

    return rank_df


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    metrics: list,
    prefix: str = "",
    model: Optional[Any] = None,
    **kwargs,
) -> dict:
    """
    Calculate the metrics.

    Parameters
    ----------
    y_true : series
        The actual column name
    y_pred : series
        The predicted column name
    metrics : list
        The metrics to calculate
    prefix : str, optional (default="")
        The prefix for the metrics
    model : model, optional (default=None)
        The model to calculate the AIC
    kwargs : dict
        The keyword arguments for the metrics calculation from sklearn.metrics

    Returns
    -------
    metric_dict : dict
        The dictionary of metrics

    """
    metric_dict = {}
    for metric in metrics:
        if metric == "smape":
            metric_dict[f"{prefix}_{metric}"] = smape(y_true, y_pred)
        elif metric == "shape":
            metric_dict[f"{prefix}_{metric}"] = y_true.shape[0]
        elif metric == "ae":
            metric_dict[f"{prefix}_{metric}"] = ae(y_true, y_pred)
        elif metric == "aic":
            try:
                metric_dict[f"{prefix}_{metric}"] = (
                    model.aic if model is not None else None
                )
            except AttributeError:
                logger.error(
                    f"Model `{model}` does not have AIC attribute, returning None"
                )
                metric_dict[f"{prefix}_{metric}"] = None
        else:
            try:
                metric_dict[f"{prefix}_{metric}"] = getattr(skm, metric)(
                    y_true, y_pred, **kwargs
                )
            except AttributeError:
                logger.error(f"Metric `{metric}` not found in sklearn.metrics")
                metric_dict[f"{prefix}_{metric}"] = None
    return metric_dict


class ModelResults:
    """
    Class to store model results.

    The class will hold the model information and the scorecard.

    Using json as the preferred format for storing the results to be human
    readable, however pickle is also an option.

    Parameters
    ----------
    filepath : str
        The filepath to load the model results
    metrics : list
        The metrics for scorecard

    """

    def __init__(
        self, filepath: Optional[str] = None, metrics: Optional[list] = None
    ) -> None:
        self.filepath = filepath

        # load model results from file
        if filepath is not None:
            filepath = helpers.test_path(filepath)
            if not str(filepath).endswith(".json"):
                raise ValueError("Filepath must end with .json")
            logger.info(f"loading results from {filepath}")

            with open(filepath, "r") as file:
                data = json.load(file)
            model_df = pd.read_json(StringIO(json.dumps(data["model"])), orient="split")
            model_df["date_added"] = pd.to_datetime(model_df["date_added"], unit="ms")
            scorecard_df = pd.read_json(
                StringIO(json.dumps(data["scorecard"])), orient="split"
            )
            scorecard_df.columns = pd.MultiIndex.from_tuples(scorecard_df.columns)
            importance_df = pd.read_json(
                StringIO(json.dumps(data["importance"])), orient="split"
            )

            self.model = model_df
            self.scorecard = scorecard_df
            self.importance = importance_df
            self.metrics = list(
                {col[1] for col in scorecard_df.columns if col[1] != ""}
            )
        else:
            self.model = pd.DataFrame()
            self.scorecard = pd.DataFrame()
            self.importance = pd.DataFrame()
            self.metrics = metrics

    def add_model(
        self,
        model_name: str,
        data_path: str,
        data_shape: tuple,
        preprocess_dict: dict,
        model_params: dict,
        scorecard: pd.DataFrame,
        importance: pd.DataFrame = None,
    ) -> None:
        """
        Add the model.

        Parameters
        ----------
        model_name : str
            The model name
        data_path : str
            The data path
        data_shape : tuple
            The data shape
        preprocess_dict : dict
            The preprocess dictionary
        model_params : dict
            The model parameters
        scorecard : pd.DataFrame
            The scorecard row
        importance : pd.DataFrame, optional (default=None)
            The importance of the features

        Returns
        -------
        None

        """
        if self.check_duplicate_name(model_name):
            logger.error(
                f"Model name {model_name} already exists, not adding model. "
                f"Please use a different name."
            )
            return

        logger.info(f"Adding model '{model_name}'")

        # check if preprocess_dict is None
        if preprocess_dict is None:
            preprocess_params = None
            feature_dict = None
            model_features = None
        else:
            preprocess_params = preprocess_dict["params"]
            feature_dict = preprocess_dict["feature_dict"]
            model_features = preprocess_dict["model_features"]

        # initialize model row
        model_row = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "data_path": data_path,
                    "data_shape": data_shape,
                    "preprocess_params": preprocess_params,
                    "feature_dict": feature_dict,
                    "model_params": model_params,
                    "model_features": model_features,
                    "date_added": pd.Timestamp.now(),
                }
            ]
        ).dropna(axis="columns", how="all")

        # initialize scorecard row and put model_name in the first column
        scorecard_row = scorecard.dropna(axis="columns", how="all")
        scorecard_row.insert(0, "model_name", model_name)

        # initialize importance row
        importance_row = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "importance": importance,
                }
            ]
        ).dropna(axis="columns", how="all")

        # appending to the model and scorecard
        if self.model.empty or self.scorecard.empty:
            self.model = model_row
            self.scorecard = scorecard_row
            self.importance = importance_row
        else:
            self.model = pd.concat([self.model, model_row], ignore_index=True)
            self.scorecard = pd.concat(
                [self.scorecard, scorecard_row], ignore_index=True
            )
            self.importance = pd.concat(
                [self.importance, importance_row], ignore_index=True
            )

    def remove_model(self, model_name: str) -> None:
        """
        Remove the model.

        Parameters
        ----------
        model_name : str
            The model name

        Returns
        -------
        None

        """
        logger.info(f"Removing model '{model_name}'")
        self.model = self.model[self.model["model_name"] != model_name]
        self.scorecard = self.scorecard[self.scorecard["model_name"] != model_name]
        self.importance = self.importance[self.importance["model_name"] != model_name]

    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the model.

        Parameters
        ----------
        filepath : str
            The path to save the model

        Returns
        -------
        None

        """
        if filepath is None:
            filepath = (
                self.filepath
                if self.filepath
                else helpers.FILES_PATH / "result" / "model_results.json"
            )

        if not str(filepath).endswith(".json"):
            raise ValueError("Filname must end with .json")
        logger.info(f"saving results to {filepath}")

        # saving the results
        model_json = json.loads(self.model.to_json(orient="split", index=False))
        scorecard_json = json.loads(self.scorecard.to_json(orient="split", index=False))
        importance_json = json.loads(
            self.importance.to_json(orient="split", index=False)
        )
        model_results = {
            "model": model_json,
            "scorecard": scorecard_json,
            "importance": importance_json,
        }
        with open(filepath, "w") as file:
            json.dump(model_results, file, indent=4)

    def get_scorecard(
        self,
        y_true_train: pd.Series,
        y_pred_train: pd.Series,
        weights_train: pd.Series = None,
        y_true_test: pd.Series = None,
        y_pred_test: pd.Series = None,
        weights_test: pd.Series = None,
        metrics: Optional[list] = None,
        model: Optional[Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get the metrics.

        In general a higher score indicates better performance. Metrics that
        are better with lower score are:
        - smape
        - mean_squared_error
        - mean_absolute_error

        To get a list of possible metrics:
        - https://scikit-learn.org/stable/modules/model_evaluation.html

        Parameters
        ----------
        y_true_train : series
            The actual column name
        y_pred_train : series
            The predicted column name
        weights_train : series, optional (default=None)
            The train weights column name
        y_true_test : series, optional (default=None)
            The test actual column name
        y_pred_test : series, optional (default=None)
            The test predicted column name
        weights_test : series, optional (default=None)
            The test weights column name
        model : model
            there are some models that provide metrics which we can use
        metrics : list
            list of metrics to use
        kwargs : dict
            The keyword arguments for the metrics

        Returns
        -------
        scorecard : pd.DataFrame
            Metrics scorecard in a DataFrame

        """
        if metrics is None:
            metrics = self.metrics
        results = {}

        # apply weights to predictions
        if weights_train is not None:
            if weights_test is None:
                raise ValueError("weights_test must be provided if weights_train is")
            y_true_train = y_true_train * weights_train
            y_pred_train = y_pred_train * weights_train
            y_true_test = y_true_test * weights_test
            y_pred_test = y_pred_test * weights_test

        # calculate train
        results.update(
            calculate_metrics(
                y_true=y_true_train,
                y_pred=y_pred_train,
                prefix="train",
                metrics=metrics,
                model=model,
                **kwargs,
            )
        )

        # calculate test if provided
        if y_true_test is not None and y_pred_test is not None:
            results.update(
                calculate_metrics(
                    y_true=y_true_test,
                    y_pred=y_pred_test,
                    prefix="test",
                    metrics=metrics,
                    model=model,
                    **kwargs,
                )
            )

        # create dataframe
        scorecard = pd.DataFrame([results])
        scorecard.columns = pd.MultiIndex.from_tuples(
            [
                (c.split("_", 1)[0], c.split("_", 1)[1]) if "_" in c else (c, "")
                for c in scorecard.columns
            ]
        )

        return scorecard

    def check_duplicate_name(self, model_name: str) -> bool:
        """
        Check if the model name already exists.

        Parameters
        ----------
        model_name : str
            The model name

        Returns
        -------
        duplicate : bool
            True if the model name already exists

        """
        if self.model.empty:
            duplicate = False
        else:
            duplicate = model_name in self.model["model_name"].values

        return duplicate
