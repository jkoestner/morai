"""Metrics used in the models."""

import json
from io import StringIO

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def smape(y_true, y_pred, epsilon=1e-10):
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


def ae(y_true, y_pred):
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


def ae_rank(df, features, actuals, expecteds, exposures):
    """
    Calculate the actual/expected ranking.

    There are 2 types of ranks
      - rank1 = abs((actuals - expected) * (actuals/expected - 1))
         - high loss value with high loss percentage will rank higher, which identifies
           issues with data better
      - rank2 = abs((actuals - expected) * (1 - exposure/total_exposure))
         - high loss value with small exposure will rank higher, which identifies
           drivers better

    Parameters
    ----------
    df : pd.DataFrame
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
    df = df[[*features, actuals, expecteds, exposures]]
    total_exposure = df[exposures].sum()
    ae = df[actuals].sum() / df[expecteds].sum()

    # dataframe with attributes and attribute values
    rank_df = pd.melt(
        df,
        id_vars=[actuals, expecteds, exposures],
        value_vars=features,
        var_name="attribute",
        value_name="attribute_value",
    )
    rank_df = rank_df.groupby(["attribute", "attribute_value"]).sum().reset_index()

    # calculate ranks and sort by rank_combined
    logger.info(f"The total AE with {expecteds} as E for dataset is {ae*100:.1f}%")
    rank_df["ae"] = rank_df[actuals] / rank_df[expecteds]
    rank_df["a-e"] = rank_df[actuals] - rank_df[expecteds]
    rank_df["exposure_pct"] = rank_df[exposures] / total_exposure
    rank_df["rank_issue"] = (
        abs((rank_df[actuals] - rank_df[expecteds]) * (rank_df["ae"] - 1))
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    rank_df["rank_driver"] = (
        abs((rank_df[actuals] - rank_df[expecteds]) * (1 - rank_df["exposure_pct"]))
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    rank_df["rank_combined"] = rank_df["rank_issue"] + rank_df["rank_driver"]
    rank_df = rank_df.sort_values(by="rank_combined")

    return rank_df


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

    def __init__(self, filepath=None, metrics=None):
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

            self.model = model_df
            self.scorecard = scorecard_df
            self.metrics = scorecard_df.columns.tolist().remove("model_name")
        else:
            self.model = pd.DataFrame()
            self.scorecard = pd.DataFrame()
            self.metrics = metrics

    def add_model(
        self,
        model_name,
        data_path,
        data_shape,
        feature_dict,
        scorecard,
        preprocess_params=None,
        model_params=None,
    ):
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
        feature_dict : dict
            The feature dictionary
        scorecard : pd.DataFrame
            The scorecard row
        preprocess_params : dict, optional (default=None)
            The preprocessing parameters
        model_params : dict, optional (default=None)
            The model parameters

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
        # adding model row
        model_row = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "data_path": data_path,
                    "data_shape": data_shape,
                    "preprocess_params": preprocess_params,
                    "feature_dict": feature_dict,
                    "model_params": model_params,
                    "date_added": pd.Timestamp.now(),
                }
            ]
        ).dropna(axis="columns", how="all")
        # adding scorecard row
        scorecard_row = scorecard.dropna(axis="columns", how="all")
        scorecard_row["model_name"] = model_name

        # appending to the model and scorecard
        if self.model.empty or self.scorecard.empty:
            self.model = model_row
            self.scorecard = scorecard_row
        else:
            self.model = pd.concat([self.model, model_row], ignore_index=True)
            self.scorecard = pd.concat(
                [self.scorecard, scorecard_row], ignore_index=True
            )

    def remove_model(self, model_name):
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

    def save_model(self, filepath=None):
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
        model_results = {"model": model_json, "scorecard": scorecard_json}
        with open(filepath, "w") as file:
            json.dump(model_results, file, indent=4)

    def get_scorecard(self, y_true, y_pred, metrics=None, model=None, **kwargs):
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
        y_true : series
            The actual column name
        y_pred : series
            The predicted column name
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
        metric_dict = {}
        for metric in metrics:
            if metric == "smape":
                metric_dict[metric] = smape(y_true, y_pred)
            elif metric == "ae":
                metric_dict[metric] = ae(y_true, y_pred)
            elif metric == "aic":
                try:
                    metric_dict[metric] = model.aic if model is not None else None
                except AttributeError:
                    logger.error(
                        f"Model {model} does not have AIC attribute, returning None"
                    )
                    metric_dict[metric] = None
            else:
                try:
                    metric_dict[metric] = getattr(skm, metric)(y_true, y_pred, **kwargs)
                except AttributeError:
                    logger.error(f"Metric {metric} not found in sklearn.metrics")
                    metric_dict[metric] = None

        scorecard = pd.DataFrame([metric_dict])
        return scorecard

    def check_duplicate_name(self, model_name):
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
