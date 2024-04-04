"""Metrics used in the models."""

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from morai.utils import custom_logger

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
    logger.info(f"The total AE with {expecteds} as E for dataset is {ae:.1f}%")
    rank_df["ae"] = rank_df[actuals] / rank_df[expecteds]
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

    Parameters
    ----------
    scorecard_cols : list
        The scorecard columns

    """

    def __init__(self, scorecard_cols=None):
        self.model = pd.DataFrame(
            columns=[
                "model_name",
                "model_data_shape",
                "feature_dict",
                "date_added",
            ]
        )
        self.scorecard_cols = scorecard_cols
        self.scorecard = pd.DataFrame(columns=["model_name", *scorecard_cols])

    def add_model(
        self,
        model_name,
        model_data,
        feature_dict,
        y_true,
        y_pred,
        model=None,
        **kwargs,
    ):
        """
        Add the model.

        Parameters
        ----------
        model_name : str
            The model name
        model_data : pd.DataFrame
            The model data
        feature_dict : dict
            The feature dictionary
        y_true : series
            The columns to create metrics on (actuals)
        y_pred : series
            The columns to create metrics on (predicted)
        model : model, optional (default=None)
            The model that has metrics to use
        kwargs : dict
            The keyword arguments for the metrics

        Returns
        -------
        None

        """
        logger.info(f"Adding model '{model_name}'")
        if self.check_duplicate_name(model_name):
            logger.error(
                f"Model name {model_name} already exists, not adding model. "
                f"Please use a different name."
            )
            return

        # adding model and scorecard rows
        model_row = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "model_data_shape": model_data.shape,
                    "feature_dict": feature_dict,
                    "date_added": pd.Timestamp.now(),
                }
            ]
        ).dropna(axis="columns", how="all")
        scorecard_row = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    **self.get_metrics(
                        y_true=y_true,
                        y_pred=y_pred,
                        model=model,
                        metrics=None,
                        **kwargs,
                    ),
                }
            ]
        ).dropna(axis="columns", how="all")

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

    def get_metrics(self, y_true, y_pred, metrics=None, model=None, **kwargs):
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
        metric_dict : dict
            The metric dictionary

        """
        if metrics is None:
            metrics = self.scorecard_cols
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
        return metric_dict

    def check_duplicate_name(self, model_name):
        """
        Check if the model name already exists.

        Parameters
        ----------
        model_name : str
            The model name

        Returns
        -------
        bool
            True if the model name already exists

        """
        return model_name in self.model["model_name"].values
