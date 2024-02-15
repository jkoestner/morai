"""Metrics used in the models."""
import logging
import logging.config
import os

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from xact.utils import helpers

logging.config.fileConfig(
    os.path.join(helpers.ROOT_PATH, "logging.ini"),
)
logger = logging.getLogger(__name__)


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


def get_metrics(y_true, y_pred, metrics, model=None, **kwargs):
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
    metrics : list
        list of metrics to use
    model : model
        there are some models that provide metrics which we can use
    kwargs : dict
        The keyword arguments for the metrics

    Returns
    -------
    metric_dict : dict
        The metric dictionary

    """
    metric_dict = {}
    for metric in metrics:
        if metric == "smape":
            metric_dict[metric] = smape(y_true, y_pred)
        elif metric == "ae":
            metric_dict[metric] = ae(y_true, y_pred)
        elif metric == "aic":
            metric_dict[metric] = model.aic if model is not None else None
        else:
            metric_dict[metric] = getattr(skm, metric)(y_true, y_pred, **kwargs)
    return metric_dict


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
    logger.info(f"the total ae for dataset is {ae}")
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
