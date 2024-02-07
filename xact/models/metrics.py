"""Metrics used in the models."""
import logging
import logging.config
import os

import numpy as np
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


def get_metrics(y_true, y_pred, metrics, **kwargs):
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
        The list of metrics
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
        else:
            metric_dict[metric] = getattr(skm, metric)(y_true, y_pred, **kwargs)
    return metric_dict
