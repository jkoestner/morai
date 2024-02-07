"""Preprocessors used in the models."""
import logging
import logging.config
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xact.utils import helpers

logging.config.fileConfig(
    os.path.join(helpers.ROOT_PATH, "logging.ini"),
)
logger = logging.getLogger(__name__)


def preprocess_data(
    model_data,
    model_target,
    numeric_features=None,
    ordinal_features=None,
    nominal_features=None,
    model_weight=None,
    standardize=False,
):
    """
    Preprocess the features.

    Parameters
    ----------
    model_data : pd.DataFrame
        The model data.
    model_target : str
        The target.
    numeric_features : list
        The numeric features.
    ordinal_features : list
        The ordinal features.
    nominal_features : list
        The nominal features.
    model_weight : str
        The model weight.
    standardize : bool, optional (default=False)
        Whether to standardize the data.

    Returns
    -------
    preprocess_dict : dict
        The preprocessed which includes the X, y, weights, and mapping.
    """
    mapping = {}
    y = model_data[model_target].copy()
    weights = model_data[model_weight].copy() if model_weight else None
    x_features = model_data.columns.drop([model_target, model_weight])

    numeric_cols = [col for col in x_features if col in numeric_features]
    ordinal_cols = [col for col in x_features if col in ordinal_features]
    nominal_cols = [col for col in x_features if col in nominal_features]

    X = model_data[numeric_cols + ordinal_cols + nominal_cols].copy()

    # numeric - passthrough
    logger.info(f"numeric - passthrough: {numeric_cols}")
    X[numeric_cols] = X[numeric_cols]
    for col in numeric_cols:
        mapping[col] = {k: k for k in X[col].unique()}

    # ordinal - ordinal encoded
    logger.info(f"ordinal - ordinal encoded: {ordinal_cols}")
    ordinal_encoder = OrdinalEncoder()
    X[ordinal_cols] = ordinal_encoder.fit_transform(X[ordinal_cols])
    for col, categories in zip(ordinal_cols, ordinal_encoder.categories_):
        mapping[col] = {category: i for i, category in enumerate(categories)}

    # # nominal - one hot encoded
    # logger.info(f"nominal - one hot encoded: {nominal_cols}")
    # X = pd.get_dummies(X, columns=nominal_cols, dtype=int, drop_first=True)

    # nominal - weighted average target encoded
    logger.info(f"nominal - weighted average encoded: {nominal_cols}")
    for col in nominal_cols:
        # Compute the weighted average for each category
        weighted_avg = model_data.groupby(col, observed=True)[model_target].mean()
        X[col] = pd.to_numeric(model_data[col].map(weighted_avg))
        mapping[col] = weighted_avg.to_dict()

    if standardize:
        logger.info("standardizing the data")
        scaler = StandardScaler()
        scale_features = X.select_dtypes(include=[np.number]).columns.to_list()
        # fit data
        scaler.fit(X[scale_features])
        # standardize data
        X_standardized = scaler.transform(X[scale_features])
        X[scale_features] = X_standardized

        # update the mapping to reflect the standardization if
        # columns are in the mapping
        means = scaler.mean_
        stds = scaler.scale_
        for idx, col in enumerate(scale_features):
            if col in mapping:
                mapping[col] = {
                    k: (v - means[idx]) / stds[idx] for k, v in mapping[col].items()
                }

    preprocess_dict = {
        "X": X,
        "y": y,
        "weights": weights,
        "mapping": mapping,
    }

    return preprocess_dict
