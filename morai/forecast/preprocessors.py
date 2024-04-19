"""Preprocessors used in the models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def preprocess_data(
    model_data,
    feature_dict,
    add_constant=False,
    standardize=False,
    preset=None,
):
    """
    Preprocess the features.

    Parameters
    ----------
    model_data : pd.DataFrame
        The model data with the features, target, and weights.
    feature_dict : dict
        A feature dictionary with multiple options (passthrough, cat_pass, ordinal,
        nominal, ohe)
    add_constant : bool, optional (default=False)
        Whether to add a constant column to the data.
    standardize : bool, optional (default=False)
        Whether to standardize the data, which uses the StandardScaler.
    preset : str, optional (default=None)
        There are some preset options that will process the data for features
        considering the model type so the feature_dict doesn't have to be changed
        multiple times.

    Returns
    -------
    preprocess_dict : dict
        The preprocessed which includes the X, y, weights, and mapping.
        - params : the parameters used in the preprocessing
        - X : the feature columns after processing
        - y : the target column (no processing)
        - weights : the weights column (no processing)
        - mapping : the mapping of the features to the encoding. this includes after
          standardization.
        - md_encoded : the model_data with the encoded features
        - features : the features that were used in the model

    """
    # initializing the variables
    mapping = {}
    y = None
    weights = None
    constant_col = []

    # check if the feature_dict has the acceptable keys
    for key in feature_dict.keys():
        acceptable_keys = [
            "target",
            "weight",
            "passthrough",
            "cat_pass",
            "ordinal",
            "nominal",
            "ohe",
        ]
        if key not in acceptable_keys:
            logger.warning(f"{key} not in the acceptable categories")

    # check if the features are in the model_data
    model_feature_dict = {}
    missing_features = []
    column_set = set(model_data.columns)
    for feature_cat, features in feature_dict.items():
        feature_set = set(features)
        model_feature_dict[feature_cat] = list(feature_set & column_set)
        missing = list(feature_set - column_set)
        missing_features.extend(missing)
    if missing_features:
        logger.warning(f"{missing_features} not in the model_data")

    # check for duplicate features
    model_features = []
    for features in model_feature_dict.values():
        model_features.extend(features)
    if len(model_features) != len(set(model_features)):
        seen = set()
        duplicates = {x for x in model_features if x in seen or seen.add(x)}
        raise ValueError(f"duplicates found: {duplicates}")

    # get the dictionary values
    model_target = model_feature_dict.get("target", [])
    model_weight = model_feature_dict.get("weight", [])
    passthrough_cols = model_feature_dict.get("passthrough", [])
    cat_pass_cols = model_feature_dict.get("cat_pass", [])
    ordinal_cols = model_feature_dict.get("ordinal", [])
    nominal_cols = model_feature_dict.get("nominal", [])
    ohe_cols = model_feature_dict.get("ohe", [])
    model_features = (
        passthrough_cols + cat_pass_cols + ordinal_cols + nominal_cols + ohe_cols
    )

    if preset == "tree":
        logger.info(
            "using 'tree' preset which doesn't need to use 'nominal' "
            "or 'ohe' and instead uses 'ordinal'"
        )
        ordinal_cols = ordinal_cols + nominal_cols + ohe_cols
        nominal_cols = None
        ohe_cols = None
    elif preset == "pass":
        logger.info("using 'pass' preset which makes all features passthrough")
        passthrough_cols = model_features
        cat_pass_cols = None
        ordinal_cols = None
        nominal_cols = None
        ohe_cols = None

    # get y, weights, and X
    if model_target:
        logger.info(f"model target: {model_target}")
        y = model_data[model_target].copy()
    if model_weight:
        logger.info(f"model weights: {model_weight}")
        weights = model_data[model_weight].copy()
    if add_constant:
        logger.info("adding a constant column to the data")
        constant_col = ["constant"]
        model_data[constant_col] = 1
    X = model_data[model_features + constant_col].copy()

    # numeric - passthrough
    if passthrough_cols:
        logger.info(f"passthrough - (generally numeric): {passthrough_cols}")
        for col in passthrough_cols:
            mapping[col] = {
                "values": {k: k for k in X[col].unique()},
                "type": "passthrough",
            }

    # cat - passthrough
    if cat_pass_cols:
        logger.info(f"passthrough - (categorical): {cat_pass_cols}")
        for col in cat_pass_cols:
            mapping[col] = {
                "values": {k: k for k in X[col].unique()},
                "type": "cat_pass",
            }

    # ordinal - ordinal encoded
    if ordinal_cols:
        logger.info(f"ordinal - ordinal encoded: {ordinal_cols}")
        ordinal_encoder = OrdinalEncoder()
        X[ordinal_cols] = ordinal_encoder.fit_transform(X[ordinal_cols]).astype("int16")
        for col, categories in zip(ordinal_cols, ordinal_encoder.categories_):
            mapping[col] = {
                "values": {category: i for i, category in enumerate(categories)},
                "type": "ordinal",
            }

    # nominal - one hot encoded
    if ohe_cols:
        logger.info(f"nominal - one hot encoded: {ohe_cols}")
        for col in ohe_cols:
            mapping[col] = {
                "values": {k: col + "_" + k for k in sorted(model_data[col].unique())},
                "type": "ohe",
            }
        # sparse=True is used to save memory however many models don't support sparse
        X = pd.get_dummies(X, columns=ohe_cols, dtype="int8", sparse=False)

    # nominal - weighted average target encoded
    if nominal_cols:
        logger.info(f"nominal - weighted average of target encoded: {nominal_cols}")
        for col in nominal_cols:
            # Compute the weighted average for each category
            weighted_avg = model_data.groupby(col, observed=True).apply(
                lambda x: helpers._weighted_mean(
                    x[model_target], x[model_weight] if model_weight else None
                ),
                include_groups=False,
            )
            X[col] = pd.to_numeric(model_data[col].map(weighted_avg))
            mapping[col] = {"values": weighted_avg.to_dict(), "type": "weighted_avg"}

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
                    "values": {
                        k: (v - means[idx]) / stds[idx] for k, v in mapping[col].items()
                    },
                    "type": "standardized",
                }

    # model_data that is encoded
    md_encoded = pd.concat([model_data.drop(columns=model_features), X], axis=1)
    params = {
        "standardize": standardize,
        "preset": preset,
        "add_constant": add_constant,
    }

    preprocess_dict = {
        "params": params,
        "feature_dict": feature_dict,
        "model_features": model_features,
        "mapping": mapping,
        "X": X,
        "y": y,
        "weights": weights,
        "md_encoded": md_encoded,
    }

    return preprocess_dict


def bin_feature(feature, bins):
    """
    Bin a feature.

    Parameters
    ----------
    feature : pd.Series
        The numerical series to bin.
    bins : int
        The number of bins to use.

    Returns
    -------
    binned_feature : pd.Series
        The binned feature.

    Examples
    --------
    >>> feature = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> bins = 2
    >>> bin_feature(feature, bins)

    """
    # test if feature is numeric
    if not pd.api.types.is_numeric_dtype(feature):
        raise ValueError(f"feature: [{feature.name}] is not numeric")
    range_min, range_max = (feature.min() - 1), feature.max()
    bin_edges = np.linspace(range_min, range_max, bins + 1)

    # generate lables for the bins
    labels = [
        f"{int(bin_edges[i])+1}-{int(bin_edges[i+1])}"
        for i in range(len(bin_edges) - 1)
    ]

    # note that the bins are exclusive on the right side by default
    # include_lowest=True makes the first bin inclusive of the left side
    binned_feature = pd.cut(
        feature, bins=bin_edges, labels=labels, include_lowest=True, right=True
    )
    return binned_feature
