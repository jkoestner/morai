"""Preprocessors used in the models."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xact.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def preprocess_data(
    model_data,
    feature_dict,
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
        - X : the feature columns after processing
        - y : the target column (no processing)
        - weights : the weights column (no processing)
        - mapping : the mapping of the features to the encoding. this includes after
          standardization.

    """
    # initializing the variables
    mapping = {}
    model_target = feature_dict.get("target", [])[0]
    if not model_target:
        raise ValueError("model_target not found in the feature_dict")
    logger.info(f"model target: {model_target}")
    y = model_data[model_target].copy()

    model_weight = feature_dict.get("weight", [])[0]
    logger.info(f"model weights: {model_weight}")
    weights = model_data[model_weight].copy() if model_weight else None
    x_features = [
        col for col in model_data.columns if col not in [model_target, model_weight]
    ]

    # check if the feature_dict has the acceptable categories
    for cat in feature_dict.keys():
        acceptable_cats = [
            "target",
            "weight",
            "passthrough",
            "cat_pass",
            "ordinal",
            "nominal",
            "ohe",
        ]
        if cat not in acceptable_cats:
            logger.warning(f"{cat} not in the acceptable categories")

    # check if the features are in the model_data
    model_feature_dict = {}
    for feature_cat, features in feature_dict.items():
        model_feature_dict[feature_cat] = [col for col in x_features if col in features]
        missing_features = [
            col
            for col in features
            if col not in [*x_features, model_target, model_weight]
        ]
        if missing_features:
            logger.warning(f"{missing_features} not in the model_data")

    # check for duplicates
    model_features = []
    for features in model_feature_dict.values():
        model_features.extend(features)
    if len(model_features) != len(set(model_features)):
        seen = set()
        duplicates = {x for x in model_features if x in seen or seen.add(x)}
        raise ValueError(f"duplicates found: {duplicates}")

    # features to preprocess
    passthrough_cols = model_feature_dict.get("passthrough", [])
    cat_pass_cols = model_feature_dict.get("cat_pass", [])
    ordinal_cols = model_feature_dict.get("ordinal", [])
    nominal_cols = model_feature_dict.get("nominal", [])
    ohe_cols = model_feature_dict.get("ohe", [])

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

    X = model_data[model_features].copy()

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
        X = pd.get_dummies(X, columns=ohe_cols, dtype="int8", sparse=True)
        for col in ohe_cols:
            mapping[col] = {
                "values": {k: col + "_" + k for k in sorted(model_data[col].unique())},
                "type": "ohe",
            }

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

    preprocess_dict = {
        "X": X,
        "y": y,
        "weights": weights,
        "mapping": mapping,
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
