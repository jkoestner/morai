"""Preprocessors used in the models."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xact.utils import custom_logger, helpers

logger = custom_logger.setup_logging()


def preprocess_data(
    model_data,
    model_target,
    passthrough_features=None,
    cat_pass_features=None,
    ordinal_features=None,
    nominal_features=None,
    ohe_features=None,
    model_weight=None,
    standardize=False,
):
    """
    Preprocess the features.

    Parameters
    ----------
    model_data : pd.DataFrame
        The dataframe to create feature encodings from.
    model_target : str
        The target column name.
    passthrough_features : list
        The features that are just passthrough
    cat_pass_features : list
        The categorical features that are just passthrough
    ordinal_features : list
        The ordinal features to encode.
    nominal_features : list
        The nominal features to encode.
    ohe_features : list
        The one hot encoded features to encode.
    model_weight : str
        The model weights to use in forecasing models. (There is no preprocessing for
        this, but the weights are included in the returned dictionary.)
    standardize : bool, optional (default=False)
        Whether to standardize the data, which uses the StandardScaler.

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
    # make features empty lists if they are None
    if not passthrough_features:
        passthrough_features = []
    if not cat_pass_features:
        cat_pass_features = []
    if not ordinal_features:
        ordinal_features = []
    if not nominal_features:
        nominal_features = []
    if not ohe_features:
        ohe_features = []

    # initializing the variables
    mapping = {}
    logger.info(f"model target: {model_target}")
    y = model_data[model_target].copy()
    logger.info(f"model weights: {model_weight}")
    weights = model_data[model_weight].copy() if model_weight else None
    x_features = [
        col for col in model_data.columns if col not in [model_target, model_weight]
    ]

    # get the features based on what's in the model_data
    passthrough_cols = [col for col in x_features if col in passthrough_features]
    cat_pass_cols = [col for col in x_features if col in cat_pass_features]
    ordinal_cols = [col for col in x_features if col in ordinal_features]
    nominal_cols = [col for col in x_features if col in nominal_features]
    ohe_cols = [col for col in x_features if col in ohe_features]
    X = model_data[
        passthrough_cols + cat_pass_cols + ordinal_cols + nominal_cols + ohe_cols
    ].copy()

    # numeric - passthrough
    if passthrough_cols:
        logger.info(f"passthrough - (generally numeric): {passthrough_cols}")
        X[passthrough_cols] = X[passthrough_cols]
        for col in passthrough_cols:
            mapping[col] = {
                "values": {k: k for k in X[col].unique()},
                "type": "passthrough",
            }

    # cat - passthrough
    if cat_pass_cols:
        logger.info(f"passthrough - (categorical): {cat_pass_cols}")
        X[cat_pass_cols] = X[cat_pass_cols]
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

    preprocess_dict = {
        "X": X,
        "y": y,
        "weights": weights,
        "mapping": mapping,
    }

    return preprocess_dict


def bin_feature(feature, bins, labels):
    """
    Bin a feature.

    Parameters
    ----------
    feature : pd.Series
        The feature to bin.
    bins : list
        The bin edges.
    labels : list
        The bin labels.

    Returns
    -------
    binned_feature : pd.Series
        The binned feature.

    Examples
    --------
    >>> import pandas as pd
    >>> feature = pd.Series([1, 2, 3, 4, 5])
    >>> bins = list(range(0, 10, 5))
    >>> labels = [f"{i}-{i+4}" for i in range(0, 10, 5)]
    >>> bin_feature(feature, bins, labels)

    """
    # note that the bins are exclusive on the right side by default
    # include_lowest=True makes the first bin inclusive of the left side
    binned_feature = pd.cut(feature, bins=bins, labels=labels, include_lowest=True)
    return binned_feature
