"""Preprocessors used in the models."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


def preprocess_data(
    model_data: pd.DataFrame,
    feature_dict: dict,
    add_constant: bool = False,
    standardize: bool = False,
    preset: Optional[str] = None,
) -> dict:
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
        - 'tree' : doesn't need to use 'nominal' or 'ohe' and instead uses 'ordinal'
        - 'pass' : makes all features passthrough

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
        y = model_data[model_target].squeeze().copy()
    if model_weight:
        logger.info(f"model weights: {model_weight}")
        weights = model_data[model_weight].squeeze().copy()
    if add_constant:
        logger.info("adding a constant column to the data")
        constant_col = ["constant"]
        model_data.loc[:, constant_col] = 1
        model_features = model_features + constant_col
        passthrough_cols = passthrough_cols + constant_col
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
        for col, categories in zip(
            ordinal_cols, ordinal_encoder.categories_, strict=False
        ):
            mapping[col] = {
                "values": {category: i for i, category in enumerate(categories)},
                "type": "ordinal",
            }

    # nominal - one hot encoded
    if ohe_cols:
        logger.info(f"nominal - one hot encoded (dropping first col): {ohe_cols}")
        X = X.drop(columns=ohe_cols)
        for col in ohe_cols:
            unique_values = sorted(model_data[col].unique())
            mapping[col] = {
                "values": {k: col + "_" + str(k) for k in unique_values[:]},
                "type": "ohe",
            }
            # sparse=True is used to save memory however many models
            # don't support sparse
            dummies = pd.get_dummies(
                model_data[col], prefix=col, dtype="int8", sparse=False
            )
            X = pd.concat([X, dummies], axis=1)

    # nominal - weighted average target encoded
    if nominal_cols:
        logger.info(f"nominal - weighted average of target encoded: {nominal_cols}")
        for col in nominal_cols:
            # Compute the weighted average for each category
            weighted_avg = model_data.groupby(col, observed=True).apply(
                lambda x: helpers._weighted_mean(
                    values=x[model_target].squeeze(),
                    weights=x[model_weight].squeeze() if model_weight else None,
                ),
                include_groups=False,
            )
            X[col] = pd.to_numeric(model_data[col].map(weighted_avg))
            mapping[col] = {"values": weighted_avg.to_dict(), "type": "weighted_avg"}

    if standardize:
        logger.info("standardizing the data with StandardScaler")
        scaler = StandardScaler()
        scale_features = X.select_dtypes(include="number").columns.to_list()
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
                        k: (v - means[idx]) / stds[idx]
                        for k, v in mapping[col]["values"].items()
                    },
                    "type": "standardized",
                }

    # create replicatable order of columns
    x_sorted_columns = sorted(X.columns)
    X = X[x_sorted_columns]
    model_features = sorted(model_features)

    # drop the first column of the one hot encoded columns to avoid multicollinearity
    if ohe_cols:
        for col in ohe_cols:
            first_col = next(iter(mapping[col]["values"].items()))[1]
            X = X.drop(columns=[first_col])

    # sort the mapping dict
    sorted_keys = sorted(mapping.keys())
    mapping = {key: mapping[key] for key in sorted_keys}
    for key in mapping.keys():
        mapping[key]["values"] = dict(sorted(mapping[key]["values"].items()))

    # model_data that is encoded
    md_encoded = pd.concat([model_data.drop(columns=model_features), X], axis=1)

    # create the dictionaries
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


def bin_feature(feature: pd.Series, bins: int) -> pd.Series:
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
        f"{int(bin_edges[i]) + 1}-{int(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 1)
    ]

    # note that the bins are exclusive on the right side by default
    # include_lowest=True makes the first bin inclusive of the left side
    binned_feature = pd.cut(
        feature, bins=bin_edges, labels=labels, include_lowest=True, right=True
    )
    return binned_feature


def lazy_bin_feature(
    lf: pl.LazyFrame, feature: str, bins: int, inplace: bool = False
) -> pl.LazyFrame:
    """
    Bin a feature in a LazyFrame.

    This is separated as there is logic to update the LazyFrame.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame containing the feature to bin.
    feature : str
        The name of the feature/column to bin.
    bins : int
        The number of bins to use.
    inplace : bool, optional
        If True, the original column will be replaced with the binned values.
        If False, a new column named "{feature}_binned" will be created.
        Default is False.

    Returns
    -------
    pl.LazyFrame
        A new LazyFrame with the binned feature added or replaced.

    """
    # check if feature exists
    if feature not in lf.columns:
        raise ValueError(
            f"Feature '{feature}' not found in LazyFrame columns: {lf.columns}"
        )

    # get min/max
    stats = lf.select(
        [pl.col(feature).min().alias("min"), pl.col(feature).max().alias("max")]
    ).collect()
    range_min, range_max = stats["min"][0] - 1, stats["max"][0]

    # create bin edges, labels, and unique values
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    labels = [
        f"{int(bin_edges[i]) + 1}-{int(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 1)
    ]
    unique_vals = lf.select(pl.col(feature).unique()).collect()[feature].to_list()

    # create a new lzdf with binned values column
    binned_vals = pd.cut(
        pd.Series(unique_vals),
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    bin_mapping = dict(zip(unique_vals, binned_vals.astype(str), strict=False))
    output_col = feature if inplace else f"{feature}_binned"

    lf = lf.with_columns(pl.col(feature).replace(bin_mapping).alias(output_col))

    return lf


def lazy_groupby(
    df: pl.LazyFrame,
    groupby_cols: Union[str, List[str]],
    agg_cols: Union[str, List[str]],
    agg: str,
) -> pl.LazyFrame:
    """
    Mimics a Pandas groupby call using Polars' lazy API.

    Parameters
    ----------
    df : pl.LazyFrame
        The LazyFrame to group.
    groupby_cols : str or list
        The column name(s) to group by.
    agg_cols : str or list
        The column name(s) on which to perform the aggregation.
    agg : str
        The aggregation function to apply.
        Supported values are 'sum', 'mean', and 'count'.

    Returns
    -------
    pl.LazyFrame
        The grouped and aggregated LazyFrame.
    """
    # check if columns are list or string and build aggregation expressions
    # defaults to sum aggregation
    if isinstance(agg_cols, list):
        agg_expr = []
        for col in agg_cols:
            if agg == "sum":
                agg_expr.append(pl.sum(col).alias(col))
            elif agg == "mean":
                agg_expr.append(pl.mean(col).alias(col))
            elif agg == "count":
                agg_expr.append(pl.count(col).alias(col))
            else:
                agg_expr.append(pl.sum(col).alias(col))
    else:
        if agg == "sum":
            agg_expr = pl.sum(agg_cols).alias(agg_cols)
        elif agg == "mean":
            agg_expr = pl.mean(agg_cols).alias(agg_cols)
        elif agg == "count":
            agg_expr = pl.count(agg_cols).alias(agg_cols)
        else:
            agg_expr = pl.sum(agg_cols).alias(agg_cols)

    # groupby and aggregate
    grouped_df = df.group_by(groupby_cols).agg(agg_expr)

    return grouped_df


def get_dimensions(mapping: Dict[str, Any]) -> pd.DataFrame:
    """
    Get the dimensions for each feature in the mapping.

    Parameters
    ----------
    mapping : dict
        The mapping of the features to the encoding.

    Returns
    -------
    dimensions : pd.DataFrame
        The dimensions of the mapping.

    """
    dimensions = pd.DataFrame(
        [
            {
                "feature": feature,
                "dimension": len(mapping[feature]["values"]),
                "type": mapping[feature]["type"],
            }
            for feature in mapping
        ]
    )
    return dimensions


def remap_values(df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    """
    Remap the values using the mapping.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        The object to remap.
    mapping : dict
        The mapping of the features to the encoding.

    Returns
    -------
    df : pd.DataFrame or pd.Series
        The remapped object.

    """
    for column, value_map in mapping.items():
        reversed_map = {v: k for k, v in value_map["values"].items()}
        if isinstance(df, pd.Series) and column == df.name:
            df = df.replace(reversed_map)
        elif isinstance(df, pd.DataFrame):
            # remap one hot encoded columns
            if value_map["type"] == "ohe":
                ohe_dict = dict(list(value_map["values"].items())[1:])
                dropped_cat = next(iter(value_map["values"].items()))[0]
                df[column] = dropped_cat
                for cat, col in ohe_dict.items():
                    df.loc[df[col] == 1, column] = cat
                df = df.drop(columns=ohe_dict.values())

            elif column in df.columns:
                df[column] = df[column].map(reversed_map)
    return df


def update_mapping(mapping: Dict[str, Any], key: str, values: Any) -> Dict[str, Any]:
    """
    Update the mapping key values.

    Parameters
    ----------
    mapping : dict
        The mapping of the features to the encoding.
    key : str
        The key to update.
    values : list, tuple, dict
        The values to update.

    Returns
    -------
    mapping : dict
        The updated mapping.

    """
    if isinstance(values, list):
        values = dict(enumerate(values))
    elif isinstance(values, tuple):
        values = dict(enumerate(range(values[0], values[1] + 1)))
    elif isinstance(values, dict):
        pass
    else:
        raise ValueError("values must be a list, tuple, or dict")
    mapping[key]["values"] = values
    return mapping
