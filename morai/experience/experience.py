"""Experience study model."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def normalize(
    df: pd.DataFrame,
    features: List[str],
    normalize_col: List[str] or str,
    weight_col: Optional[List[str] or str] = None,
    add_norm_col: Optional[bool] = False,
    ratio: Optional[bool] = False,
    relative_to: Optional[str] = "aggregate",
    relative_cols: Optional[List[str] or str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Normalize a column (normalize_col) based on a number of features.

    The normalization is done by calculating the relative risk of the normalize_col
    for each feature group and can be weighted by the weight_col if provided.

    Normalizing over the features is a crude method to adjust data for differences
    in the feature groups. when the features are independent, this method is
    appropriate. When the features are not independent, this method will blend
    the effects.

    Creates a new column with the suffix '_norm' if add_norm_col is True.

    Tip:
    -----
    When normalizing using a denominator, you should use the denominator of what
    calculated the rate. For example, if you are normalizing a mortality rate,
    you should use the exposure as the denominator.

    a/o = use exposure as denominator
    a/e = use expected as denominator

    Example:
    --------
    Male = 100 deaths / 1000 exposures = 0.1
    Female = 100 deaths / 500 exposures = 0.2
    Total = 200 deaths / 1500 exposures = 0.133
    Male_risk = .1 / .133 = .75
    Female_risk = .2 / .133 = 1.5
    Normalization would be:
    Male = 100 / .75 = 133.3, rate = .133
    Female = 100 / 1.5 = 66.6, rate = .133

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    features : list
        List of columns to group by.
    normalize_col : list or str
        Column to normalize.
    weight_col : list or str, optional default=None
        Weighting column.
    add_norm_col : bool, optional default=False
        Add the normalized column instead of overwriting the column being normalized.
    ratio : bool, optional default=False
        If True, the normalize_col is expected to be used as the numerator for
        the ratio.
    relative_to : str, optional default="aggregate"
        Column to calculate relative risk relative to, default is "aggregate".
        Options are "aggregate" or "reference".
    relative_cols : list or str, optional
        List of columns to have the relative risk compared to.

        For instance, if relative_cols is not used there will only differ by the
        features list. However, if relative_cols is used, there will be a risk for the
        features list and would differ by the relative_cols list.
    **kwargs : dict
        Additional keyword arguments to pass to calc_relative_risk.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with normalized values.


    """
    # check if lazy
    is_lazy = isinstance(df, pl.LazyFrame)

    # handling input types and warnings
    if isinstance(normalize_col, list):
        normalize_col = normalize_col[0]
    if isinstance(weight_col, list):
        weight_col = weight_col[0]

    # calculate the relative risk
    df = calc_relative_risk(
        df=df,
        features=features,
        risk_col=normalize_col,
        weight_col=weight_col,
        ratio=ratio,
        relative_to=relative_to,
        relative_cols=relative_cols,
        **kwargs,
    )

    # normalize the numerator
    normalized_col_name = f"{normalize_col}_norm" if add_norm_col else normalize_col

    if is_lazy:
        df = df.with_columns(
            (pl.col(normalize_col) / pl.col("relative_risk")).alias(normalized_col_name)
        )
        df = df.drop("relative_risk")
    else:
        df[normalized_col_name] = df[normalize_col] / df["relative_risk"]
        df = df.drop(columns=["relative_risk"])

    return df


def calc_relative_risk(
    df: Union[pd.DataFrame, pl.LazyFrame],
    features: List[str],
    risk_col: Union[List[str], str],
    weight_col: Optional[Union[List[str], str]] = None,
    ratio: Optional[bool] = False,
    relative_to: Optional[str] = "aggregate",
    relative_cols: Optional[Union[List[str], str]] = None,
    subset_dict: Optional[Dict[str, Any]] = None,
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """
    Calculate relative risk of a column (risk_col) based on a number of features.

    The relative risk is calculated as the average risk for the feature group
    divided by the average risk for all groups. The relative risk is weighted by
    the 'weight_col' if provided.

    Parameters
    ----------
    df : pd.DataFrame or pl.LazyFrame
        DataFrame to calculate relative risk for.
    features : list
        List of columns to group by.
    risk_col : list or str
        Column to calculate relative risk for.
    weight_col : list or str, optional
        Weighting column. If None, uses uniform weights of 1.
    ratio : bool, optional
        If True, the risk_col is expected to be used as the numerator for the ratio
        and the weight_col is expected to be used as the denominator for the ratio.
    relative_to : str, optional
        Column to calculate relative risk relative to, default is "aggregate".
        Options are "aggregate", "subset", or "reference".
    relative_cols : list or str, optional
        List of columns to have the relative risk compared to.

        For instance, if relative_cols is not used there will only differ by the
        features list. However, if relative_cols is used, there will be a risk for the
        features list and would differ by the relative_cols list.
    subset_dict : dict, optional
        Dictionary to subset the DataFrame to use as the aggregate.

    Returns
    -------
    df : pd.DataFrame or pl.LazyFrame
        DataFrame with additional column for relative risk 'risk'

    """
    is_lazy = isinstance(df, pl.LazyFrame)

    # validation
    if relative_to not in ["aggregate", "subset", "reference"]:
        raise ValueError("relative_to must be 'aggregate', 'subset', or 'reference'")
    if relative_to == "subset" and subset_dict is None:
        raise ValueError("subset_dict must be provided if relative_to is 'subset'")

    # normalize inputs
    risk_col = risk_col[0] if isinstance(risk_col, list) else risk_col
    weight_col = (
        weight_col[0] if isinstance(weight_col, list) and weight_col else weight_col
    )
    relative_cols = [relative_cols] if isinstance(relative_cols, str) else relative_cols

    # check columns exist
    subset_cols = list(subset_dict.keys()) if subset_dict else []
    check_columns = (
        features
        + [risk_col]
        + (relative_cols if relative_cols else [])
        + ([weight_col] if weight_col else [])
        + subset_cols
    )
    if is_lazy:
        df_columns = df.collect_schema().names()
    else:
        df_columns = df.columns
    for col in check_columns:
        if col not in df_columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # add temporary columns
    use_temp_weight = weight_col is None
    use_temp_relative = relative_cols is None

    if use_temp_weight:
        weight_col = "_temp_weight"
    if use_temp_relative:
        relative_cols = ["_temp_relative"]

    if is_lazy:
        if use_temp_weight:
            df = df.with_columns(pl.lit(1).alias(weight_col))
        if use_temp_relative:
            df = df.with_columns(pl.lit(1).alias(relative_cols[0]))
    else:  # pandas
        if use_temp_weight:
            df = df.copy()
            df[weight_col] = 1
        if use_temp_relative:
            df = df.copy() if not use_temp_weight else df
            df[relative_cols[0]] = 1

    # group cols
    group_cols = features + relative_cols

    if is_lazy:
        grouped_df = df.group_by(group_cols, maintain_order=True).agg(
            [
                pl.col(risk_col).sum().alias("risk_sum"),
                pl.col(weight_col).sum().alias("weight_sum"),
            ]
        )
        grouped_df = grouped_df.with_columns(
            (pl.col("risk_sum") * pl.col("weight_sum")).alias("weighted_risk")
        )

        # calculate baseline ratios for each relative group
        if subset_dict:
            subset_conditions = []
            for col, values in subset_dict.items():
                subset_conditions.append(pl.col(col).is_in(values))

            subset_filter = subset_conditions[0]
            for condition in subset_conditions[1:]:
                subset_filter = subset_filter & condition

            subset_df = df.filter(subset_filter)
            subset_grouped_df = subset_df.group_by(group_cols, maintain_order=True).agg(
                [
                    pl.col(risk_col).sum().alias("risk_sum"),
                    pl.col(weight_col).sum().alias("weight_sum"),
                ]
            )
            subset_grouped_df = subset_grouped_df.with_columns(
                (pl.col("risk_sum") * pl.col("weight_sum")).alias("weighted_risk")
            )
            base_df = subset_grouped_df
        else:
            base_df = grouped_df

        baseline_ratios = base_df.group_by(relative_cols, maintain_order=True).agg(
            [
                pl.col("risk_sum").sum().alias("total_risk"),
                pl.col("weight_sum").sum().alias("total_weight"),
                pl.col("weighted_risk").sum().alias("total_weighted_risk"),
            ]
        )

        if ratio:  # simple ratio
            baseline_ratios = baseline_ratios.with_columns(
                (pl.col("total_risk") / pl.col("total_weight")).alias("baseline_ratio")
            )
            grouped_df = grouped_df.with_columns(
                (pl.col("risk_sum") / pl.col("weight_sum")).alias("risk_numerator")
            )
        else:  # weighted ratio
            baseline_ratios = baseline_ratios.with_columns(
                (pl.col("total_weighted_risk") / pl.col("total_weight")).alias(
                    "baseline_ratio"
                )
            )
            grouped_df = grouped_df.with_columns(
                pl.col("risk_sum").alias("risk_numerator")
            )

        grouped_df = grouped_df.join(
            baseline_ratios.select([*relative_cols, "baseline_ratio"]),
            on=relative_cols,
            how="left",
        )
        grouped_df = grouped_df.with_columns(
            (pl.col("risk_numerator") / pl.col("baseline_ratio")).alias("relative_risk")
        )

        # handle zero
        grouped_df = grouped_df.with_columns(
            pl.when(pl.col("relative_risk") == 0)
            .then(pl.lit(1.0))
            .otherwise(pl.col("relative_risk"))
            .alias("relative_risk")
        )

        if relative_to == "reference":
            reference_risks = grouped_df.group_by(
                relative_cols, maintain_order=True
            ).agg(pl.col("relative_risk").first().alias("reference_risk"))

            grouped_df = grouped_df.join(reference_risks, on=relative_cols, how="left")
            grouped_df = grouped_df.with_columns(
                (pl.col("relative_risk") / pl.col("reference_risk")).alias(
                    "relative_risk"
                )
            )

        # merge to original data
        df = df.join(
            grouped_df.select(
                [*group_cols, "relative_risk", "risk_numerator", "baseline_ratio"]
            ),
            on=group_cols,
            how="left",
        )

    else:  # pandas
        grouped_df = (
            df.groupby(group_cols, observed=True, sort=False)
            .agg({risk_col: "sum", weight_col: "sum"})
            .reset_index()
            .rename(columns={risk_col: "risk_sum", weight_col: "weight_sum"})
        )
        grouped_df["weighted_risk"] = grouped_df["risk_sum"] * grouped_df["weight_sum"]

        # calculate baseline ratios for each relative group
        if subset_dict:
            subset_mask = pd.Series(True, index=df.index)
            for col, values in subset_dict.items():
                subset_mask &= df[col].isin(values)
            subset_df = df[subset_mask]
            subset_grouped_df = (
                subset_df.groupby(group_cols, observed=True, sort=False)
                .agg({risk_col: "sum", weight_col: "sum"})
                .reset_index()
                .rename(columns={risk_col: "risk_sum", weight_col: "weight_sum"})
            )
            subset_grouped_df["weighted_risk"] = (
                subset_grouped_df["risk_sum"] * subset_grouped_df["weight_sum"]
            )
            base_df = subset_grouped_df
        else:
            base_df = grouped_df

        baseline_ratios = (
            base_df.groupby(relative_cols, observed=True, sort=False)
            .agg({"risk_sum": "sum", "weight_sum": "sum", "weighted_risk": "sum"})
            .reset_index()
        )

        if ratio:  # simple ratio
            baseline_ratios["baseline_ratio"] = (
                baseline_ratios["risk_sum"] / baseline_ratios["weight_sum"]
            )
            grouped_df["risk_numerator"] = (
                grouped_df["risk_sum"] / grouped_df["weight_sum"]
            )
        else:  # weighted ratio
            baseline_ratios["baseline_ratio"] = (
                baseline_ratios["weighted_risk"] / baseline_ratios["weight_sum"]
            )
            grouped_df["risk_numerator"] = grouped_df["risk_sum"]

        grouped_df = grouped_df.merge(
            baseline_ratios[[*relative_cols, "baseline_ratio"]],
            on=relative_cols,
            how="left",
        )
        grouped_df["relative_risk"] = (
            grouped_df["risk_numerator"] / grouped_df["baseline_ratio"]
        )

        # handle zero
        grouped_df.loc[grouped_df["relative_risk"] == 0, "relative_risk"] = 1.0

        if relative_to == "reference":
            reference_risks = (
                grouped_df.groupby(relative_cols, observed=True, sort=False)[
                    "relative_risk"
                ]
                .first()
                .reset_index()
                .rename(columns={"relative_risk": "reference_risk"})
            )

            grouped_df = grouped_df.merge(reference_risks, on=relative_cols, how="left")
            grouped_df["relative_risk"] = (
                grouped_df["relative_risk"] / grouped_df["reference_risk"]
            )

        # merge to original data
        df = df.merge(
            grouped_df[
                [
                    *group_cols,
                    "relative_risk",
                    "risk_numerator",
                    "baseline_ratio",
                ]
            ],
            on=group_cols,
            how="left",
        )

    # clean up temporary columns
    temp_cols_to_drop = []
    if use_temp_weight:
        temp_cols_to_drop.append(weight_col)
    if use_temp_relative:
        temp_cols_to_drop.extend(relative_cols)

    if temp_cols_to_drop:
        if is_lazy:
            df = df.drop(temp_cols_to_drop)
        else:
            df = df.drop(columns=temp_cols_to_drop)

    return df


def calc_variance(
    df: pd.DataFrame,
    rate_col: str,
    exposure_col: str,
    amount_col: Optional[str] = None,
) -> pd.Series:
    """
    Calculate the variance of a binomial distribution.

    variance = amount^2 * exposure * rate * (1 - rate)

    Notes
    -----
    Needs to be based on seriatim data and not aggregated data.

    Reference
    ---------
    https://www.soa.org/resources/tables-calcs-tools/table-development/
    page 59

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.
    amount_col : str, optional
        Column name of the face amount.

    Returns
    -------
    variance : pd.Series
        Series with the variance values.

    """
    # check the columns exist
    missing_cols = [col for col in [rate_col, exposure_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )
    amount = 1 if amount_col is None else df[amount_col]

    # calculate the variance
    variance = amount**2 * df[exposure_col] * df[rate_col] * (1 - df[rate_col])

    return variance


def calc_moments(
    df: pd.DataFrame,
    rate_col: str,
    exposure_col: str,
    amount_col: Optional[str] = None,
    sffx: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate the moment variables of a binomial distribution.

    moment_1 = amount * exposure * rate
    moment_2_p1 = amount^2 * exposure * rate
    moment_2_p2 = amount^2 * exposure * rate^2
    moment_3_p1 = amount^3 * exposure * rate
    moment_3_p2 = amount^3 * exposure * rate^2
    moment_3_p3 = amount^3 * exposure * rate^3

    mean = moment_1
    variance = (moment_2_p1 - moment_2_p2)
    skewness = -(moment_3_p1 - 3 * moment_3_p2 + 2 * moment_3_p3) / variance ** 1.5

    Notes
    -----
    Needs to be based on seriatim data and not aggregated data.

    skewnes can also be calculated as:
    skewness = (2 * rate - 1) / (amount^2 * exposure * rate * (1 - rate)) ^ 0.5

    Reference
    ---------
    https://en.wikipedia.org/wiki/Moment_(mathematics)
    https://proofwiki.org/wiki/Skewness_of_Binomial_Distribution

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.
    amount_col : str, optional
        Column name of the face amount.
    sffx : str, optional
        Suffix for the moment columns.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for variance measures.


    """
    # check the columns exist
    missing_cols = [col for col in [rate_col, exposure_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )
    if sffx is None:
        sffx = ""
    else:
        sffx = f"_{sffx}"
        logger.info(f"Adding the label: '{sffx}' to the moment columns.")
    amount = 1 if amount_col is None else df[amount_col]

    # calculate the moments
    logger.info(
        "Calculating moments for the binomial distribution, need to be seriatim data."
    )
    moment_1 = amount * df[exposure_col] * df[rate_col]
    moment_2_p1 = amount**2 * df[exposure_col] * df[rate_col]
    moment_2_p2 = amount**2 * df[exposure_col] * df[rate_col] ** 2
    moment_3_p1 = amount**3 * df[exposure_col] * df[rate_col]
    moment_3_p2 = amount**3 * df[exposure_col] * df[rate_col] ** 2
    moment_3_p3 = amount**3 * df[exposure_col] * df[rate_col] ** 3

    # add the moments to the dataframe
    df[f"moment{sffx}_1"] = moment_1
    df[f"moment{sffx}_2_p1"] = moment_2_p1
    df[f"moment{sffx}_2_p2"] = moment_2_p2
    df[f"moment{sffx}_3_p1"] = moment_3_p1
    df[f"moment{sffx}_3_p2"] = moment_3_p2
    df[f"moment{sffx}_3_p3"] = moment_3_p3

    return df


def calc_qx_exp_ae(
    model_data: pd.DataFrame,
    predictions: pd.Series,
    model_name: str,
    exposure_col: str,
    actual_col: str,
) -> pd.DataFrame:
    """
    Add to the model data the qx, expected amount, and ae.

    Parameters
    ----------
    model_data : pd.DataFrame
        DataFrame with the model data.
    predictions : pd.DataFrame
        DataFrame with the predictions.
    model_name : str
        Name of the model.
    exposure_col : str
        Column name of the exposure.
    actual_col : str
        Column name of the actual values.

    Returns
    -------
    model_data : pd.DataFrame
        DataFrame with additional columns for the model data.

    """
    model_data[f"qx_{model_name}"] = predictions
    model_data[f"exp_amt_{model_name}"] = (
        model_data[f"qx_{model_name}"] * model_data[exposure_col]
    )
    model_data[f"ae_{model_name}"] = np.where(
        model_data[exposure_col] == 0,
        0,
        np.where(
            model_data[f"exp_amt_{model_name}"] == 0,
            1,
            model_data[actual_col] / model_data[f"exp_amt_{model_name}"],
        ),
    )
    return model_data
