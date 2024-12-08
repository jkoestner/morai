"""Experience study model."""

from typing import List, Optional

import numpy as np
import pandas as pd

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def normalize(
    df: pd.DataFrame,
    features: List[str],
    numerator: str,
    denominator: Optional[str] = None,
    add_norm_col: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Normalize a column (numerator) based on a number of features.

    The normalization is done by calculating the relative risk of the numerator
    for each feature group and can be weighted by the denominator if provided.

    Normalizing over the features is a crude method to adjust data for differences
    in the feature groups. when the features are independent, this method is
    appropriate. When the features are not independent, this method will blend
    the effects.

    Creates a new column with the normalized numerator with the suffix '_norm'.

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
    Normalization would be:
    Male = 100 / .75 = 133.3, rate = .133
    Female = 50 / 1.5 = 66.6, rate = .133

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    features : list
        List of columns to group by.
    numerator : str
        Column to normalize.
    denominator : str, optional
        Weighting column.
    add_norm_col : bool, optional
        Add the normalized column instead of overwriting.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with normalized values.


    """
    numerator_col = numerator if not add_norm_col else f"{numerator}_norm"

    # calculate the relative risk
    df = calc_relative_risk(
        df=df, features=features, numerator=numerator, denominator=denominator
    )

    # normalize the numerator
    df[numerator_col] = df[numerator] / df["risk"]
    df = df.drop(columns=["risk"])

    return df


def calc_relative_risk(
    df: pd.DataFrame,
    features: List[str],
    numerator: str,
    denominator: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate relative risk of a column (numerator) based on a number of features.

    The relative risk is calculated as the average rate for the feature group
    divided by the average rate for all groups. The relative risk is weighted by
    the denominator if provided.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    features : list
        List of columns to group by.
    numerator : str
        Column to normalize.
    denominator : str, optional
        Weighting column.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional column for relative risk 'risk'


    """
    if denominator is None:
        df["temp_one"] = 1
        denominator = "temp_one"

    # calculated the relative risk for each feature group
    grouped_df = (
        df.groupby(features, observed=True)[[numerator, denominator]]
        .sum()
        .reset_index()
    )
    total_ratio = grouped_df[numerator].sum() / grouped_df[denominator].sum()
    grouped_df["risk"] = (grouped_df[numerator] / grouped_df[denominator]) / total_ratio

    # merge risk back to the original dataframe
    df = df.merge(grouped_df[[*features, "risk"]], on=features, how="left")
    if "temp_one" in df.columns:
        df = df.drop(columns=["temp_one"])

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
