"""Experience study model."""

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def normalize(df, features, numerator, denominator=None, suffix=None):
    """
    Normalize a column (numerator) based on a number of features.

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
    suffix : str, optional
        Additional suffix for the normalized column if doing multiple normalizations.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional column for normalized values with prefix '_norm'.


    """
    numerator_norm = f"{numerator}_norm"
    if suffix is not None:
        numerator_norm = f"{numerator_norm}_{suffix}"

    # calculate the relative risk
    df = calc_relative_risk(
        df=df, features=features, numerator=numerator, denominator=denominator
    )

    # normalize the numerator
    df[numerator_norm] = df[numerator] / df["risk"]
    df = df.drop(columns=["risk"])

    return df


def calc_relative_risk(df, features, numerator, denominator=None):
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


def calc_variance(df, amount_col, rate_col, exposure_col):
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
    amount_col : str
        Column name of the face amount.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.

    Returns
    -------
    variance : pd.Series
        Series with the variance values.

    """
    # check the columns exist
    missing_cols = [col for col in [amount_col, rate_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # calculate the variance
    variance = (
        df[amount_col] ** 2 * df[exposure_col] * df[rate_col] * (1 - df[rate_col])
    )

    return variance


def calc_skewness(df, amount_col, rate_col, exposure_col):
    """
    Calculate the skewness of a binomial distribution.

    skewness = (2 * rate - 1) / (amount^2 * exposure * rate * (1 - rate)) ^ 0.5

    Notes
    -----
    Needs to be based on seriatim data and not aggregated data.

    Reference
    ---------
    https://proofwiki.org/wiki/Skewness_of_Binomial_Distribution

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    amount_col : str
        Column name of the face amount.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.

    Returns
    -------
    skewness : pd.Series
        Series with the skewness values.


    """
    # check the columns exist
    missing_cols = [col for col in [amount_col, rate_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # calculate the skewness
    variance = calc_variance(df, amount_col, rate_col, exposure_col)
    skewness = (df[rate_col] * 2 - 1) / (variance) ** 0.5

    return skewness


def calc_moments(df, amount_col, rate_col, exposure_col):
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

    Reference
    ---------
    https://en.wikipedia.org/wiki/Moment_(mathematics)
    https://proofwiki.org/wiki/Skewness_of_Binomial_Distribution

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    amount_col : str
        Column name of the face amount.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for variance measures.


    """
    # check the columns exist
    missing_cols = [col for col in [amount_col, rate_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # calculate the moments
    moment_1 = df[amount_col] * df[exposure_col] * df[rate_col]
    moment_2_p1 = df[amount_col] ** 2 * df[exposure_col] * df[rate_col]
    moment_2_p2 = df[amount_col] ** 2 * df[exposure_col] * df[rate_col] ** 2
    moment_3_p1 = df[amount_col] ** 3 * df[exposure_col] * df[rate_col]
    moment_3_p2 = df[amount_col] ** 3 * df[exposure_col] * df[rate_col] ** 2
    moment_3_p3 = df[amount_col] ** 3 * df[exposure_col] * df[rate_col] ** 3

    # add the moments to the dataframe
    df["moment_1"] = moment_1
    df["moment_2_p1"] = moment_2_p1
    df["moment_2_p2"] = moment_2_p2
    df["moment_3_p1"] = moment_3_p1
    df["moment_3_p2"] = moment_3_p2
    df["moment_3_p3"] = moment_3_p3

    return df
