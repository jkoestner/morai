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
