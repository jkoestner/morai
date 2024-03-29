"""Experience study model."""

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def normalize(df, features, numerator, denominator=None):
    """
    Normalize a column (numerator) based on a number of features.

    Normalizing over the features is a crude method to adjust data for differences
    in the feature groups. when the features are independent, this method is
    appropriate. When the features are not independent, this method will blend
    the effects.

    Tip:
    -----
    When normalizing using a denominator, you should use the denominator of what
    calculated the rate. For example, if you are normalizing a mortality rate,
    you should use the exposure as the denominator.

    a/e = use expected as denominator
    a/o = use exposure as denominator

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
        DataFrame with additional column for normalized values with prefix '_norm'.


    """
    numerator_norm = f"{numerator}_norm"
    df[numerator_norm] = df[numerator]
    df = calc_relative_risk(
        df=df, features=features, numerator=numerator, denominator=denominator
    )
    df[numerator_norm] = df[numerator_norm] / df["risk"]
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

    grouped_df = (
        df.groupby(features, observed=True)[[numerator, denominator]]
        .sum()
        .reset_index()
    )
    total_ratio = grouped_df[numerator].sum() / grouped_df[denominator].sum()
    grouped_df["risk"] = (grouped_df[numerator] / grouped_df[denominator]) / total_ratio

    df = df.merge(grouped_df[[*features, "risk"]], on=features, how="left")
    df = df.drop(columns=["temp_one"], errors="ignore")

    return df
