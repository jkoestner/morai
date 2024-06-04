"""Credibility measures."""

from scipy import stats

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def limited_fluctuation(df, measure, p=0.90, r=0.05, sd=1, u=1, groupby_cols=None):
    """
    Determine the credibility of a measure based on limited fluctuation.

    This is also known as "classical credibility". It is a method to determine
    the probablity (p) that the measure is within a certain range (r) of the
    unknown true value. This then determines the credibility of the measure (z).

    full credibility threshold:
    full_credibility = (z_score / r)^2 * (sd^2 / u)

    Square root fomula to calculate z:
    z = min(1, sqrt(n / full_credibility))

    The z value can then be used to blend a measure with a prior measure.
    new_estimate = z * measure + (1 - z) * prior

    Strengths:
    ----------
    - simple to calculate

    Weaknesses:
    ------------
    - the prior measure may not be applicable
    - the prior measure may not be fully credible
    - counts can overstate the credibility
    - credibility reaches 1 prematurely instead of assymptotically

    Reference:
    ----------
    https://www.soa.org/globalassets/assets/files/resources/tables-calcs-tools/credibility-methods-life-health-pensions.pdf
    https://openacttexts.github.io/Loss-Data-Analytics/ChapCredibility.html#limited-fluctuation-credibility

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    measure : str
        Column name of the measure.
    p : float, optional
        Probability of the measure being within the range.
        Default is 90% which equates to a z-score of 1.645.
    r : float, optional
        The range the measure is within.
    sd : float, optional
        Standard deviation of the measure. (Assuming Poisson if not provided.)
    u : float, optional
        Mean of the measure. (Assuming Poisson if not provided.)
    groupby_cols : list, optional
        Columns to group by before calculating the credibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    # get the number of observations for full credibility
    z_score = stats.norm.ppf(1 - (1 - p) / 2)
    full_credibility = (z_score / r) ** 2 * (sd**2 / u)
    logger.info(
        f"Credibility calculated using 'limited fluctuation' on '{measure}'.\n"
        f"Dataframe does not need to be seriatim.\n"
        f"Full credibility threshold: {full_credibility}.\n"
        f"Probability: {p}\n"
        f"Range: {r}."
    )
    if groupby_cols:
        df = df.groupby(groupby_cols)[[measure]].sum().reset_index()

    # calculate the credibility and cap at 1
    df["credibility_lf"] = ((df[measure] / full_credibility) ** 0.5).clip(upper=1)

    return df


def asymptotic(df, measure, k=270, groupby_cols=None):
    """
    Determine the credibility of a measure using asymptotic credibility.

    fomula to calculate z, where k is chosen subjectively by practitioner:
    z = n / (n + k)

    The z value can then be used to blend a measure with a prior measure.
    new_estimate = z * measure + (1 - z) * prior

    Strengths:
    ----------
    - simple to calculate
    - credibility reaches 1 assymptotically

    Weaknesses:
    ------------
    - not statistically based

    Reference:
    ----------
    https://www.soa.org/globalassets/assets/files/resources/tables-calcs-tools/credibility-methods-life-health-pensions.pdf

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    measure : str
        Column name of the measure.
    k : float, optional
        Constant for the credibility.
        Default is 270.
            270 was chosen as the default because this will produce 50% credibility
            at the same point that 50% would be calculated using the
            limited fluctuation method.
    groupby_cols : list, optional
        Columns to group by before calculating the credibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    # calculate the credibility
    logger.info(
        f"Credibility calculated using 'asymptotic' on '{measure}'.\n"
        f"Dataframe does not need to be seriatim.\n"
        f"Constant k: {k}."
    )
    if groupby_cols:
        df = df.groupby(groupby_cols)[[measure]].sum().reset_index()
    df["credibility_as"] = df[measure] / (df[measure] + k)

    return df


def vm20_buhlmann(df, amount_col, rate_col, exposure_col=None, groupby_cols=None):
    """
    Determine the credibility of a measure using the SOA VM-20 method.

    The SOA provides an approximation for the Bühlmann Empirical Bayesian Method.

    fomula to calculate z:
    z = A / (A + ((1.090 * B) - (1.204 * C)) / (0.019604 * A))

    The z value can then be used to blend a measure with a prior measure.
    new_estimate = z * measure + (1 - z) * prior

    A = Σ[(face_amount) * (exposure) * (mortality rate)]
    B = Σ[(face_amount)^2 * (exposure) * (mortality rate)]
    C = Σ[(face_amount)^2 * (exposure)^2 * (mortality rate)^2]

    Notes
    -----
    If exposure is not provided, it is assumed to be 1.

    Should be based on seriatim data.

    Strengths
    ---------
    - simple to calculate

    Weaknesses
    ----------
    - approximation

    References
    ----------
      https://content.naic.org/pbr_data.htm
        2024 Valuation Manual, Section VM-20, Subsection 9.C.5.a (page 20-59)
      https://www.soa.org/493869/globalassets/assets/files/research/projects/research-cred-theory-pract.pdf
        describes the development of the approximation method

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    amount_col : str
        Column name of the 'amount' field.
    rate_col : str
        Column name of the 'rate' field.
    exposure_col : str
        Column name of the 'exposure' field.
    groupby_cols : list, optional
        Columns to group by before calculating the credibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    logger.info(
        "Credibility calculated using 'SOA VM-20'.\n" "Dataframe should be seriatim.\n"
    )
    vm20_df = df.copy()
    # check if the columns exist
    if groupby_cols is None:
        vm20_df["aggregate"] = "all"
        groupby_cols = ["aggregate"]

    missing_cols = [
        col
        for col in [amount_col, rate_col, *groupby_cols]
        if col not in vm20_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # parameters
    if exposure_col is None:
        logger.warning(
            "Using approximation for credibility, due to the exposure "
            "string was not provided."
        )
        vm20_df["exposure"] = 1

    # calculate the vm20 parameters
    vm20_df["a"] = vm20_df["amount"] * vm20_df["exposure"] * vm20_df["rate"]
    vm20_df["b"] = vm20_df["amount"] ** 2 * vm20_df["exposure"] * vm20_df["rate"]
    vm20_df["c"] = vm20_df["a"] ** 2

    # group by and sum the vm20 parameters
    vm20_df = vm20_df.groupby(groupby_cols)[["a", "b", "c"]].sum().reset_index()

    # calculate the credibility
    vm20_df["credibility_vm20"] = vm20_df["a"] / (
        vm20_df["a"]
        + ((1.090 * vm20_df["b"]) - (1.204 * vm20_df["c"])) / (0.019604 * vm20_df["a"])
    )

    return vm20_df
