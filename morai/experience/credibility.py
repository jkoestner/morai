"""Credibility measures."""

from scipy import stats

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def limited_fluctuation(df, measure, p=0.90, r=0.05, sd=1, u=1):
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

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    # get the number of observations for full credibility
    z_score = stats.norm.ppf(1 - (1 - p) / 2)
    full_credibility = (z_score / r) ** 2 * (sd**2 / u)
    logger.info(
        f"Full credibility threshold "
        f"using 'limited fluctuation': {full_credibility}. "
        f"Using probability: {p} and range: {r}."
    )

    # calculate the credibility and cap at 1
    df["credibility_lf"] = ((df[measure] / full_credibility) ** 0.5).clip(upper=1)

    return df


def asymptotic(df, measure, k):
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
    k : float
        Constant for the credibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    # calculate the credibility
    logger.info(f"Using 'asymptotic credibility' with constant: {k}.")
    df["credibility_as"] = df[measure] / (df[measure] + k)

    return df


def vm20_buhlmann(df, amount_col, rate_col, exposure_col=None):
    """
    Determine the credibility of a measure using the SOA VM-20 method.

    The SOA provides an approximation for the Bühlmann Empirical Bayesian Method.

    fomula to calculate z:
    z = A / (A + ((1.090 * B) - (1.204 * C)) / (0.019604 * A))

    The z value can then be used to blend a measure with a prior measure.
    new_estimate = z * measure + (1 - z) * prior

    A = Σ[(face_amount) * (exposure) * (mortality rate)]
    B = Σ[(face_amount)^2 * (exposure) * (mortality rate)]
    C = A^2

    Notes
    -----
    If exposure is not provided, it is assumed to be 1. This will underestimate
    the variance, however will be relatively close.

    Needs to be based on seriatim data and not aggregated data.

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
        Column name of the amount.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    # assign values and check columns
    logger.info("Using 'SOA VM-20 credibility'")
    # check if the columns exist
    missing_cols = [col for col in [amount_col, rate_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # parameters
    amount = df[amount_col]
    rate = df[rate_col]
    if exposure_col is None:
        logger.warning(
            "Using approximation for credibility, due to the exposure "
            "string was not provided."
        )
        exposure = 1
    else:
        exposure = df[exposure_col]

    # calculate the vm20 parameters
    a = amount * exposure * rate
    b = amount**2 * exposure * rate
    c = a**2

    # calculate the credibility
    z = a / (a + ((1.090 * b) - (1.204 * c)) / (0.019604 * a))
    df["credibility_vm20"] = z

    return df
