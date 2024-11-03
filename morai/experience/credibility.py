"""
Credibility measures.

Todo:
----
- add in more resources including buhlmann credibility
- CREDIBILITY - HOWARD C. MAHLER AND CURTIS GARY DEAN
   - https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/bf4517bb19eee4cec125704600554ce6/$FILE/chapter8.pdf
- pymc
   - https://github.com/pymc-devs/pymc?tab=readme-ov-file
- pymc notebooks
   - https://github.com/aloctavodia/BAP3/blob/main/exercises/Chapter_01.ipynb
- A CREDIBILITY APPROACH TO MORTALITY RISK - M.R. HARDY AND H.H. PANJER
   - https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/769998e0a65ea348c1257052003eb94f/3daaa4e8b3c51c64c12570c20061400e/$FILE/Hardy%20Panjer.pdf
- julia actuarial project
   - https://juliaactuary.org/posts/bayes-vs-limited-fluctuation/index.html#limited-fluctuation-credibility
- Chapter 9 Experience Rating Using Credibility
   - https://openacttexts.github.io/Loss-Data-Analytics/ChapCredibility.html
- https://www.soa.org/globalassets/assets/files/resources/tables-calcs-tools/credibility-methods-life-health-pensions.pdf
- Think Bayes
    - https://allendowney.github.io/ThinkBayes2/

"""

from typing import TYPE_CHECKING, List, Optional

from scipy import stats

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

if TYPE_CHECKING:
    import pandas as pd


def limited_fluctuation(
    df: "pd.DataFrame",
    measure: str,
    p: float = 0.90,
    r: float = 0.05,
    sd: float = 1,
    u: float = 1,
    groupby_cols: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """
    Determine the credibility of a measure based on limited fluctuation.

    Does not need to be seriatim.

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
    - counts can overstate the credibility
    - credibility reaches 1 prematurely instead of assymptotically
    - the prior measure may not be applicable
    - the prior measure may not be fully credible

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
        f"Created column 'credibility_lf'.\n"
        f"Full credibility threshold: {full_credibility:,.1f}\n"
        f"Probability measure within range: {p:,.1f}\n"
        f"Range +/-: {r:,.1f}\n"
        f"Standard deviation: {sd:,.1f}\n"
        f"Mean: {u:,.1f}"
    )
    if groupby_cols:
        df = df.groupby(groupby_cols, observed=True)[[measure]].sum().reset_index()

    # calculate the credibility and cap at 1
    df["credibility_lf"] = ((df[measure] / full_credibility) ** 0.5).clip(upper=1)

    return df


def asymptotic(
    df: "pd.DataFrame",
    measure: str,
    k: float = 270,
    groupby_cols: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """
    Determine the credibility of a measure using asymptotic credibility.

    Does not need to be seriatim.

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
            limited fluctuation method using 90% probability and a range of 5%.
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
        f"Created column 'credibility_as'.\n"
        f"Constant k: {k}."
    )
    if groupby_cols:
        df = df.groupby(groupby_cols, observed=True)[[measure]].sum().reset_index()
    df["credibility_as"] = df[measure] / (df[measure] + k)

    return df


def vm20_buhlmann(
    df: "pd.DataFrame",
    amount_col: str,
    rate_col: str,
    exposure_col: Optional[str] = None,
    groupby_cols: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """
    Determine the credibility of a measure using the SOA VM-20 method.

    Should be based on seriatim data.

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
        "Credibility calculated using 'SOA VM-20'.\n"
        "Created column 'credibility_vm20'.\n"
        "Dataframe should be seriatim.\n"
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
    vm20_df["c"] = (
        vm20_df["amount"] ** 2 * vm20_df["exposure"] ** 2 * vm20_df["rate"] ** 2
    )

    # group by and sum the vm20 parameters
    vm20_df = (
        vm20_df.groupby(groupby_cols, observed=True)[["a", "b", "c"]]
        .sum()
        .reset_index()
    )

    # calculate the credibility
    vm20_df["credibility_vm20"] = vm20_df["a"] / (
        vm20_df["a"]
        + ((1.090 * vm20_df["b"]) - (1.204 * vm20_df["c"])) / (0.019604 * vm20_df["a"])
    )

    return vm20_df


def vm20_buhlmann_approx(
    df: "pd.DataFrame",
    a_col: str,
    b_col: str,
    c_col: str,
    groupby_cols: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """
    Determine the credibility of a measure using an SOA VM-20 approximation.

    Does not need to be seriatim.

    The SOA provides an approximation for the Bühlmann Empirical Bayesian Method.

    fomula to calculate z:
    z = A / (A + ((1.090 * B) - (1.204 * C)) / (0.019604 * A))

    The z value can then be used to blend a measure with a prior measure.
    new_estimate = z * measure + (1 - z) * prior

    Using non-seriatim data that includes the momemnts we can approximate the
    parameters A, B, and C, which are used to calculate the credibility.

    A = Σ[(face_amount) * (exposure) * (mortality rate)]
    moment_1 = amount * exposure * rate
    A = Σ[moment_1]

    B = Σ[(face_amount)^2 * (exposure) * (mortality rate)]
    moment_2_p1 = amount^2 * exposure * rate
    B = Σ[moment_2_p1]

    C = Σ[(face_amount)^2 * (exposure)^2 * (mortality rate)^2]
    moment_2_p2 = amount^2 * exposure * rate^2
    C ≈ Σ[moment_2_p2]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    a_col : str
        Column name of the 'a' field.
    b_col : str
        Column name of the 'b' field.
    c_col : str
        Column name of the 'c' field.
    groupby_cols : list, optional
        Columns to group by before calculating the credibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    logger.info(
        "Credibility calculated using 'SOA VM-20 approximation'.\n"
        "Created column 'credibility_vm20_approx'.\n"
        "Dataframe does not need to be seriatim.\n"
    )
    vm20_df = df.copy()
    # check if the columns exist
    if groupby_cols is None:
        vm20_df["aggregate"] = "all"
        groupby_cols = ["aggregate"]

    missing_cols = [
        col
        for col in [a_col, b_col, c_col, *groupby_cols]
        if col not in vm20_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # calculate the vm20 parameters
    vm20_df["a"] = vm20_df[a_col]
    vm20_df["b"] = vm20_df[b_col]
    vm20_df["c"] = vm20_df[c_col]

    # group by and sum the vm20 parameters
    vm20_df = (
        vm20_df.groupby(groupby_cols, observed=True)[["a", "b", "c"]]
        .sum()
        .reset_index()
    )

    # calculate the credibility
    vm20_df["credibility_vm20_approx"] = vm20_df["a"] / (
        vm20_df["a"]
        + ((1.090 * vm20_df["b"]) - (1.204 * vm20_df["c"])) / (0.019604 * vm20_df["a"])
    )

    return vm20_df


def buhlmann(
    df: "pd.DataFrame", measure: str, k: float, groupby_cols: Optional[List[str]] = None
) -> "pd.DataFrame":
    """
    Determine the credibility of a measure using Bühlmann credibility.

    Does not need to be seriatim.

    Bühlmann credibility is a linear approximation of the Bayesian credibility.
    K is calculated by taking the ratio of the EPV/VHM. Where EPV is the expected
    process variance and VHM is the variance of the hypothetical means. The formula to
    calculate credibility is then the same as the asymptotic credibility.

    fomula to calculate z, where k is chosen subjectively by practitioner:
    z = n / (n + k)

    The z value can then be used to blend a measure with a prior measure.
    new_estimate = z * measure + (1 - z) * prior

    Strengths:
    ----------
    - all aspects of the method are clearly spelled out.
    - credibility reaches 1 assymptotically

    Weaknesses:
    ------------
    - requires a lot of data to estimate the parameters

    Reference:
    ----------
    https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/bf4517bb19eee4cec125704600554ce6/$FILE/chapter8.pdf
    https://www.soa.org/globalassets/assets/files/resources/tables-calcs-tools/credibility-methods-life-health-pensions.pdf

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    measure : str
        Column name of the measure.
    k : float, optional
        Constant for the credibility.
    groupby_cols : list, optional
        Columns to group by before calculating the credibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for credibility measures.

    """
    # calculate the credibility
    logger.info(
        f"Credibility calculated using 'bühlmann' on '{measure}'.\n"
        f"Dataframe does not need to be seriatim.\n"
        f"Created column 'credibility_bu'.\n"
        f"Constant k: {k}."
    )
    if groupby_cols:
        df = df.groupby(groupby_cols, observed=True)[[measure]].sum().reset_index()
    df["credibility_bu"] = df[measure] / (df[measure] + k)

    return df
