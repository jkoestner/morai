"""
Experience study model.

There are several functions related to experience studies in this module.
This includes (exposures, actuals, variance, moments, etc...)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def create_study(
    df: pd.DataFrame,
    bos: str,
    eos: str,
    study_decrement: str,
    study_frequency: str = "annually",
    mapping: dict | None = None,
    get_exposures: bool = True,
    exposure_method: str = "annual",
    calendar_exposure: bool = True,
    get_actuals: bool = True,
) -> pd.DataFrame:
    """
    Create an experience study including exposures and actuals.

    The dataframe expects to have the following columns:
    - termination_date: the date of termination of the policy
    - termination_reason: the reason for termination of the policy
    - issue_date: the date of issue of the policy

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    bos : str
        Beginning of the study.
    eos : str
        End of the study.
    study_decrement : str
        Decrement under study, for example "death".
    study_frequency : str, optional default="annually"
        Study period to calculate exposures for.
        The available options are "annually", "semi-annually",
        "quarterly", "monthly", "weekly", or "daily"
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names
        (termination_date, termination_reason, issue_date, bos_date, eos_date).
    get_exposures : bool, optional default=True
        Wether to add exposures to the study
    exposure_method : str, optional default="annual"
        One of "annual", "distributed", or "exact".
    calendar_exposure : bool, optional default=True
        Whether to use calendar year days (365/366) or policy year days as denominator.
    get_actuals : bool, optional default=True
        Wether to add actuals to the study

    Returns
    -------
    exposures : pd.DataFrame
        DataFrame with additional columns for exposure.
          - bos_date: the beginning of the study period as a datetime
          - eos_date: the end of the study period as a datetime
          - anniversary_date: the anniversary date of the policy in the
            study period
          - policy_dur: the policy duration in years for the study period,
            based on the anniversary date
          - anniversary_position: whether the anniversary date is before or
            on/after the study period (before, on_after)

    """
    study_df = format_study_df(
        df=df,
        bos=bos,
        eos=eos,
        study_decrement=study_decrement,
        study_frequency=study_frequency,
        mapping=mapping,
    )
    if get_exposures:
        study_df["exposure"] = calc_exposures(
            study_df=study_df,
            study_decrement=study_decrement,
            exposure_method=exposure_method,
            calendar_exposure=calendar_exposure,
            mapping=mapping,
        )
    if get_actuals:
        study_df["actuals"] = calc_actuals(
            study_df=study_df, study_decrement=study_decrement, mapping=mapping
        )

    return study_df


def format_study_df(
    df: pd.DataFrame,
    bos: str,
    eos: str,
    study_decrement: str,
    study_frequency: str = "annually",
    mapping: dict | None = None,
) -> pd.DataFrame:
    """
    Format a policy-level df to a study-level df.

    The dataframe expects to have the following columns:
    - termination_date: the date of termination of the policy
    - termination_reason: the reason for termination of the policy
    - issue_date: the date of issue of the policy

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    bos : str
        Beginning of the study.
    eos : str
        End of the study.
    study_decrement : str
        Decrement under study, for example "death".
    study_frequency : str, optional default="annually"
        Study period to calculate exposures for.
        The available options are "annually", "semi-annually",
        "quarterly", "monthly", "weekly", or "daily"
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names
        (termination_date, termination_reason, issue_date, bos_date, eos_date).

    Returns
    -------
    exposures : pd.DataFrame
        DataFrame with additional columns for exposure.
          - bos_date: the beginning of the study period as a datetime
          - eos_date: the end of the study period as a datetime
          - anniversary_date: the anniversary date of the policy in the
            study period
          - policy_dur: the policy duration in years for the study period,
            based on the anniversary date
          - anniversary_position: whether the anniversary date is before or
            on/after the study period (before, on_after)

    """
    shape_before = df.shape
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    issue_date_col = "issue_date"
    bos_date_col = "bos_date"
    eos_date_col = "eos_date"
    anniversary_date_col = "anniversary_date"

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        issue_date_col = mapping.get("issue_date", issue_date_col)
        bos_date_col = mapping.get("bos_date", bos_date_col)
        eos_date_col = mapping.get("eos_date", eos_date_col)
        anniversary_date_col = mapping.get("anniversary_date", anniversary_date_col)

    # validations
    # date checks
    # eos date is less than or equal to bos date
    if pd.to_datetime(eos) <= pd.to_datetime(bos):
        raise ValueError(
            f"The eos_date '{eos}' needs to be after the bos_date '{bos}'."
        )
    # missing columns
    missing_cols = [
        col
        for col in [termination_date_col, termination_reason_col, issue_date_col]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # convert dates to datetime
    df[termination_date_col] = pd.to_datetime(df[termination_date_col], errors="coerce")
    df[issue_date_col] = pd.to_datetime(df[issue_date_col], errors="coerce")
    bos_date = pd.to_datetime(bos)
    eos_date = pd.to_datetime(eos)

    # set up the study_periods
    study_periods = _get_study_periods(
        bos_date=bos_date, eos_date=eos_date, study_frequency=study_frequency
    )

    # logging
    if not study_periods:
        raise ValueError(
            f"No study periods generated. Please check the "
            f"bos: `{bos}`, eos: `{eos}`, and study_frequency: `{study_frequency}`."
        )
    logger.info("formatting df...")
    logger.info(
        f"study periods: "
        f"`{study_periods[0][0].date()}` to `{study_periods[-1][1].date()}`"
    )
    logger.info(f"study frequency: `{study_frequency}`")

    # loop through study periods to create study_df
    df = df.reset_index(names="_idx")  # to keep track policy-level id for sorting
    dfs = []
    for study_period in study_periods:
        _df_period = df.copy()

        # create new columns
        _df_period[bos_date_col] = study_period[0]
        _df_period[eos_date_col] = study_period[1]
        _df_period[anniversary_date_col] = pd.to_datetime(
            {
                "year": _df_period[bos_date_col].dt.year,
                "month": _df_period[issue_date_col].dt.month,
                "day": _df_period[issue_date_col].dt.day,
            },
            errors="coerce",
        )

        # remove policies that will have zero exposure for the year
        # - termination before prior anniversary
        # - or issue date after eos
        _df_period = _df_period[
            ~(
                (
                    _df_period[termination_date_col]
                    < _df_period[anniversary_date_col] - pd.DateOffset(years=1)
                )
                | (_df_period[issue_date_col] > _df_period[eos_date_col])
            )
        ].copy()
        if _df_period.empty:
            continue

        dur = (
            _df_period[anniversary_date_col].dt.year
            - _df_period[issue_date_col].dt.year
        )

        # creating before/after records
        # before
        _before = _df_period.assign(policy_dur=dur, anniversary_position="before")
        term_b = _before[termination_date_col]
        is_study_dec_b = _before[termination_reason_col] == study_decrement

        keep_before = (
            # anniversary after bos
            (_before[anniversary_date_col] > _before[bos_date_col])
            # not issued newly issued
            & (_before["policy_dur"] != 0)
            # inforce or (termination after_on bos) or (study decrement)
            # - study decrements before bos are kept for distributed
            & (term_b.isna() | (term_b >= _before[bos_date_col]) | is_study_dec_b)
        )
        _before = _before[keep_before]

        # after
        _after = _df_period.assign(policy_dur=dur + 1, anniversary_position="on_after")
        term_a = _after[termination_date_col]
        is_study_dec_a = _after[termination_reason_col] == study_decrement

        keep_after = (
            # anniversary before or on eos
            (_after[anniversary_date_col] <= _after[eos_date_col])
            # inforce or (terminated after_on anniversary)
            & (term_a.isna() | (term_a >= _after[anniversary_date_col]))
            # inforce or (terminated after_on bos) or (study decrement)
            # - study decrements before bos are kept for distributed
            & (term_a.isna() | (term_a >= _after[bos_date_col]) | is_study_dec_a)
        )
        _after = _after[keep_after]

        dfs.extend([_before, _after])

    if not dfs:
        raise ValueError("No policies have exposure in the study period.")

    study_df = pd.concat(dfs, ignore_index=True)

    # sort values
    study_df = study_df.sort_values(
        by=["_idx", "policy_dur", bos_date_col]
    ).reset_index(drop=True)
    study_df = study_df.drop(columns=["_idx"])

    shape_after = study_df.shape
    logger.info(f"shape before: {shape_before}, shape_after: {shape_after}")

    return study_df


def calc_exposures(
    study_df: pd.DataFrame,
    study_decrement: str,
    exposure_method: str = "annual",
    calendar_exposure: bool = True,
    mapping: dict | None = None,
) -> pd.Series:
    """
    Calculate the exposure for each row in the study DataFrame.

    The three exposure methods differ only in how they treat policies that decrement
    under the study:
    - annual: before gets proportional days, after gets a full year (Balducci)
    - distributed: before gets proportional days, after gets proportional days (UDD)
    - exact: both before and after get exact days to decrement (constant force)
      - mx ≈ ux
      - qx = 1 - exp(-ux)
      - ux = -log(1-ux)

    Expects the DataFrame to already have these columns:
    - termination_date, termination_reason, issue_date
    - anniversary_date
    - bos_date, eos_date

    Notes
    -----
    - `anniversary_date` is included in the `exposure_after` column.
    - `termination_date` is included in the exposure.
    - when the study_frequency is daily the qx that is calculated at that frequency
      is basically the force of mortality (ux). If the exposure is grouped at a
      higher frequency than daily to calculate the qx it will not equal ux.

    References
    ----------
    - https://www.soa.org/resources/tables-calcs-tools/experience-study-tool/
    - https://www.soa.org/globalassets/assets/files/research/experience-study-calculations.pdf

    Parameters
    ----------
    study_df : pd.DataFrame
        DataFrame with the data.
    study_decrement : str
        Decrement under study, for example "death".
    exposure_method : str, optional default="annual"
        One of "annual", "distributed", or "exact".
    calendar_exposure : bool, optional default=True
        Whether to use calendar year days (365/366) or policy year days as denominator.
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names.

    Returns
    -------
    exposure : pd.Series
        Series with the exposure for each row in the DataFrame.

    """
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    issue_date_col = "issue_date"
    anniversary_date_col = "anniversary_date"
    bos_date_col = "bos_date"
    eos_date_col = "eos_date"
    if "id" in study_df.columns:
        idx_col = "id"
    else:
        idx_col = None

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        issue_date_col = mapping.get("issue_date", issue_date_col)
        anniversary_date_col = mapping.get("anniversary_date", anniversary_date_col)
        bos_date_col = mapping.get("bos_date", bos_date_col)
        eos_date_col = mapping.get("eos_date", eos_date_col)
        idx_col = mapping.get("idx", idx_col)

    # validations
    # missing columns
    required_cols = [
        termination_date_col,
        termination_reason_col,
        issue_date_col,
        anniversary_date_col,
        bos_date_col,
        eos_date_col,
    ]
    if idx_col is not None:
        required_cols.append(idx_col)
    missing_cols = [col for col in required_cols if col not in study_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )

    # logging
    logger.info("getting exposures...")
    rate_type = "qx" if exposure_method in ["annual", "distributed"] else "mx"
    logger.info(f"exposure method: `{exposure_method}` - rate type: `{rate_type}`")
    logger.info(f"calendar exposure: `{calendar_exposure}`")
    logger.info(f"study decrement: `{study_decrement}`")

    # creating temporary series for calculations
    next_anniversary = pd.to_datetime(
        {
            "year": study_df[anniversary_date_col].dt.year + 1,
            "month": study_df[issue_date_col].dt.month,
            "day": study_df[issue_date_col].dt.day,
        },
        errors="coerce",
    )
    prior_anniversary = pd.to_datetime(
        {
            "year": study_df[anniversary_date_col].dt.year - 1,
            "month": study_df[issue_date_col].dt.month,
            "day": study_df[issue_date_col].dt.day,
        },
        errors="coerce",
    )
    is_before = study_df["anniversary_position"] == "before"
    is_decrement = study_df[termination_reason_col] == study_decrement

    # calculate date differences needed for exposure calculations
    anniversary_minus_bos = (
        study_df[anniversary_date_col] - study_df[bos_date_col]
    ).dt.days
    termination_minus_bos = (
        study_df[termination_date_col] - study_df[bos_date_col]
    ).dt.days + 1
    eos_minus_anniversary = (
        study_df[eos_date_col] - study_df[anniversary_date_col]
    ).dt.days + 1
    eos_minus_bos = (study_df[eos_date_col] - study_df[bos_date_col]).dt.days + 1
    termination_minus_anniversary = (
        study_df[termination_date_col] - study_df[anniversary_date_col]
    ).dt.days + 1
    next_anniversary_minus_anniversary = (
        next_anniversary - study_df[anniversary_date_col]
    ).dt.days
    next_anniversary_minus_bos = (next_anniversary - study_df[bos_date_col]).dt.days

    # logic conditions
    terminated = study_df[termination_date_col].notna()
    term_before_bos = study_df[termination_date_col] < study_df[bos_date_col]
    term_in_before = (
        (study_df[termination_date_col] >= study_df[bos_date_col])
        & (study_df[termination_date_col] <= study_df[eos_date_col])
        & (study_df[termination_date_col] <= study_df[anniversary_date_col])
    )
    term_in_after = (
        (study_df[termination_date_col] >= study_df[bos_date_col])
        & (study_df[termination_date_col] <= study_df[eos_date_col])
        & (study_df[termination_date_col] >= study_df[anniversary_date_col])
    )
    issued_in_period = (
        study_df[issue_date_col].dt.year == study_df[eos_date_col].dt.year
    )

    # denominators
    if calendar_exposure:  # calendar
        calendar_days = np.where(study_df[eos_date_col].dt.is_leap_year, 366, 365)
        total_days = calendar_days
        # update variables that are affected by leap
        leap_adj = (
            next_anniversary.dt.is_leap_year & (next_anniversary.dt.month > 2)
        ).astype(int)
        next_anniversary_minus_anniversary -= leap_adj
        next_anniversary_minus_bos -= leap_adj
    else:  # policy
        policy_before = (study_df[anniversary_date_col] - prior_anniversary).dt.days
        policy_after = (next_anniversary - study_df[anniversary_date_col]).dt.days
        total_days = np.where(
            is_before,
            policy_before,
            policy_after,
        )

    # numerators
    # inforce policies
    inforce_days = np.where(
        is_before,
        np.minimum(anniversary_minus_bos, eos_minus_bos),
        np.minimum(eos_minus_anniversary, eos_minus_bos),
    )

    # non-study decrements
    other_decrement_days = np.where(
        is_before,
        termination_minus_bos,
        np.minimum(termination_minus_anniversary, termination_minus_bos),
    )

    # study decrements - method-specific
    if exposure_method == "annual":
        study_decrement_days = np.where(
            is_before,
            anniversary_minus_bos,
            np.minimum(next_anniversary_minus_anniversary, next_anniversary_minus_bos),
        )
        term_in_before_for_decrement = term_in_before
    elif exposure_method == "distributed":
        study_decrement_days = np.where(
            is_before,
            np.minimum(anniversary_minus_bos, eos_minus_bos),
            np.minimum(eos_minus_anniversary, eos_minus_bos),
        )
        # the distributed method has before exposure for policies that terminated in the
        # prior study periods
        term_in_before_for_decrement = (
            study_df[termination_date_col] >= prior_anniversary
        ) & (study_df[termination_date_col] < study_df[anniversary_date_col])
    else:  # exact
        study_decrement_days = np.where(
            is_before,
            np.minimum(termination_minus_bos, anniversary_minus_bos),
            np.minimum(termination_minus_anniversary, termination_minus_bos),
        )
        term_in_before_for_decrement = term_in_before

    # exposure conditions
    # zero exposure
    zero = (
        # before anniversary
        (is_before & (study_df[anniversary_date_col] <= study_df[bos_date_col]))
        | (is_before & issued_in_period)
        | (
            is_before
            & terminated
            & term_before_bos
            & ~(
                (exposure_method == "distributed")
                & is_decrement
                & is_before
                & (study_df[termination_date_col] >= prior_anniversary)
                & (study_df[termination_date_col] < study_df[bos_date_col])
            )
        )
        # on_after anniversary
        | (~is_before & (study_df[anniversary_date_col] > study_df[eos_date_col]))
        | (
            ~is_before
            & terminated
            & (study_df[termination_date_col] < study_df[anniversary_date_col])
        )
        | (
            ~is_before
            & terminated
            & (study_df[termination_date_col] < study_df[bos_date_col])
            & ~(
                (exposure_method == "distributed")
                & is_decrement
                & (study_df[termination_date_col] >= prior_anniversary)
            )
        )
    )

    # study decrement
    study_decrement_cond = is_decrement & (
        (is_before & term_in_before_for_decrement & ~issued_in_period)
        | (~is_before & term_in_after)
    )

    # other decrement
    other_decrement_cond = (
        ~is_decrement
        & terminated
        & ((is_before & term_in_before) | (~is_before & term_in_after))
    )

    # calculate exposures
    exposures = pd.Series(
        np.select(
            [zero, study_decrement_cond, other_decrement_cond],
            [0, study_decrement_days / total_days, other_decrement_days / total_days],
            default=inforce_days / total_days,  # inforce
        ),
        index=study_df.index,
    )

    # check for errors
    num_negative = (exposures < 0).sum()
    if num_negative > 0:
        logger.error(f"negative exposure: {num_negative}, this should not happen.")

    if idx_col:
        grouped_exposure = (
            study_df.assign(exposure=exposures.values)
            .groupby(["id", "policy_dur"], observed=True, sort=False)["exposure"]
            .sum()
            .reset_index()
        )
        num_zero_grouped = len(grouped_exposure[grouped_exposure["exposure"] == 0])
        if num_zero_grouped > 0:
            logger.warning(
                f"zero exposure - grouped: {num_zero_grouped}, this may be due to "
                f"using a partial year. It's worth reviewing."
            )

    num_zero_seriatim = (exposures == 0).sum()
    if num_zero_seriatim > 0:
        logger.debug(
            f"zero exposure - seriatim: {num_zero_seriatim}, "
            f"This is likely due to the distributed exposure would be non-zero "
            f"at these cells."
        )

    return exposures


def calc_actuals(
    study_df: pd.DataFrame,
    study_decrement: str,
    mapping: dict[str, Any] | None = None,
) -> pd.Series:
    """
    Get the actuals for the decrement under study.

    Parameters
    ----------
    study_df : pd.DataFrame
        DataFrame with the data.
    study_decrement : str
        Decrement under study, for example "death".
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names.

    Returns
    -------
    actuals : pd.Series
        Series with 1 if the decrement under study occurred in the study period,
        0 otherwise.

    """
    # handle mapping
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    anniversary_date_col = "anniversary_date"
    bos_date_col = "bos_date"
    eos_date_col = "eos_date"

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        anniversary_date_col = mapping.get("anniversary_date", anniversary_date_col)
        bos_date_col = mapping.get("bos_date", bos_date_col)
        eos_date_col = mapping.get("eos_date", eos_date_col)

    logger.info("getting actuals...")

    # actuals are 1 if the termination reason is the study decrement and
    # termination date is within the study period, 0 otherwise.
    actuals = (
        (study_df[termination_reason_col] == study_decrement)
        & (study_df[bos_date_col] <= study_df[termination_date_col])
        & (study_df[termination_date_col] <= study_df[eos_date_col])
        & (
            (
                (study_df["anniversary_position"] == "before")
                & (study_df[termination_date_col] < study_df[anniversary_date_col])
            )
            | (
                (study_df["anniversary_position"] == "on_after")
                & (study_df[termination_date_col] >= study_df[anniversary_date_col])
            )
        )
    ).astype(int)

    logger.info(f"total actuals: {actuals.sum():,}")

    return actuals


def _get_study_periods(
    bos_date: pd.Timestamp,
    eos_date: pd.Timestamp,
    study_frequency: str = "annually",
) -> list[Any]:
    """
    Generate study periods between bos and eos based on frequency.

    Uses pandas date_range to generate the start dates of the periods and
    then calculates the end dates using an offset.

    Parameters
    ----------
    bos_date : pd.Timestamp
        Beginning of study date.
    eos_date : pd.Timestamp
        End of study date.
    study_frequency : str, optional
        Frequency of study periods. Default is "annually".

    Returns
    -------
    periods : list
        list of (bos, eos) tuples for each period.

    """
    freq_map = {
        "annually": "YS",
        "semi-annually": "6MS",
        "quarterly": "QS",
        "monthly": "MS",
        "weekly": "W-MON",
        "daily": "D",
    }

    date_offset_kwargs = {
        "annually": {"years": 1},
        "semi-annually": {"months": 6},
        "quarterly": {"months": 3},
        "monthly": {"months": 1},
        "weekly": {"weeks": 1},
    }

    if study_frequency not in freq_map:
        raise ValueError(
            f"Unsupported frequency: {study_frequency}, "
            f"supported frequencies are: {', '.join(freq_map.keys())}."
        )

    if study_frequency == "daily":
        return [(d, d) for d in pd.date_range(bos_date, eos_date, freq="D")]

    starts = pd.date_range(bos_date, eos_date, freq=freq_map[study_frequency])
    periods = []
    for start in starts:
        end = min(
            start
            + pd.tseries.offsets.DateOffset(**date_offset_kwargs[study_frequency])
            - pd.Timedelta(days=1),
            eos_date,
        )
        periods.append((start, end))

    return periods


def normalize(
    df: pd.DataFrame,
    features: list[str],
    normalize_col: list[str] | str,
    weight_col: list[str] | str | None = None,
    add_norm_col: bool | None = False,
    ratio: bool | None = False,
    relative_to: str | None = "aggregate",
    relative_cols: list[str] | str | None = None,
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
    df: pd.DataFrame | pl.LazyFrame,
    features: list[str],
    risk_col: list[str] | str,
    weight_col: list[str] | str | None = None,
    ratio: bool | None = False,
    relative_to: str | None = "aggregate",
    relative_cols: list[str] | str | None = None,
    subset_dict: dict[str, Any] | None = None,
) -> pd.DataFrame | pl.LazyFrame:
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
    if relative_cols is None:
        relative_cols = ["_temp_relative"]
        use_temp_relative = True
    else:
        use_temp_relative = False

    if weight_col is None:
        weight_col = "_temp_weight"
        use_temp_weight = True
    else:
        use_temp_weight = False

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

        # fallback for groups not represented in subset
        if subset_dict:
            if ratio:
                fallback = base_df.select(
                    (pl.col("risk_sum").sum() / pl.col("weight_sum").sum()).alias(
                        "fallback"
                    )
                )
            else:
                fallback = base_df.select(
                    (
                        (pl.col("risk_sum") * pl.col("weight_sum")).sum()
                        / pl.col("weight_sum").sum()
                    ).alias("fallback")
                )
            grouped_df = grouped_df.with_columns(
                pl.col("baseline_ratio")
                .fill_null(fallback.collect().item())
                .alias("baseline_ratio")
            )

        # calculate relative risk
        grouped_df = grouped_df.with_columns(
            (pl.col("risk_numerator") / pl.col("baseline_ratio")).alias("relative_risk")
        )

        # handle zero by replacing with 1.0
        grouped_df = grouped_df.with_columns(
            pl.when(pl.col("relative_risk") == 0)
            .then(pl.lit(1.0))
            .otherwise(pl.col("relative_risk"))
            .alias("relative_risk")
        )

        if relative_to == "reference":
            reference_risks = grouped_df.group_by(
                relative_cols, maintain_order=True
            ).agg(pl.col("relative_risk").min().alias("reference_risk"))

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

        # fallback for groups not represented in subset
        if subset_dict:
            if ratio:
                fallback = base_df["risk_sum"].sum() / base_df["weight_sum"].sum()
            else:
                fallback = base_df["weighted_risk"].sum() / base_df["weight_sum"].sum()
            grouped_df["baseline_ratio"] = grouped_df["baseline_ratio"].fillna(fallback)

        # calculate relative risk
        grouped_df["relative_risk"] = (
            grouped_df["risk_numerator"] / grouped_df["baseline_ratio"]
        )

        # handle zero by replacing with 1.0
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
    amount_col: str | None = None,
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
    amount_col: str | None = None,
    sffx: str | None = None,
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
    Add to the model data the qx, expected amount, and ae using the predictions.

    Parameters
    ----------
    model_data : pd.DataFrame
        DataFrame with the model data.
    predictions : pd.DataFrame
        DataFrame with the predictions.
    model_name : str
        Name of the model, which will be used as a suffix for the new columns.
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
