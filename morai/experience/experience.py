"""Experience study model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def calc_exposure(
    df: pd.DataFrame,
    bos: str,
    eos: str,
    study_decrement: str,
    exposure_method: str = "annual",
    calendar_exposure: bool = True,
    study_frequency: str | None = "annually",
    mapping: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate the exposure for each row in the DataFrame.

    The dataframe expects to have the following columns:
    - termination_date: the date of termination of the policy
    - termination_reason: the reason for termination of the policy
    - issue_date: the date of issue of the policy

    The following columns will be added to the DataFrame:

    Note:
    - `anniversary_date` is included in the `exposure_after` column.
    - `bos_date` should be the beginning of the year.
    - `termination_date` is included in the exposure.
    - exposures by calendar will add to 1, but exposures by policy will be
      slightly more than 1 for leap years.

    Reference:
    - https://www.soa.org/resources/tables-calcs-tools/experience-study-tool/
    - https://www.soa.org/globalassets/assets/files/research/experience-study-calculations.pdf

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    bos : str
        Beginning of the study period.
    eos : str
        End of the study period.
    study_decrement : str
        Decrement under study, for example "death". This is used to determine the
        exposure for policies that terminate during the study period.
    exposure_method : str, optional default="annual"
        Method for calculating exposure.
        Options are "annual", "distributed", or "exact".
    calendar_exposure : bool, optional default=True
        Whether to calculate exposure by calendar year or by policy year.
        This has a very minor impact on the exposure calculation for policies,
        however when there is a leap year, the exposure will either be 1 for a
        calendar or 1 for a policy. It can't be 1 for both.
    study_frequency : str, optional default="annually"
        Study period to calculate exposures for.
        The available options are "annually", "semi-annually",
        "quarterly", "monthly", "weekly", or "daily"
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names.

    Returns
    -------
    study_df : pd.DataFrame
        DataFrame with additional columns for exposure.
          - exposure_before: the exposure before the policy anniversary during the study
          - exposure_after: the exposure after the policy anniversary during the study
          - bos_date: the beginning of the study period as a datetime
          - eos_date: the end of the study period as a datetime

    """
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    issue_date_col = "issue_date"
    bos_date_col = "bos_date"
    eos_date_col = "eos_date"

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        issue_date_col = mapping.get("issue_date", issue_date_col)
        bos_date_col = mapping.get("bos_date", bos_date_col)
        eos_date_col = mapping.get("eos_date", eos_date_col)

    # convert dates to datetime
    df[termination_date_col] = pd.to_datetime(df[termination_date_col], errors="coerce")
    df[issue_date_col] = pd.to_datetime(df[issue_date_col], errors="coerce")
    bos_date = pd.to_datetime(bos)
    eos_date = pd.to_datetime(eos)

    # validations
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
    # not allowed method
    allowed_methods = ["annual", "distributed", "exact"]
    if exposure_method not in allowed_methods:
        raise ValueError(
            f"Invalid exposure method: {exposure_method}. "
            f"Allowed methods are: {', '.join(allowed_methods)}."
        )
    # termination reason not in the data
    unique_termination_reasons = df[termination_reason_col].dropna().unique()
    if study_decrement not in unique_termination_reasons:
        logger.warning(
            f"Study decrement '{study_decrement}' not found in termination reasons. "
            f"Unique termination reasons are: {', '.join(unique_termination_reasons)}."
        )
    # date checks
    # termination date
    if df[termination_date_col].isna().sum() != df[termination_reason_col].isna().sum():
        raise ValueError(
            f"Termination date column '{termination_date_col}' and "
            f"termination reason column '{termination_reason_col}' "
            f"should both be present or both be missing."
        )
    mask = df[termination_date_col].notna()
    if (df.loc[mask, termination_date_col] < df.loc[mask, issue_date_col]).any():
        raise ValueError(
            f"'{termination_date_col}' should be after '{issue_date_col}'."
        )
    # eos date
    if pd.to_datetime(eos) <= pd.to_datetime(bos):
        raise ValueError(
            f"The eos_date '{eos}' needs to be after the bos_date '{bos}'."
        )

    # set up the periods
    study_periods = _get_study_periods(
        bos_date=bos_date, eos_date=eos_date, study_frequency=study_frequency
    )

    # logging
    if not study_periods:
        raise ValueError(
            f"No study periods generated. Please check the "
            f"bos: `{bos}`, eos: `{eos}`, and study_frequency: `{study_frequency}`."
        )
    logger.info(
        f"study periods: "
        f"`{study_periods[0][0].date()}` to `{study_periods[-1][1].date()}`"
    )
    logger.info(f"study frequency: `{study_frequency}`")
    logger.info(f"exposure method: `{exposure_method}`")
    logger.info(f"study decrement: `{study_decrement}`")
    logger.info(f"calendar exposure: `{calendar_exposure}`")
    rate_type = "qx" if exposure_method in ["annual", "distributed"] else "ux"
    logger.info(f"rate type: `{rate_type}`")

    # calculate exposures
    dfs = []

    for study_period in study_periods:
        _df_period = df.copy()

        # create new columns
        _df_period["bos_date"] = study_period[0]
        _df_period["eos_date"] = study_period[1]
        year = study_period[0].year
        _df_period["anniversary_date"] = pd.to_datetime(
            {
                "year": year,
                "month": _df_period[issue_date_col].dt.month,
                "day": _df_period[issue_date_col].dt.day,
            },
            errors="coerce",
        )
        _df_period["next_anniversary_date"] = pd.to_datetime(
            {
                "year": year + 1,
                "month": _df_period[issue_date_col].dt.month,
                "day": _df_period[issue_date_col].dt.day,
            }
        )
        _df_period["prior_anniversary_date"] = pd.to_datetime(
            {
                "year": year - 1,
                "month": _df_period[issue_date_col].dt.month,
                "day": _df_period[issue_date_col].dt.day,
            }
        )
        _df_period["policy_dur_before"] = (
            _df_period["anniversary_date"].dt.year - _df_period[issue_date_col].dt.year
        )
        _df_period["policy_dur_after"] = _df_period["policy_dur_before"] + 1

        # remove policies that will have zero exposure for the year
        _df_period = _df_period[
            ~(
                (
                    _df_period[termination_date_col]
                    < _df_period["prior_anniversary_date"]
                )
                | (_df_period[issue_date_col] > _df_period["eos_date"])
            )
        ].copy()
        # check for any policies
        if _df_period.empty:
            continue

        # calculate exposures
        if exposure_method == "annual":
            _df_period = _annual_exposure(
                _df_period, study_decrement, calendar_exposure, mapping
            )
        elif exposure_method == "distributed":
            _df_period = _dist_exposure(
                _df_period, study_decrement, calendar_exposure, mapping
            )
        elif exposure_method == "exact":
            _df_period = _exact_exposure(
                _df_period, study_decrement, calendar_exposure, mapping
            )
        else:
            raise ValueError(f"Unsupported exposure method: {exposure_method}")

        dfs.append(_df_period)

    if not dfs:
        raise ValueError("No policies have exposure in the study period.")

    study_df = pd.concat(dfs, ignore_index=True)

    # stack before/after columns into 1 column
    id_cols = [c for c in study_df.columns if not c.endswith(("_before", "_after"))]
    before = study_df[[*id_cols, "policy_dur_before", "exposure_before"]].rename(
        columns={"policy_dur_before": "policy_dur", "exposure_before": "exposure"}
    )
    after = study_df[[*id_cols, "policy_dur_after", "exposure_after"]].rename(
        columns={"policy_dur_after": "policy_dur", "exposure_after": "exposure"}
    )
    study_df = pd.concat([before, after], ignore_index=True)

    # remove policies that have zero exposure for the year
    study_df = study_df[study_df["exposure"] != 0].copy()
    # remove temporary columns
    study_df = study_df.drop(
        columns=["next_anniversary_date", "prior_anniversary_date"]
    )

    return study_df


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


def _annual_exposure(
    df: pd.DataFrame,
    study_decrement: str,
    calendar_exposure: bool = True,
    mapping: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate annual exposure.

    The annual exposure method for a policy that decrements under the study puts
    the exposure in the calendar year it terminated in.

    Annual exposure method aligns with Balducci Hypothesis, which means death rates
    decrease through the year.

    The rate calculated using the annual exposure method will be qx (initial rate)

    Expects the DataFrame to already have these columns:
    - termination_date, termination_reason
    - anniversary_date, next_anniversary_date, prior_anniversary_date
    - bos_date, eos_date

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    study_decrement : str
        Decrement under study, for example "death". This is used to determine the
        exposure for policies that terminate during the study period.
    calendar_exposure : bool, optional default=True
        Whether to calculate exposure by calendar year or by policy year.
        This has a very minor impact on the exposure calculation for policies,
        however when there is a leap year, the exposure will either be 1 for a
        calendar or 1 for a policy. It can't be 1 for both.
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names.

    Returns
    -------
    df : pd.DataFrame
        df with exposure_before and exposure_after columns added.

    """
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    issue_date_col = "issue_date"

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        issue_date_col = mapping.get("issue_date", issue_date_col)

    # series for calculations
    anniversary_minus_bos = (df["anniversary_date"] - df["bos_date"]).dt.days
    termination_minus_bos = (df[termination_date_col] - df["bos_date"]).dt.days + 1
    eos_minus_anniversary = (df["eos_date"] - df["anniversary_date"]).dt.days + 1
    eos_minus_bos = (df["eos_date"] - df["bos_date"]).dt.days + 1
    termination_minus_anniversary = (
        df[termination_date_col] - df["anniversary_date"]
    ).dt.days + 1
    calendar_before = (
        366
        if pd.Timestamp(year=df["eos_date"].iloc[0].year, month=12, day=31).is_leap_year
        else 365
    )
    calendar_after = calendar_before
    policy_before = (df["anniversary_date"] - df["prior_anniversary_date"]).dt.days
    policy_after = (df["next_anniversary_date"] - df["anniversary_date"]).dt.days
    if calendar_exposure:
        total_days_before = calendar_before
        total_days_after = calendar_after
    else:
        total_days_before = policy_before
        total_days_after = policy_after

    # calculate exposures
    df["exposure_before"] = np.where(
        # decrement under study - annual exposure
        (df[termination_reason_col] == study_decrement)
        & (df[termination_date_col] >= df["bos_date"])
        & (df[issue_date_col].dt.year != df["eos_date"].dt.year)
        & (df["anniversary_date"] > df["bos_date"]),
        anniversary_minus_bos / total_days_before,
        # not in the study period or issued in the study period
        np.where(
            (df[termination_date_col] < df["bos_date"])
            | (df["anniversary_date"] <= df["bos_date"])
            | (df[issue_date_col].dt.year == df["eos_date"].dt.year),
            0,
            # inforce
            np.where(
                (df[termination_date_col].isna())
                | (df[termination_date_col] > df["anniversary_date"])
                | (df[termination_date_col] > df["eos_date"]),
                np.minimum(anniversary_minus_bos, eos_minus_bos) / total_days_before,
                np.where(
                    # decrement not under study
                    (df[termination_reason_col] != study_decrement)
                    & (df[termination_date_col] <= df["anniversary_date"]),
                    termination_minus_bos / total_days_before,
                    # else, error
                    -1,
                ),
            ),
        ),
    )

    df["exposure_after"] = np.where(
        # decrement under study - annual exposure
        (df[termination_reason_col] == study_decrement)
        & (df[termination_date_col] >= df["anniversary_date"])
        & (df[termination_date_col] <= df["eos_date"]),
        1,
        # not in the study period or issued in the study period
        np.where(
            (df[termination_date_col] < df["anniversary_date"])
            | (df["eos_date"] < df["anniversary_date"]),
            0,
            # inforce
            np.where(
                (df[termination_date_col].isna())
                | (df[termination_date_col] > df["eos_date"]),
                eos_minus_anniversary / total_days_after,
                np.where(
                    # decrement not under study
                    df[termination_reason_col] != study_decrement,
                    termination_minus_anniversary / total_days_after,
                    # else, error
                    -1,
                ),
            ),
        ),
    )

    # check for errors
    num_negative_exposure = (df["exposure_before"] < 0).sum() + (
        df["exposure_after"] < 0
    ).sum()
    if num_negative_exposure > 0:
        logger.error(
            f"Number of rows with negative exposure: {num_negative_exposure}, "
            f"this should not happen."
        )

    return df


def _dist_exposure(
    df: pd.DataFrame,
    study_decrement: str,
    calendar_exposure: bool = True,
    mapping: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate distributed exposure.

    The distributed exposure method for a policy that decrements under the study splits
    the exposure across the calendar year.

    Distributed exposure method aligns with Uniform Distribution of Deaths,
    which means death rates increase through the year.

    The rate calculated using the distributed exposure method will be qx (initial rate)

    Expects the DataFrame to already have these columns:
    - termination_date, termination_reason
    - anniversary_date, next_anniversary_date, prior_anniversary_date
    - bos_date, eos_date

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    study_decrement : str
        Decrement under study, for example "death". This is used to determine the
        exposure for policies that terminate during the study period.
    calendar_exposure : bool, optional default=True
        Whether to calculate exposure by calendar year or by policy year.
        This has a very minor impact on the exposure calculation for policies,
        however when there is a leap year, the exposure will either be 1 for a
        calendar or 1 for a policy. It can't be 1 for both.
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names.

    Returns
    -------
    df : pd.DataFrame
        df with exposure_before and exposure_after columns added.

    """
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    issue_date_col = "issue_date"

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        issue_date_col = mapping.get("issue_date", issue_date_col)

    # series for calculations
    anniversary_minus_bos = (df["anniversary_date"] - df["bos_date"]).dt.days
    termination_minus_bos = (df[termination_date_col] - df["bos_date"]).dt.days + 1
    eos_minus_anniversary = (df["eos_date"] - df["anniversary_date"]).dt.days + 1
    eos_minus_bos = (df["eos_date"] - df["bos_date"]).dt.days + 1
    termination_minus_anniversary = (
        df[termination_date_col] - df["anniversary_date"]
    ).dt.days + 1
    calendar_before = (
        366
        if pd.Timestamp(year=df["eos_date"].iloc[0].year, month=12, day=31).is_leap_year
        else 365
    )
    calendar_after = calendar_before
    policy_before = (df["anniversary_date"] - df["prior_anniversary_date"]).dt.days
    policy_after = (df["next_anniversary_date"] - df["anniversary_date"]).dt.days
    if calendar_exposure:
        total_days_before = calendar_before
        total_days_after = calendar_after
    else:
        total_days_before = policy_before
        total_days_after = policy_after

    # calculate exposures
    df["exposure_before"] = np.where(
        # decrement under study - distributed exposure
        (df[termination_reason_col] == study_decrement)
        & (df[termination_date_col] >= df["prior_anniversary_date"])
        & (df[issue_date_col].dt.year != df["eos_date"].dt.year)
        & (df["anniversary_date"] > df["bos_date"]),
        anniversary_minus_bos / total_days_before,
        # not in the study period or issued in the study period
        np.where(
            (df[termination_date_col] < df["bos_date"])
            | (df["anniversary_date"] <= df["bos_date"])
            | (df[issue_date_col].dt.year == df["eos_date"].dt.year),
            0,
            # inforce
            np.where(
                (df[termination_date_col].isna())
                | (df[termination_date_col] > df["anniversary_date"])
                | (df[termination_date_col] > df["eos_date"]),
                np.minimum(anniversary_minus_bos, eos_minus_bos) / total_days_before,
                np.where(
                    # decrement not under study
                    (df[termination_reason_col] != study_decrement)
                    & (df[termination_date_col] <= df["anniversary_date"]),
                    termination_minus_bos / total_days_before,
                    # else, error
                    -1,
                ),
            ),
        ),
    )

    df["exposure_after"] = np.where(
        # decrement under study - distributed exposure
        (df[termination_reason_col] == study_decrement)
        & (df[termination_date_col] >= df["anniversary_date"])
        & (df[termination_date_col] <= df["eos_date"]),
        eos_minus_anniversary / total_days_after,
        # not in the study period or issued in the study period
        np.where(
            (df[termination_date_col] < df["anniversary_date"])
            | (df["eos_date"] < df["anniversary_date"]),
            0,
            # inforce
            np.where(
                (df[termination_date_col].isna())
                | (df[termination_date_col] > df["eos_date"]),
                eos_minus_anniversary / total_days_after,
                np.where(
                    # decrement not under study
                    df[termination_reason_col] != study_decrement,
                    termination_minus_anniversary / total_days_after,
                    # else, error
                    -1,
                ),
            ),
        ),
    )

    # check for errors
    num_negative_exposure = (df["exposure_before"] < 0).sum() + (
        df["exposure_after"] < 0
    ).sum()
    if num_negative_exposure > 0:
        logger.error(
            f"Number of rows with negative exposure: {num_negative_exposure}, "
            f"this should not happen."
        )

    return df


def _exact_exposure(
    df: pd.DataFrame,
    study_decrement: str,
    calendar_exposure: bool = True,
    mapping: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate exact exposure.

    The exact exposure method for a policy that decrements under the study provides
    exposure up to the exact date of decrement.

    Exact exposure method aligns with Constant Force of Mortality,
    which means death rates are constant through the year.

    The rate calculated using the exact exposure method will be ux (central rate)

    Expects the DataFrame to already have these columns:
    - termination_date, termination_reason
    - anniversary_date, next_anniversary_date, prior_anniversary_date
    - bos_date, eos_date

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    study_decrement : str
        Decrement under study, for example "death". This is used to determine the
        exposure for policies that terminate during the study period.
    calendar_exposure : bool, optional default=True
        Whether to calculate exposure by calendar year or by policy year.
        This has a very minor impact on the exposure calculation for policies,
        however when there is a leap year, the exposure will either be 1 for a
        calendar or 1 for a policy. It can't be 1 for both.
    mapping : dict, optional default=None
        Mapping for the column names if they differ from the expected column names.

    Returns
    -------
    df : pd.DataFrame
        df with exposure_before and exposure_after columns added.

    """
    # default column names
    termination_date_col = "termination_date"
    termination_reason_col = "termination_reason"
    issue_date_col = "issue_date"

    # handle mapping
    if mapping:
        termination_date_col = mapping.get("termination_date", termination_date_col)
        termination_reason_col = mapping.get(
            "termination_reason", termination_reason_col
        )
        issue_date_col = mapping.get("issue_date", issue_date_col)

    # series for calculations
    anniversary_minus_bos = (df["anniversary_date"] - df["bos_date"]).dt.days
    termination_minus_bos = (df[termination_date_col] - df["bos_date"]).dt.days + 1
    eos_minus_anniversary = (df["eos_date"] - df["anniversary_date"]).dt.days + 1
    eos_minus_bos = (df["eos_date"] - df["bos_date"]).dt.days + 1
    termination_minus_anniversary = (
        df[termination_date_col] - df["anniversary_date"]
    ).dt.days + 1
    calendar_before = (
        366
        if pd.Timestamp(year=df["eos_date"].iloc[0].year, month=12, day=31).is_leap_year
        else 365
    )
    calendar_after = calendar_before
    policy_before = (df["anniversary_date"] - df["prior_anniversary_date"]).dt.days
    policy_after = (df["next_anniversary_date"] - df["anniversary_date"]).dt.days
    if calendar_exposure:
        total_days_before = calendar_before
        total_days_after = calendar_after
    else:
        total_days_before = policy_before
        total_days_after = policy_after

    # calculate exposures
    df["exposure_before"] = np.where(
        # decrement under study - exact exposure
        (df[termination_reason_col] == study_decrement)
        & (df[termination_date_col] >= df["bos_date"])
        & (df[issue_date_col].dt.year != df["eos_date"].dt.year)
        & (df["anniversary_date"] > df["bos_date"]),
        np.minimum(termination_minus_bos, anniversary_minus_bos) / total_days_before,
        # not in the study period or issued in the study period
        np.where(
            (df[termination_date_col] < df["bos_date"])
            | (df["anniversary_date"] <= df["bos_date"])
            | (df[issue_date_col].dt.year == df["eos_date"].dt.year),
            0,
            # inforce
            np.where(
                (df[termination_date_col].isna())
                | (df[termination_date_col] > df["anniversary_date"])
                | (df[termination_date_col] > df["eos_date"]),
                np.minimum(anniversary_minus_bos, eos_minus_bos) / total_days_before,
                np.where(
                    # decrement not under study
                    (df[termination_reason_col] != study_decrement)
                    & (df[termination_date_col] <= df["anniversary_date"]),
                    termination_minus_bos / total_days_before,
                    # else, error
                    -1,
                ),
            ),
        ),
    )

    df["exposure_after"] = np.where(
        # decrement under study - exact exposure
        (df[termination_reason_col] == study_decrement)
        & (df[termination_date_col] >= df["anniversary_date"])
        & (df[termination_date_col] <= df["eos_date"]),
        termination_minus_anniversary / total_days_after,
        # not in the study period or issued in the study period
        np.where(
            (df[termination_date_col] < df["anniversary_date"])
            | (df["eos_date"] < df["anniversary_date"]),
            0,
            # inforce
            np.where(
                (df[termination_date_col].isna())
                | (df[termination_date_col] > df["eos_date"]),
                eos_minus_anniversary / total_days_after,
                np.where(
                    # decrement not under study
                    df[termination_reason_col] != study_decrement,
                    termination_minus_anniversary / total_days_after,
                    # else, error
                    -1,
                ),
            ),
        ),
    )

    # check for errors
    num_negative_exposure = (df["exposure_before"] < 0).sum() + (
        df["exposure_after"] < 0
    ).sum()
    if num_negative_exposure > 0:
        logger.error(
            f"Number of rows with negative exposure: {num_negative_exposure}, "
            f"this should not happen."
        )

    return df


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
