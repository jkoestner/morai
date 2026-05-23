"""Tests the experience."""

import pandas as pd
import polars as pl

from morai.experience import experience
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
complex_norm_df = pd.read_csv(test_experience_path / "complex_normalization.csv")
normalization_df = pd.read_csv(test_experience_path / "simple_normalization.csv")
experience_df = pd.read_csv(
    test_experience_path / "sample_experience_data_specific.csv"
)


def test_relative_risk_aggregate() -> None:
    """Tests the relative risk calculation."""
    test_df = experience.calc_relative_risk(
        df=complex_norm_df, features=["year"], risk_col=["year_lob_rate"]
    )
    total_mean = test_df["year_lob_rate"].mean()
    year_mean = test_df[test_df["year"] == 2019]["year_lob_rate"].mean()
    year_risk = round(year_mean / total_mean, 3)

    # polars
    experience_lf = pl.from_pandas(complex_norm_df).lazy()
    test_lf = experience.calc_relative_risk(
        df=experience_lf, features=["year"], risk_col=["year_lob_rate"]
    )
    test_lf = test_lf.collect().to_pandas()

    assert (
        round(test_df[test_df["year"] == 2019]["relative_risk"].iloc[0], 3) == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for all groups"
    )
    assert (
        round(test_lf[test_lf["year"] == 2019]["relative_risk"].iloc[0], 3) == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for all groups"
    )


def test_relative_risk_weighted() -> None:
    """Tests the relative risk calculation with weights."""
    test_df = experience.calc_relative_risk(
        df=normalization_df,
        features=["sex"],
        risk_col=["rate"],
        weight_col=["exposures"],
    )

    # polars
    experience_lf = pl.from_pandas(normalization_df).lazy()
    test_lf = experience.calc_relative_risk(
        df=experience_lf,
        features=["sex"],
        risk_col=["rate"],
        weight_col=["exposures"],
    )
    test_lf = test_lf.collect().to_pandas()

    assert round(test_df[test_df["sex"] == "M"]["relative_risk"].iloc[0], 3) == 0.75, (
        "Expected relative risk to be weighted average rate for feature group divided "
        "by the weighted average rate for all groups"
    )
    assert round(test_lf[test_lf["sex"] == "M"]["relative_risk"].iloc[0], 3) == 0.75, (
        "Expected relative risk to be weighted average rate for feature group divided "
        "by the weighted average rate for all groups"
    )


def test_relative_risk_reference() -> None:
    """Tests the relative risk calculation with reference."""
    test_df = experience.calc_relative_risk(
        df=complex_norm_df,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="reference",
    )
    reference_mean = test_df[test_df["year"] == 2019]["year_lob_multi_rate"].mean()
    year_mean = test_df[test_df["year"] == 2020]["year_lob_multi_rate"].mean()
    year_risk = round(year_mean / reference_mean, 3)

    # polars
    experience_lf = pl.from_pandas(complex_norm_df).lazy()
    test_lf = experience.calc_relative_risk(
        df=experience_lf,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="reference",
    )
    test_lf = test_lf.collect().to_pandas()

    assert (
        round(test_df[test_df["year"] == 2020]["relative_risk"].iloc[0], 3) == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for reference"
    )
    assert (
        round(test_lf[test_lf["year"] == 2020]["relative_risk"].iloc[0], 3) == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for reference"
    )


def test_relative_risk_reference_by() -> None:
    """Tests the relative risk calculation with reference."""
    test_df = experience.calc_relative_risk(
        df=complex_norm_df,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="reference",
        relative_cols=["lob"],
    )
    reference_mean = test_df[(test_df["year"] == 2019) & (test_df["lob"] == "UL")][
        "year_lob_multi_rate"
    ].mean()
    year_mean = test_df[(test_df["year"] == 2020) & (test_df["lob"] == "UL")][
        "year_lob_multi_rate"
    ].mean()
    year_risk = round(year_mean / reference_mean, 3)

    # polars
    experience_lf = pl.from_pandas(complex_norm_df).lazy()
    test_lf = experience.calc_relative_risk(
        df=experience_lf,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="reference",
        relative_cols=["lob"],
    )
    test_lf = test_lf.collect().to_pandas()

    assert (
        round(
            test_df[(test_df["year"] == 2020) & (test_df["lob"] == "UL")][
                "relative_risk"
            ].iloc[0],
            3,
        )
        == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for reference that is in the same relative group"
    )
    assert (
        round(
            test_lf[(test_df["year"] == 2020) & (test_df["lob"] == "UL")][
                "relative_risk"
            ].iloc[0],
            3,
        )
        == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for reference that is in the same relative group"
    )


def test_relative_risk_subset() -> None:
    """Tests the relative risk calculation with subset."""
    test_df = experience.calc_relative_risk(
        df=complex_norm_df,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="subset",
        relative_cols=["lob"],
        subset_dict={"year": [2019, 2020]},
    )
    reference_mean = test_df[
        (test_df["year"].isin([2019, 2020])) & (test_df["lob"] == "UL")
    ]["year_lob_multi_rate"].mean()
    year_mean = test_df[(test_df["year"] == 2020) & (test_df["lob"] == "UL")][
        "year_lob_multi_rate"
    ].mean()
    year_risk = round(year_mean / reference_mean, 3)

    # polars
    experience_lf = pl.from_pandas(complex_norm_df).lazy()
    test_lf = experience.calc_relative_risk(
        df=experience_lf,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="subset",
        relative_cols=["lob"],
        subset_dict={"year": [2019, 2020]},
    )
    test_lf = test_lf.collect().to_pandas()

    assert (
        round(
            test_df[(test_df["year"] == 2020) & (test_df["lob"] == "UL")][
                "relative_risk"
            ].iloc[0],
            3,
        )
        == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for subset"
    )
    assert (
        round(
            test_lf[(test_df["year"] == 2020) & (test_df["lob"] == "UL")][
                "relative_risk"
            ].iloc[0],
            3,
        )
        == year_risk
    ), (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for subset"
    )


def test_normalize() -> None:
    """Tests the normalization calculation."""
    test_df = experience.normalize(
        df=complex_norm_df,
        features=["year"],
        normalize_col=["year_rate"],
        add_norm_col=True,
    )

    # polars
    experience_lf = pl.from_pandas(complex_norm_df).lazy()
    test_lf = experience.normalize(
        df=experience_lf,
        features=["year"],
        normalize_col=["year_rate"],
        add_norm_col=True,
    )
    test_lf = test_lf.collect().to_pandas()

    assert (
        round(test_df[test_df["year"] == 2019]["year_rate_norm"].iloc[0], 3) == 1.075
    ), "The normalized rate should be 1.075 after normalizing column"
    assert (
        round(test_lf[test_lf["year"] == 2019]["year_rate_norm"].iloc[0], 3) == 1.075
    ), "The normalized rate should be 1.075 after normalizing column"


def test_normalize_weighted() -> None:
    """Tests the normalization calculation with weights."""
    test_df = experience.normalize(
        df=normalization_df,
        features=["sex"],
        normalize_col=["rate"],
        weight_col=["exposures"],
        add_norm_col=True,
    )

    # polars
    experience_lf = pl.from_pandas(normalization_df).lazy()
    test_lf = experience.normalize(
        df=experience_lf,
        features=["sex"],
        normalize_col=["rate"],
        weight_col=["exposures"],
        add_norm_col=True,
    )
    test_lf = test_lf.collect().to_pandas()

    assert round(test_df[test_df["sex"] == "M"]["rate_norm"].iloc[0], 3) == 0.133, (
        "The normalized rate should be 0.133 after normalizing column"
    )
    assert round(test_lf[test_lf["sex"] == "M"]["rate_norm"].iloc[0], 3) == 0.133, (
        "The normalized rate should be 0.133 after normalizing column"
    )


def test_normalize_ratio() -> None:
    """Tests the normalization calculation with ratio option."""
    test_df = experience.normalize(
        df=normalization_df,
        features=["sex"],
        normalize_col=["deaths"],
        weight_col=["exposures"],
        add_norm_col=True,
        ratio=True,
    )

    # polars
    experience_lf = pl.from_pandas(normalization_df).lazy()
    test_lf = experience.normalize(
        df=experience_lf,
        features=["sex"],
        normalize_col=["deaths"],
        weight_col=["exposures"],
        add_norm_col=True,
        ratio=True,
    )
    test_lf = test_lf.collect().to_pandas()

    assert round(test_df[test_df["sex"] == "M"]["deaths_norm"].iloc[0], 3) == 133.333, (
        "The normalized rate should be 133.333 after normalizing column"
    )
    assert round(test_lf[test_lf["sex"] == "M"]["deaths_norm"].iloc[0], 3) == 133.333, (
        "The normalized rate should be 133.333 after normalizing column"
    )


def test_exposure_policy() -> None:
    """Tests using policy exposure instead of calendar exposure."""
    test_df = experience.create_study(
        df=experience_df,
        bos="1/1/2023",
        eos="12/31/2023",
        study_frequency="annually",
        study_decrement="D",
        exposure_method="annual",
        calendar_exposure=False,
        get_exposures=True,
        get_actuals=False,
    )

    # 2024 is a leap year and uses 366 days and the policy duration crosses.
    # The after exposure on a policy basis will go from 10/23/2023 to 10/22/2024,
    # which crosses the leap day.
    assert (
        test_df[
            (test_df["id"] == 1)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 70 / 366
    ), "Expected exposure to be using 366 days for policy year as in leap year."


def test_exposure_annual_calendar() -> None:
    """Tests the annual exposure calculation."""
    test_df = experience.create_study(
        df=experience_df,
        bos="1/1/2022",
        eos="3/31/2024",
        study_frequency="annually",
        study_decrement="D",
        exposure_method="annual",
        calendar_exposure=True,
        get_exposures=True,
        get_actuals=False,
    )
    test_df = test_df[test_df["exposure_cnt"] != 0]

    # not in study period
    # issued after study period
    assert test_df[(test_df["id"] == 1) & (test_df["bos_date"] == "1/1/2022")].empty, (
        "Expected no rows for policy not in study period"
    )
    # decrement under study - terminated before study period
    assert test_df[(test_df["id"] == 2) & (test_df["bos_date"] == "1/1/2023")].empty, (
        "Expected no rows for policy not in study period"
    )
    # decrement not under study - terminated before study period
    assert test_df[(test_df["id"] == 3) & (test_df["bos_date"] == "1/1/2023")].empty, (
        "Expected no rows for policy not in study period"
    )

    # inforce
    # active
    assert (
        test_df[
            (test_df["id"] == 4)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be 365 for inforce policy in total"
    assert (
        test_df[
            (test_df["id"] == 4)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 5)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), "Expected exposure to be 365 for inforce policy in total"
    # issued during study period
    assert test_df[
        (test_df["id"] == 5)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 5)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 70 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # new issue - death
    assert test_df[
        (test_df["id"] == 6)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 6)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # new issue - lapse
    assert test_df[
        (test_df["id"] == 7)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 7)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # decrement not under study
    # before anniversary
    assert (
        test_df[
            (test_df["id"] == 8)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 3)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert test_df[
        (test_df["id"] == 8)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 4)
    ].empty, (
        "Expected exposure to be up until termination for decrement not under study, "
        "since termination is before anniversary there is no exposure after anniversary"
    )
    # on anniversary
    assert (
        test_df[
            (test_df["id"] == 9)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 10)
        ]["exposure_cnt"].iloc[0]
        == 99 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert (
        test_df[
            (test_df["id"] == 9)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 1 / 365
    ), (
        "Expected exposure to be up until termination for decrement not under study, "
        "since termination is on anniversary there is 1 day exposure after anniversary"
    )
    # after anniversary
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 7)
        ]["exposure_cnt"].iloc[0]
        == 197 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 8)
        ]["exposure_cnt"].iloc[0]
        == 100 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    # in the future
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 6)
        ]["exposure_cnt"].iloc[0]
        == 197 / 365
    ), "Expected exposure to be same as inforce policy"
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 7)
        ]["exposure_cnt"].iloc[0]
        == 168 / 365
    ), "Expected exposure to be same as inforce policy"

    # decrement under study
    # before anniversary
    assert (
        test_df[
            (test_df["id"] == 11)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 9)
        ]["exposure_cnt"].iloc[0]
        == 130 / 365
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert test_df[
        (test_df["id"] == 11)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 10)
    ].empty, (
        "Expected exposure to be up until anniversary for decrement under study, "
        "since termination is before anniversary there is no exposure after anniversary"
    )
    # on anniversary
    assert (
        test_df[
            (test_df["id"] == 12)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 10)
        ]["exposure_cnt"].iloc[0]
        == 65 / 365
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert (
        test_df[
            (test_df["id"] == 12)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 365 / 365
    ), (
        "Expected exposure to be up until anniversary for decrement under study, "
        "since termination is on anniversary there is a full year of exposure after "
        "anniversary"
    )
    # after anniversary
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be up until termination for decrement under study"
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 5)
        ]["exposure_cnt"].iloc[0]
        == 365 / 365
    ), (
        "Expected exposure to be up until termination for decrement under study, "
        "since termination is after anniversary there is a full year of exposure after "
        "anniversary"
    )
    # in the future
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 3)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be same as inforce policy"
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), "Expected exposure to be same as inforce policy"

    # partial year
    # decrement not under study
    assert (
        test_df[
            (test_df["id"] == 14)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 66 / 366
    ), (
        "Expected exposure to be up until min(termination, anniversary, eos) for "
        "decrement not under study"
    )
    assert (
        test_df[
            (test_df["id"] == 14)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 12)
        ]["exposure_cnt"].iloc[0]
        == 25 / 366
    ), (
        "Expected exposure to be up until min(termination, eos) for "
        "decrement not under study"
    )
    # decrement under study
    assert (
        test_df[
            (test_df["id"] == 15)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 331 / 366
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert test_df[
        (test_df["id"] == 15)
        & (test_df["bos_date"] == "1/1/2024")
        & (test_df["policy_dur"] == 5)
    ].empty, (
        "Expected exposure to be 0 for policy that decremented before the anniversary"
    )

    # frequency exposure check
    frequency_df = experience.create_study(
        df=experience_df,
        bos="1/1/2023",
        eos="12/31/2023",
        study_frequency="monthly",
        study_decrement="D",
        exposure_method="annual",
        calendar_exposure=True,
        get_exposures=True,
        get_actuals=False,
    )

    assert frequency_df[frequency_df["exposure_cnt"] < 0].empty, (
        "There should be no negative exposures"
    )

    assert (
        frequency_df[
            (frequency_df["id"] == 12)
            & (frequency_df["bos_date"] == "3/1/2023")
            & (frequency_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 365 / 365
    ), "Expected exposure to be from bos to anniversary"

    assert (
        frequency_df[
            (frequency_df["id"] == 12)
            & (frequency_df["bos_date"] == "4/1/2023")
            & (frequency_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 0 / 365
    ), "Expected exposure to be from bos to anniversary"

    assert (
        frequency_df[
            (frequency_df["id"] == 11)
            & (frequency_df["bos_date"] == "1/1/2023")
            & (frequency_df["policy_dur"] == 9)
        ]["exposure_cnt"].iloc[0]
        == 31 / 365
    ), "Expected exposure to be from bos to eos"


def test_exposure_distributed_calendar() -> None:
    """Tests the distributed exposure calculation."""
    test_df = experience.create_study(
        df=experience_df,
        bos="1/1/2022",
        eos="3/31/2024",
        study_frequency="annually",
        study_decrement="D",
        exposure_method="distributed",
        calendar_exposure=True,
        get_exposures=True,
        get_actuals=False,
    )
    test_df = test_df[test_df["exposure_cnt"] != 0]

    # not in study period
    # issued after study period
    assert test_df[(test_df["id"] == 1) & (test_df["bos_date"] == "1/1/2022")].empty, (
        "Expected no rows for policy not in study period"
    )
    # decrement not under study - terminated before study period
    assert test_df[(test_df["id"] == 3) & (test_df["bos_date"] == "1/1/2023")].empty, (
        "Expected no rows for policy not in study period"
    )

    # inforce
    # active
    assert (
        test_df[
            (test_df["id"] == 4)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be 365 for inforce policy in total"
    assert (
        test_df[
            (test_df["id"] == 4)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 5)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), "Expected exposure to be 365 for inforce policy in total"
    # issued during study period
    assert test_df[
        (test_df["id"] == 5)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 5)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 70 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # new issue - death
    assert test_df[
        (test_df["id"] == 6)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 6)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # new issue - lapse
    assert test_df[
        (test_df["id"] == 7)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 7)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # decrement not under study
    # before anniversary
    assert (
        test_df[
            (test_df["id"] == 8)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 3)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert test_df[
        (test_df["id"] == 8)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 4)
    ].empty, (
        "Expected exposure to be up until termination for decrement not under study, "
        "since termination is before anniversary there is no exposure after anniversary"
    )
    # on anniversary
    assert (
        test_df[
            (test_df["id"] == 9)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 10)
        ]["exposure_cnt"].iloc[0]
        == 99 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert (
        test_df[
            (test_df["id"] == 9)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 1 / 365
    ), (
        "Expected exposure to be up until termination for decrement not under study, "
        "since termination is on anniversary there is 1 day exposure after anniversary"
    )
    # after anniversary
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 7)
        ]["exposure_cnt"].iloc[0]
        == 197 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 8)
        ]["exposure_cnt"].iloc[0]
        == 100 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    # in the future
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 6)
        ]["exposure_cnt"].iloc[0]
        == 197 / 365
    ), "Expected exposure to be same as inforce policy"
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 7)
        ]["exposure_cnt"].iloc[0]
        == 168 / 365
    ), "Expected exposure to be same as inforce policy"

    # decrement under study
    # before anniversary
    assert (
        test_df[
            (test_df["id"] == 11)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 9)
        ]["exposure_cnt"].iloc[0]
        == 130 / 365
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert test_df[
        (test_df["id"] == 11)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 10)
    ].empty, (
        "Expected exposure to be up until anniversary for decrement under study, "
        "since termination is before anniversary there is no exposure after anniversary"
    )
    # on anniversary
    assert (
        test_df[
            (test_df["id"] == 12)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 10)
        ]["exposure_cnt"].iloc[0]
        == 65 / 365
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert (
        test_df[
            (test_df["id"] == 12)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 300 / 365
    ), (
        "Expected exposure to be up until anniversary for decrement under study, "
        "since termination is on anniversary there is a full year of exposure after "
        "anniversary"
    )
    # after anniversary
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be up until termination for decrement under study"
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 5)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until termination for decrement under study, "
        "since termination is after anniversary there is a full year of exposure after "
        "anniversary"
    )
    # in the future
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 3)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be same as inforce policy"
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), "Expected exposure to be same as inforce policy"
    # terminated before study period
    assert (
        test_df[
            (test_df["id"] == 2)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 328 / 365
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert test_df[
        (test_df["id"] == 2)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 5)
    ].empty, (
        "Expected exposure to be up until anniversary for decrement under study, "
        "since termination is before anniversary from the prior year, "
        "there is no exposure after anniversary"
    )

    # partial year
    # decrement not under study
    assert (
        test_df[
            (test_df["id"] == 14)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 66 / 366
    ), (
        "Expected exposure to be up until min(termination, anniversary, eos) for "
        "decrement not under study"
    )
    assert (
        test_df[
            (test_df["id"] == 14)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 12)
        ]["exposure_cnt"].iloc[0]
        == 25 / 366
    ), (
        "Expected exposure to be up until min(termination, eos) for "
        "decrement not under study"
    )
    # decrement under study
    assert (
        test_df[
            (test_df["id"] == 15)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 91 / 366
    ), "Expected exposure to be up until anniversary for decrement under study"
    assert test_df[
        (test_df["id"] == 15)
        & (test_df["bos_date"] == "1/1/2024")
        & (test_df["policy_dur"] == 5)
    ].empty, (
        "Expected exposure to be 0 for policy that decremented before the anniversary"
    )

    # frequency exposure check
    frequency_df = experience.create_study(
        df=experience_df,
        bos="1/1/2023",
        eos="12/31/2023",
        study_frequency="monthly",
        study_decrement="D",
        exposure_method="distributed",
        calendar_exposure=True,
        get_exposures=True,
        get_actuals=False,
    )

    assert frequency_df[frequency_df["exposure_cnt"] < 0].empty, (
        "There should be no negative exposures"
    )

    assert (
        frequency_df[
            (frequency_df["id"] == 12)
            & (frequency_df["bos_date"] == "3/1/2023")
            & (frequency_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 25 / 365
    ), "Expected exposure to be from bos to anniversary"

    assert (
        frequency_df[
            (frequency_df["id"] == 12)
            & (frequency_df["bos_date"] == "4/1/2023")
            & (frequency_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 30 / 365
    ), "Expected exposure to be from bos to anniversary"

    assert (
        frequency_df[
            (frequency_df["id"] == 11)
            & (frequency_df["bos_date"] == "1/1/2023")
            & (frequency_df["policy_dur"] == 9)
        ]["exposure_cnt"].iloc[0]
        == 31 / 365
    ), "Expected exposure to be from bos to eos"


def test_exposure_exact_calendar() -> None:
    """Tests the exact exposure calculation."""
    test_df = experience.create_study(
        df=experience_df,
        bos="1/1/2022",
        eos="3/31/2024",
        study_frequency="annually",
        study_decrement="D",
        exposure_method="exact",
        calendar_exposure=True,
        get_exposures=True,
        get_actuals=False,
    )
    test_df = test_df[test_df["exposure_cnt"] != 0]

    # not in study period
    # issued after study period
    assert test_df[(test_df["id"] == 1) & (test_df["bos_date"] == "1/1/2022")].empty, (
        "Expected no rows for policy not in study period"
    )
    # decrement under study - terminated before study period
    assert test_df[(test_df["id"] == 2) & (test_df["bos_date"] == "1/1/2023")].empty, (
        "Expected no rows for policy not in study period"
    )
    # decrement not under study - terminated before study period
    assert test_df[(test_df["id"] == 3) & (test_df["bos_date"] == "1/1/2023")].empty, (
        "Expected no rows for policy not in study period"
    )

    # inforce
    # active
    assert (
        test_df[
            (test_df["id"] == 4)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be 365 for inforce policy in total"
    assert (
        test_df[
            (test_df["id"] == 4)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 5)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), "Expected exposure to be 365 for inforce policy in total"
    # issued during study period
    assert test_df[
        (test_df["id"] == 5)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 5)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 70 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # new issue - death
    assert test_df[
        (test_df["id"] == 6)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 6)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # new issue - lapse
    assert test_df[
        (test_df["id"] == 7)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 0)
    ].empty, "Expected exposure to be 0 for inforce policy issued during study period"
    assert (
        test_df[
            (test_df["id"] == 7)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 1)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), (
        "Expected exposure to be up until eos for inforce policy issued during "
        "study period"
    )
    # decrement not under study
    # before anniversary
    assert (
        test_df[
            (test_df["id"] == 8)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 3)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert test_df[
        (test_df["id"] == 8)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 4)
    ].empty, (
        "Expected exposure to be up until termination for decrement not under study, "
        "since termination is before anniversary there is no exposure after anniversary"
    )
    # on anniversary
    assert (
        test_df[
            (test_df["id"] == 9)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 10)
        ]["exposure_cnt"].iloc[0]
        == 99 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert (
        test_df[
            (test_df["id"] == 9)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 1 / 365
    ), (
        "Expected exposure to be up until termination for decrement not under study, "
        "since termination is on anniversary there is 1 day exposure after anniversary"
    )
    # after anniversary
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 7)
        ]["exposure_cnt"].iloc[0]
        == 197 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 8)
        ]["exposure_cnt"].iloc[0]
        == 100 / 365
    ), "Expected exposure to be up until termination for decrement not under study"
    # in the future
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 6)
        ]["exposure_cnt"].iloc[0]
        == 197 / 365
    ), "Expected exposure to be same as inforce policy"
    assert (
        test_df[
            (test_df["id"] == 10)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 7)
        ]["exposure_cnt"].iloc[0]
        == 168 / 365
    ), "Expected exposure to be same as inforce policy"

    # decrement under study
    # before anniversary
    assert (
        test_df[
            (test_df["id"] == 11)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 9)
        ]["exposure_cnt"].iloc[0]
        == 33 / 365
    ), "Expected exposure to be up until date of decrement for decrement under study"
    assert test_df[
        (test_df["id"] == 11)
        & (test_df["bos_date"] == "1/1/2023")
        & (test_df["policy_dur"] == 10)
    ].empty, (
        "Expected exposure to be up until date of decrement for decrement under study, "
        "since termination is before anniversary there is no exposure after anniversary"
    )
    # on anniversary
    assert (
        test_df[
            (test_df["id"] == 12)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 10)
        ]["exposure_cnt"].iloc[0]
        == 65 / 365
    ), "Expected exposure to be up until date of decrement for decrement under study"
    assert (
        test_df[
            (test_df["id"] == 12)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 1 / 365
    ), (
        "Expected exposure to be up until date of decrement for decrement under study, "
        "since termination is on date of decrement there is 1 day of "
        "exposure after date of decrement"
    )
    # after anniversary
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be up until termination for decrement under study"
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2023")
            & (test_df["policy_dur"] == 5)
        ]["exposure_cnt"].iloc[0]
        == 33 / 365
    ), "Expected exposure to be up until termination for decrement under study."
    # in the future
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 3)
        ]["exposure_cnt"].iloc[0]
        == 329 / 365
    ), "Expected exposure to be same as inforce policy"
    assert (
        test_df[
            (test_df["id"] == 13)
            & (test_df["bos_date"] == "1/1/2022")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 36 / 365
    ), "Expected exposure to be same as inforce policy"

    # partial year
    # decrement not under study
    assert (
        test_df[
            (test_df["id"] == 14)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 66 / 366
    ), (
        "Expected exposure to be up until min(termination, anniversary, eos) for "
        "decrement not under study"
    )
    assert (
        test_df[
            (test_df["id"] == 14)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 12)
        ]["exposure_cnt"].iloc[0]
        == 25 / 366
    ), (
        "Expected exposure to be up until min(termination, eos) for "
        "decrement not under study"
    )
    # decrement under study
    assert (
        test_df[
            (test_df["id"] == 15)
            & (test_df["bos_date"] == "1/1/2024")
            & (test_df["policy_dur"] == 4)
        ]["exposure_cnt"].iloc[0]
        == 67 / 366
    ), "Expected exposure to be up until decrement for decrement under study"
    assert test_df[
        (test_df["id"] == 15)
        & (test_df["bos_date"] == "1/1/2024")
        & (test_df["policy_dur"] == 5)
    ].empty, (
        "Expected exposure to be 0 for policy that decremented before the anniversary"
    )

    # frequency exposure check
    frequency_df = experience.create_study(
        df=experience_df,
        bos="1/1/2023",
        eos="12/31/2023",
        study_frequency="monthly",
        study_decrement="D",
        exposure_method="exact",
        calendar_exposure=True,
        get_exposures=True,
        get_actuals=False,
    )

    assert frequency_df[frequency_df["exposure_cnt"] < 0].empty, (
        "There should be no negative exposures"
    )

    assert (
        frequency_df[
            (frequency_df["id"] == 12)
            & (frequency_df["bos_date"] == "3/1/2023")
            & (frequency_df["policy_dur"] == 11)
        ]["exposure_cnt"].iloc[0]
        == 1 / 365
    ), "Expected exposure to be from bos to anniversary"

    assert (
        frequency_df[
            (frequency_df["id"] == 11)
            & (frequency_df["bos_date"] == "1/1/2023")
            & (frequency_df["policy_dur"] == 9)
        ]["exposure_cnt"].iloc[0]
        == 31 / 365
    ), "Expected exposure to be from bos to eos"
