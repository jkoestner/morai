"""Tests the experience."""

import pandas as pd
import polars as pl

from morai.experience import experience
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
experience_df = pd.read_csv(test_experience_path / "simple_normalization.csv")
normalization_2_df = pd.read_csv(test_experience_path / "simple_normalization_2.csv")


def test_relative_risk_aggregate() -> None:
    """Tests the relative risk calculation."""
    test_df = experience.calc_relative_risk(
        df=experience_df, features=["year"], risk_col=["year_lob_rate"]
    )
    total_mean = test_df["year_lob_rate"].mean()
    year_mean = test_df[test_df["year"] == 2019]["year_lob_rate"].mean()
    year_risk = round(year_mean / total_mean, 3)

    # polars
    experience_lf = pl.from_pandas(experience_df).lazy()
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
        df=normalization_2_df,
        features=["sex"],
        risk_col=["rate"],
        weight_col=["exposures"],
    )

    # polars
    experience_lf = pl.from_pandas(normalization_2_df).lazy()
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
        df=experience_df,
        features=["year"],
        risk_col=["year_lob_multi_rate"],
        relative_to="reference",
    )
    reference_mean = test_df[test_df["year"] == 2019]["year_lob_multi_rate"].mean()
    year_mean = test_df[test_df["year"] == 2020]["year_lob_multi_rate"].mean()
    year_risk = round(year_mean / reference_mean, 3)

    # polars
    experience_lf = pl.from_pandas(experience_df).lazy()
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
        df=experience_df,
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
    experience_lf = pl.from_pandas(experience_df).lazy()
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
        df=experience_df,
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
    experience_lf = pl.from_pandas(experience_df).lazy()
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
        df=experience_df,
        features=["year"],
        normalize_col=["year_rate"],
        add_norm_col=True,
    )

    # polars
    experience_lf = pl.from_pandas(experience_df).lazy()
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
        df=normalization_2_df,
        features=["sex"],
        normalize_col=["rate"],
        weight_col=["exposures"],
        add_norm_col=True,
    )

    # polars
    experience_lf = pl.from_pandas(normalization_2_df).lazy()
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
        df=normalization_2_df,
        features=["sex"],
        normalize_col=["deaths"],
        weight_col=["exposures"],
        add_norm_col=True,
        ratio=True,
    )

    # polars
    experience_lf = pl.from_pandas(normalization_2_df).lazy()
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
