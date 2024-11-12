"""Tests the experience."""

import pandas as pd

from morai.experience import experience
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
experience_df = pd.read_csv(test_experience_path / "simple_normalization.csv")
experience_df = pd.read_csv(test_experience_path / "simple_graduation_1d.csv")
experience_df = pd.read_csv(test_experience_path / "simple_normalization.csv")


def test_relative_risk() -> None:
    """Tests the relative risk calculation."""
    test_df = experience.calc_relative_risk(
        df=experience_df, features=["year"], numerator="year_lob_rate"
    )
    total_mean = test_df["year_lob_rate"].mean()
    year_mean = test_df[test_df["year"] == 2019]["year_lob_rate"].mean()
    year_risk = round(year_mean / total_mean, 3)

    assert round(test_df[test_df["year"] == 2019]["risk"].iloc[0], 3) == year_risk, (
        "Expected relative risk to be average rate for feature group divided "
        "by the average rate for all groups"
    )


def test_normalize() -> None:
    """Tests the normalization calculation."""
    test_df = experience.normalize(
        df=experience_df, features=["year"], numerator="year_lob_rate"
    )
    test_df = experience.calc_relative_risk(
        df=test_df, features=["year"], numerator="year_lob_rate_norm"
    )
    assert (
        round(test_df[test_df["year"] == 2019]["risk"].iloc[0], 3) == 1.0
    ), "The relative risk should be 1.0 for the after normalizing column"


def test_whl_1d() -> None:
    """Tests the 1D WHL calculation."""
    whl_1d_df = pd.read_csv(test_experience_path / "simple_graduation_1d.csv")
    grad_rates = experience.calc_whl(
        rates=whl_1d_df["rate"],
        weights=whl_1d_df["exposure"],
        horizontal_order=3,
        horizontal_lambda=400,
        horizontal_expo=0,
        vertical_order=0,
        vertical_lambda=0,
        vertical_expo=0,
        normalize_weights=True,
    )
    checksum_grad_rates = grad_rates.sum()
    assert round(checksum_grad_rates, 4) == 0.4337, "The 1d WHL checksum is incorrect"


def test_whl_1d_age() -> None:
    """Tests the 1D WHL calculation with standard table."""
    whl_1d_age_df = pd.read_csv(test_experience_path / "simple_graduation_1d_age.csv")
    grad_rates = experience.calc_whl(
        rates=whl_1d_age_df["rate"],
        weights=whl_1d_age_df["exposure"],
        horizontal_order=3,
        horizontal_lambda=400,
        horizontal_expo=0,
        vertical_order=0,
        vertical_lambda=0,
        vertical_expo=0,
        normalize_weights=True,
        standard_rates=whl_1d_age_df["std"],
        standard_weights=whl_1d_age_df["std_weight"],
        blending_factor=0.5,
    )
    checksum_grad_rates = grad_rates.sum()
    assert (
        round(checksum_grad_rates, 4) == 11.0632
    ), "The 1d WHL with standard table checksum is incorrect"


def test_whl_2d() -> None:
    """Tests the 2D WHL calculation."""
    rates = pd.read_excel(
        test_experience_path / "simple_graduation_2d.xlsx", "rates", index_col=0
    )
    weights = pd.read_excel(
        test_experience_path / "simple_graduation_2d.xlsx", "weights", index_col=0
    )
    grad_rates = experience.calc_whl(
        rates=rates,
        weights=weights,
        horizontal_order=2,
        horizontal_lambda=300,
        horizontal_expo=0,
        vertical_order=3,
        vertical_lambda=100,
        vertical_expo=0,
        normalize_weights=True,
    )
    checksum_grad_rates = grad_rates.sum()
    assert round(checksum_grad_rates, 4) == 5.6160, "The 2d WHL checksum is incorrect"
