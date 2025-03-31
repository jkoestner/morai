"""Tests the experience."""

import pandas as pd

from morai.experience import experience
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
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
        df=experience_df,
        features=["year"],
        numerator="year_lob_rate",
        add_norm_col=True,
    )
    test_df = experience.calc_relative_risk(
        df=test_df, features=["year"], numerator="year_lob_rate_norm"
    )
    assert (
        round(test_df[test_df["year"] == 2019]["risk"].iloc[0], 3) == 1.0
    ), "The relative risk should be 1.0 for the after normalizing column"
