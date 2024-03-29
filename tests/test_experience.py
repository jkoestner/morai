"""Tests the experience."""

import pandas as pd

from morai.experience import experience, tables
from morai.utils import helpers

test_experience_path = helpers.ROOT_PATH / "tests" / "files" / "experience"
experience_df = pd.read_csv(test_experience_path / "simple_normalization.csv")


def test_table_build():
    """Checks the soa table builds."""
    t1683 = pd.read_csv(test_experience_path / "tables" / "t1683.csv", index_col=0)
    t1683_e = pd.read_csv(
        test_experience_path / "tables" / "t1683_extend.csv", index_col=0
    )
    vbt15 = pd.read_csv(test_experience_path / "tables" / "vbt15.csv", index_col=0)
    vbt15_e = pd.read_csv(
        test_experience_path / "tables" / "vbt15_extend.csv", index_col=0
    )
    vbt15_j = pd.read_csv(
        test_experience_path / "tables" / "vbt15_juv.csv", index_col=0
    )
    MortTable = tables.MortTable()

    # ultimate table
    extra_dims = None
    juv_list = None
    table_list = [1683]
    test_1683 = MortTable.build_table(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=False
    )
    test_1683_e = MortTable.build_table(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=True
    )

    # select and ultimate table with multiple dimensions
    extra_dims = {"gender": ["F", "M"], "underwriting": ["NS", "S"]}
    table_list = [3224, 3234, 3252, 3262]
    test_vbt15 = MortTable.build_table(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=False
    )
    test_vbt15_e = MortTable.build_table(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=True
    )
    juv_list = [3273, 3273, 3274, 3274]
    test_vbt15_j = MortTable.build_table(
        table_list=table_list, extra_dims=extra_dims, juv_list=juv_list, extend=False
    )

    assert test_1683.equals(t1683)
    assert test_1683_e.equals(t1683_e)
    assert test_vbt15.equals(vbt15)
    assert test_vbt15_e.equals(vbt15_e)
    assert test_vbt15_j.equals(vbt15_j)


def test_relative_risk():
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


def test_normalize():
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
