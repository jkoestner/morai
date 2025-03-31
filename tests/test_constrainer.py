"""Tests constrainer functions."""

import pandas as pd

from morai.forecast import constraint
from morai.utils import helpers

test_forecast_path = helpers.ROOT_PATH / "tests" / "files" / "forecast" / "constraint"
constraint_df = pd.read_csv(test_forecast_path / "constraint.csv")


def test_run_constraint():
    """
    Checks run constraint function.

    When issue age is constrained, qx should be monotonically increasing.
    """
    fixed_df = constraint.run_constraint(
        constraint_df,
        col_to_fix="wrong_qx",
        constraint_col="issue_age",
        other_feature_cols=["underwriting", "duration", "sex"],
        fixed_col="wrong_qx_fixed",
        min_increase=0.001,
    )

    # issue age - check monotonically increasing
    test_df = fixed_df.sort_values("issue_age", ascending=True)
    test_grouped = test_df.groupby(["underwriting", "sex", "duration"], observed=True)[
        "wrong_qx_fixed"
    ]
    for _, test_group in test_grouped:
        for i in range(len(test_group) - 1):
            assert (
                test_group.iloc[i] <= test_group.iloc[i + 1]
            ), "Expected wrong_qx_fixed to be monotonically increasing"


def test_table_constrainer():
    """Checks table constrainer function."""
    constrainer = constraint.TableConstrainer(
        col_to_fix="wrong_qx",
        issue_age_col="issue_age",
        duration_col="duration",
        attained_age_col="attained_age",
        other_feature_cols=["underwriting", "sex"],
        min_increase=0.001,
        iteration_limit=10,
    )
    fixed_df = constrainer.fix_df(constraint_df)

    # issue age - check monotonically increasing
    test_df = fixed_df.sort_values("issue_age", ascending=True)
    test_grouped = test_df.groupby(["underwriting", "sex", "duration"], observed=True)[
        "wrong_qx_fixed"
    ]
    for _, test_group in test_grouped:
        for i in range(len(test_group) - 1):
            assert (
                test_group.iloc[i] <= test_group.iloc[i + 1]
            ), "Expected wrong_qx_fixed to be monotonically increasing - issue age"

    # duration - check monotonically increasing
    test_df = fixed_df.sort_values("duration", ascending=True)
    test_grouped = test_df.groupby(["underwriting", "sex", "issue_age"], observed=True)[
        "wrong_qx_fixed"
    ]
    for _, test_group in test_grouped:
        for i in range(len(test_group) - 1):
            assert (
                test_group.iloc[i] <= test_group.iloc[i + 1]
            ), "Expected wrong_qx_fixed to be monotonically increasing - duration"

    # attained_age - check monotonically increasing
    test_df = fixed_df.sort_values("attained_age", ascending=True)
    test_grouped = test_df.groupby(["underwriting", "sex", "issue_age"], observed=True)[
        "wrong_qx_fixed"
    ]
    for _, test_group in test_grouped:
        for i in range(len(test_group) - 1):
            assert (
                test_group.iloc[i] <= test_group.iloc[i + 1]
            ), "Expected wrong_qx_fixed to be monotonically increasing - attained age"

    # sex - check monotonically increasing
    test_df = fixed_df.sort_values("sex", ascending=True)
    test_grouped = test_df.groupby(
        ["underwriting", "issue_age", "duration"], observed=True
    )["wrong_qx_fixed"]
    for _, test_group in test_grouped:
        for i in range(len(test_group) - 1):
            assert (
                test_group.iloc[i] <= test_group.iloc[i + 1]
            ), "Expected wrong_qx_fixed to be monotonically increasing - sex"

    # underwriting - check monotonically increasing
    test_df = fixed_df.sort_values("underwriting", ascending=True)
    test_grouped = test_df.groupby(["sex", "issue_age", "duration"], observed=True)[
        "wrong_qx_fixed"
    ]
    for _, test_group in test_grouped:
        for i in range(len(test_group) - 1):
            assert (
                test_group.iloc[i] <= test_group.iloc[i + 1]
            ), "Expected wrong_qx_fixed to be monotonically increasing - underwriting"
