"""Constraint functions."""

from typing import Optional

import pandas as pd

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


class TableConstrainer:
    """
    TableConstrainer class.

    Provides a method to run a set of constraints on a dataframe and
    fix a column where it is not monotonically increasing. The constrainer
    can iterate up until a set limit and will fix rates by increasing it by
    the minimum increase.

    Parameters
    ----------
    col_to_fix : str
        The column to fix
    issue_age_col : str
        The column to use for issue age
    duration_col : str
        The column to use for duration
    attained_age_col : str
        The column to use for attained age
    other_feature_cols : Optional[list], optional
        The columns to use for other features, by default None
    min_increase : float, optional
        The minimum increase to apply to the column to fix, by default 0.001
    iteration_limit : int, optional
        The maximum number of iterations to run, by default 10

    """

    def __init__(
        self,
        col_to_fix: str,
        issue_age_col: str,
        duration_col: str,
        attained_age_col: str,
        other_feature_cols: Optional[list] = None,
        min_increase: float = 0.001,
        iteration_limit: int = 10,
    ) -> None:
        self.col_to_fix = col_to_fix
        self.issue_age_col = issue_age_col
        self.duration_col = duration_col
        self.attained_age_col = attained_age_col
        self.other_feature_cols = other_feature_cols
        self.min_increase = min_increase
        self.iteration_limit = iteration_limit

    def fix_df(
        self,
        df: pd.DataFrame,
        suppress_logs: bool = True,
    ) -> pd.DataFrame:
        """
        Run a set of constraints on a dataframe.

        There are two steps that will iterate until either the iteration limit is
        reached or no more fixes are needed.
        1. Run inner constraints (issue age, duration, attained age)
        2. Run outer constraints (other features e.g. gender, underwriting)

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to run constraint on
        suppress_logs : bool, optional
            Whether to suppress downstream logs, by default True

        Returns
        -------
        fixed_df : pd.DataFrame
            The fixed dataframe

        """
        logger.info(f"Fixing dataframe on `{self.col_to_fix}`")
        fixed_df = df
        fixed_col = f"{self.col_to_fix}_fixed"
        fixed_df[fixed_col] = fixed_df[self.col_to_fix]
        fixed_ind = "fixed_ind"
        fixed_df[fixed_ind] = 0

        params = {
            "col_to_fix": fixed_col,
            "min_increase": self.min_increase,
            "fixed_col": fixed_col,
        }

        inner_constraints = [
            {
                "constraint_col": self.issue_age_col,
                "other_feature_cols": [self.duration_col, *self.other_feature_cols],
            },
            {
                "constraint_col": self.duration_col,
                "other_feature_cols": [self.issue_age_col, *self.other_feature_cols],
            },
            {
                "constraint_col": self.attained_age_col,
                "other_feature_cols": [self.issue_age_col, *self.other_feature_cols],
            },
        ]

        outer_constraints = [
            {
                "constraint_col": col,
                "other_feature_cols": [
                    self.issue_age_col,
                    self.duration_col,
                    *[c for c in self.other_feature_cols if c != col],
                ],
            }
            for col in (self.other_feature_cols or [])
        ]

        # fix the wrong rates
        for iteration in range(1, self.iteration_limit + 1):
            starting_fixes = fixed_df["fixed_ind"].sum()

            # inner constraints
            if suppress_logs:
                fixed_df = custom_logger.suppress_logs(self.run_constraints)(
                    fixed_df, inner_constraints, **params
                )
            else:
                fixed_df = self.run_constraints(fixed_df, inner_constraints, **params)
            inner_nbr_of_fixes = fixed_df["fixed_ind"].sum() - starting_fixes

            # outer constraints
            if suppress_logs:
                fixed_df = custom_logger.suppress_logs(self.run_constraints)(
                    fixed_df, outer_constraints, **params
                )
            else:
                fixed_df = self.run_constraints(fixed_df, outer_constraints, **params)
            outer_nbr_of_fixes = (
                fixed_df["fixed_ind"].sum() - inner_nbr_of_fixes - starting_fixes
            )
            iteration_nbr_of_fixes = inner_nbr_of_fixes + outer_nbr_of_fixes
            logger.info(
                f"iteration {iteration}: "
                f"{inner_nbr_of_fixes} inner fixes + {outer_nbr_of_fixes} outer fixes"
            )
            if iteration_nbr_of_fixes == 0:
                break

        nbr_of_fixes = fixed_df["fixed_ind"].sum()
        if iteration_nbr_of_fixes == 0:
            logger.info(
                f"COMPLETED with {iteration - 1} iterations "
                f"and total fixes: {nbr_of_fixes}"
            )
        else:
            logger.warning(
                f"ITERATION LIMIT REACHED with {iteration} iterations "
                f"and total fixes: {nbr_of_fixes},"
                f"with {iteration_nbr_of_fixes} fixes remaining"
            )

        return fixed_df

    def run_constraints(
        self, df: pd.DataFrame, constraints_list: list, **kwargs
    ) -> pd.DataFrame:
        """
        Run a list of constraints.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to run constraint on
        constraints_list : list
            The list of constraints to run
        **kwargs : dict
            The kwargs to pass to the constraint function

        Returns
        -------
        fixed_df : pd.DataFrame
            The fixed dataframe

        """
        for constraint in constraints_list:
            df = run_constraint(
                df=df,
                constraint_col=constraint["constraint_col"],
                other_feature_cols=constraint["other_feature_cols"],
                **kwargs,
            )
        return df


def run_constraint(
    df: pd.DataFrame,
    col_to_fix: str,
    constraint_col: str,
    other_feature_cols: list,
    min_increase: float = 0.001,
    fixed_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run constraint on a dataframe.

    Ensures that the col_to_fix has a monotonically increasing rate
    for the constraint_col when grouped by other_feature_cols

    Note
    ----
    The constraint_col should be ordered the way that you want the rates to
    increase

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to run constraint on
    col_to_fix : str
        The name of the column to fix
    constraint_col : str
        The name of the column to that is constrained
    other_feature_cols : list
        The names of the columns to group by
    min_increase : float, optional
        The minimum increase to fix, by default .001
    fixed_col : str, optional
        The name for the fixed column, by default None
        if None the column will be fixed_col = col_to_fix + '_fixed'

    Returns
    -------
    fixed_df : pd.DataFrame
        The fixed dataframe

    """
    # initiate variables
    fixed_df = df.copy()
    fixed_df = fixed_df.sort_values([*other_feature_cols, constraint_col])
    nbr_of_rates = len(fixed_df)

    # fixed column
    if fixed_col is None:
        fixed_col = f"{col_to_fix}_fixed"
    fixed_df[fixed_col] = fixed_df[col_to_fix]

    # fixed indicator
    fixed_ind = "fixed_ind"
    if fixed_ind not in fixed_df.columns:
        fixed_df[fixed_ind] = 0
    starting_nbr_of_fixes = fixed_df[fixed_ind].sum()

    # check if groupings are unique
    grouped = fixed_df.groupby(other_feature_cols)

    group_size = fixed_df.groupby([*other_feature_cols, constraint_col])[
        col_to_fix
    ].transform("count")
    if not (group_size == 1).all():
        raise ValueError(
            "Groupings are not unique. Make sure `other_feature_cols` is correct."
        )

    rates_in_group = fixed_df.groupby([*other_feature_cols])[col_to_fix].size()
    if (rates_in_group == 1).all():
        logger.warning(
            f"For the `{constraint_col}` constraint there was only one rate in "
            "each group making it unique already. Make sure this is needed."
        )

    # for each group check that
    # current rates are greater than or equal to previous rates
    logger.debug(
        f"Running constraint \n"
        f"col_to_fix: {col_to_fix} \n"
        f"constraint_col: {constraint_col} \n"
        f"other_feature_cols: {other_feature_cols} \n"
        f"fixed_col: {fixed_col} \n"
        f"min_increase: {min_increase}"
    )
    for _, group in grouped:
        idxs = group.index.tolist()
        prev_rate = None
        for idx in idxs:
            current_rate = fixed_df.at[idx, fixed_col]
            if prev_rate is not None and current_rate < prev_rate:
                # make the fix
                fixed_df.at[idx, fixed_ind] = fixed_df.at[idx, fixed_ind] + 1
                fixed_df.at[idx, fixed_col] = prev_rate + min_increase
            prev_rate = current_rate

    ending_nbr_of_fixes = fixed_df[fixed_ind].sum()
    nbr_of_fixes = ending_nbr_of_fixes - starting_nbr_of_fixes
    logger.info(
        f"{nbr_of_fixes} out of {nbr_of_rates} rates need to be "
        f"fixed for '{constraint_col}'. There are a total of "
        f"{ending_nbr_of_fixes} fixes."
    )

    return fixed_df
