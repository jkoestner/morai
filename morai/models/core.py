"""Creates models for forecasting mortality rates."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import pygam
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import l, s
from pygam.terms import TermList
from scipy.stats import chi2
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.gam.api import BSplines, GLMGam

from morai.utils import custom_logger
from morai.utils.custom_logger import suppress_logs

logger = custom_logger.setup_logging(__name__)


class GLM(BaseEstimator, RegressorMixin):
    """
    Create a GLM model.

    The BaseEstimator and RegressorMixin classes are used to interface with
    scikit-learn with certain functions.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize the model."""
        self.r_style = None
        self.mapping = None
        self.model = None
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series = None,
        family: sm.families = None,
        r_style: bool = False,
        mapping: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        """
        Fit the GLM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        weights : pd.Series, optional
            The weights
        family : sm.families, optional
            The family to use for the GLM model
        r_style : bool, optional
            Whether to use R-style formulas
        mapping : dict, optional
            The mapping of the features to the encoding and only needed
            if r_style is True
        kwargs : dict, optional
            Additional keyword arguments to apply to the model

        Returns
        -------
        model : GLM
            The GLM model

        """
        logger.info("fiting the model")
        self.r_style = r_style
        self.mapping = mapping
        model = self._setup_model(X, y, weights, family, **kwargs)
        model = model.fit()
        self.model = model
        self.is_fitted_ = True

        return model

    def predict(self, X: pd.DataFrame, manual: bool = False) -> np.ndarray:
        """
        Predict the target.

        When fitting the logistic regression, the model will output a set of
        coefficients for each feature.

        The rate is then calculated using the following formula using the coefficients:
          rate = 1 / (1 + exp(constant +
                 coefficient_1 * feature_1 + ... + coefficient_n * feature_n))

        The rate can also be calculated as the product of the odds ratio:
          product_odds = odds_1 * odds_2 * ... * odds_n
          rate = product_odds / (1 + product_odds)

        This allows the rate to be calculated by a combination of a 1d rate and a
        multiplier table. There is a simplification in the rate table in that the rate
        is calculated as the odds of feature * the 1d rate. This will not match the
        predicted rate exactly. It will be off more at higher rates. Use the multiplier
        table with caution.

        simplification method:
          rate_combined = odds_feature * rate_1d

        exact method:
          odds_rate = (rate_1d / (1 - rate_1d))
          rate_combined = (odds_feature * odds_1d) / (1 + (odds_feature * odds_1d)

        Parameters
        ----------
        X : pd.DataFrame
            The features
        manual : bool, optional
            Whether to use the manual prediction

        Returns
        -------
        predictions : np.ndarray
            The predictions

        """
        if not self.is_fitted_:
            raise ValueError("model is not fitted use fit method")

        if self.model is None:
            raise ValueError("please create a model first")

        if manual:
            rate_params = self.model.params.loc[X.columns]
            linear_combination = rate_params["constant"]
            for feature, coefficient in rate_params.items():
                if feature != "constant":
                    linear_combination += coefficient * X[feature]

            predictions = 1 / (1 + np.exp(-linear_combination))
        else:
            predictions = np.array(self.model.predict(X))

        return predictions

    def get_formula(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Get the formula for the GLM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target

        Returns
        -------
        formula : str
            The formula

        """
        # creating formula that uses categories and passthrough
        if self.mapping:
            cat_pass_keys = {
                key: value
                for key, value in self.mapping.items()
                if value["type"] == "cat_pass"
            }
            other_keys = {
                key: value
                for key, value in self.mapping.items()
                if value["type"] != "cat_pass"
            }
            non_categorical_part = " + ".join(other_keys) if other_keys else ""
            categorical_part = (
                " + ".join([f"C({key})" for key in cat_pass_keys])
                if cat_pass_keys
                else ""
            )

            if non_categorical_part and categorical_part:
                formula = f"{y.name} ~ {non_categorical_part} + {categorical_part}"
            elif non_categorical_part:
                formula = f"{y.name} ~ {non_categorical_part}"
            elif categorical_part:
                formula = f"{y.name} ~ {categorical_part}"
            else:
                formula = f"{y.name} ~ 1"
        else:
            formula = f"{y.name} ~ {' + '.join(X.columns)}"

        logger.warning(f"Caution - Not thorougly tested. R-style formula: {formula}")

        return formula

    def get_odds(self, display: bool = False) -> pd.DataFrame:
        """
        Get the odds ratio.

        Parameters
        ----------
        display : bool, optional
            Whether to display the odds ratio

        Returns
        -------
        odds_ratio : pd.DataFrame
            The odds ratio

        """
        if not self.is_fitted_:
            raise ValueError("model is not fitted use fit method")

        if self.model is None:
            raise ValueError("please create a model first")

        model = self.model

        logger.info("generating odds ratio from model")
        odds_ratio = np.exp(model.params)

        # displaying chart of odds ratio
        if display:
            coef_df = pd.DataFrame(
                {"feature": model.params.index[1:], "coefficient": model.params[1:]}
            )
            coef_df = coef_df.sort_values("coefficient", ascending=False)

            odds_ratio = px.bar(
                coef_df,
                x="feature",
                y="coefficient",
                title="Feature Importance",
                labels={"coefficient": "Coefficient Value", "feature": "Features"},
            )

        return odds_ratio

    def get_feature_contributions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series = None,
        base_features: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Get the feature contributions.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        weights : pd.Series, optional
            The weights
        base_features : list, optional
            The base features to use for the contributions

        Returns
        -------
        feature_contributions : pd.DataFrame
            The feature contributions

        """
        contributions = []
        base_features = (
            ["constant"] if base_features is None else [*base_features, "constant"]
        )
        features = [feature for feature in X.columns if feature not in base_features]
        logger.info(
            f"generating feature contributions from model for {len(features)} features "
            f"by fitting the model and seeing the reduction in deviance."
        )
        logger.info(f"base features: '{base_features}'")

        # suppress logger for fitting
        original_log_level = logger.level
        logger.setLevel(50)

        # set up the deviances
        all_deviance = self.fit(X, y, weights).deviance
        contributions.append(
            {
                "features": "all",
                "contribution": 1,
                "deviance": all_deviance,
            }
        )
        base_deviance = self.fit(X[base_features], y, weights).deviance
        contributions.append(
            {
                "features": "base",
                "contribution": 0,
                "deviance": base_deviance,
            }
        )
        for feature in features:
            temp_features = [f for f in features if f != feature]
            reduced_deviance = self.fit(
                X[[*temp_features, *base_features]], y, weights
            ).deviance
            contribution = (reduced_deviance - all_deviance) / (
                base_deviance - all_deviance
            )
            contributions.append(
                {
                    "features": feature,
                    "contribution": contribution,
                    "deviance": reduced_deviance,
                }
            )
        feature_contributions = pd.DataFrame(contributions)
        # refit the model
        self.fit(X[[*features, *base_features]], y, weights)

        # reset log level
        logger.setLevel(original_log_level)

        return feature_contributions

    def _setup_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series = None,
        family: sm.families = None,
        **kwargs,
    ) -> Any:
        """
        Set up the GLM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        weights : pd.Series, optional
            The weights
        family : sm.families, optional
            The family to use for the GLM model
        kwargs : dict, optional
            Additional keyword arguments to apply to the model

        Returns
        -------
        model : GLM
            The GLM model

        """
        if family is None:
            family = sm.families.Binomial()
        logger.info(f"setup GLM model with statsmodels and {family} family...")

        # using either r-style or python-style formula
        if self.r_style:
            model_data = pd.concat([y, X], axis=1)
            formula = self.get_formula(X, y)
            model = smf.glm(
                formula=formula,
                data=model_data,
                family=family,
                freq_weights=weights,
                **kwargs,
            )
        else:
            model = sm.GLM(
                endog=y,
                exog=X,
                family=family,
                freq_weights=weights,
                **kwargs,
            )

        return model


class ModelWrapper:
    """Create a model wrapper to get make retrieving results agnostic."""

    def __init__(self, model: Any) -> None:
        """
        Initialize the model wrapper.

        Parameters
        ----------
        model : object
            The model to wrap

        """
        self.model = model

    def get_features(self) -> list:
        """
        Get the features from the model.

        Returns
        -------
        features : list
            The features

        """
        features = None
        feature_attrs = [
            "feature_name",
            "feature_names",
            "feature_names_",
            "feature_names_in_",
            "params",
        ]
        for attr in feature_attrs:
            if hasattr(self.model, attr):
                if attr == "params":
                    features = list(getattr(self.model, attr).keys())
                elif attr == "feature_name":
                    features = self.model.feature_name()
                else:
                    features = list(getattr(self.model, attr))
                if features:
                    break
        if not features:
            raise ValueError("Could not find `features` in the model.")
        return features

    def get_importance(self) -> pd.DataFrame:
        """
        Get the importance of the features.

        Returns
        -------
        importance_df : pd.DataFrame
            The importance of the features

        """
        importance = None
        importance_attrs = ["feature_importances_", "coef_", "params"]
        for attr in importance_attrs:
            if hasattr(self.model, attr):
                if attr == "params":
                    importance = getattr(self.model, attr).tolist()
                else:
                    importance = getattr(self.model, attr)
                if importance is not None:
                    break
        if importance is None:
            raise ValueError("Could not find `importance` in the model.")
        features = self.get_features()

        importance_df = pd.DataFrame({"feature": features, "importance": importance})
        importance_df = importance_df.sort_values(by="importance", ascending=False)
        return importance_df

    def check_predict(self) -> None:
        """Check if the model has a predict method."""
        if not hasattr(self.model, "predict"):
            raise ValueError("model does not have a predict method")


class LeeCarter:
    """
    Create a Lee Carter model.

    A Lee Carter model needs the data to be structured with the attained age
    and observation year. The model will need "qx_raw". If not in dataframe
    then actual and exposure will need to be used and the structure_df method
    will calculate "qx_raw".

    Note that the Lee Carter model assumes that the mortality rates are
    log-linear over time and age.

    reference:
    - https://en.wikipedia.org/wiki/Lee%E2%80%93Carter_model
    """

    def __init__(
        self,
        age_col: str = "attained_age",
        year_col: str = "observation_year",
        actual_col: str = "death_claim_amount",
        expose_col: str = "amount_exposed",
        interval: Optional[int] = None,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        age_col : str, optional
            The column name for the attained age
        year_col : str, optional
            The column name for the observation year
        actual_col : str, optional
            The column name for the actual values
        expose_col : str, optional
            The column name for the exposure values
        interval : int, optional
            The interval for age groups

        """
        logger.info("initialized LeeCarter")
        self.age_col = age_col
        self.year_col = year_col
        self.actual_col = actual_col
        self.expose_col = expose_col
        self.interval = interval
        # calculations
        self.a_x = None
        self.k_t = None
        self.b_x = None
        self.b_x_k_t = None
        self.lc_df = None
        # forecast
        self.k_t_i = None

    def structure_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Structure the data for the Lee Carter model.

        The Lee Carter model requires the data to be grouped by the attained
        age and observation year. The mortality rates are then calculated.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing the data to structure.

        Returns
        -------
        lc_df : pd.DataFrame
            lee carter data frame with qx_raw rates

        """
        # check if columns are in the dataframe
        cols_needed = [self.age_col, self.year_col, self.actual_col, self.expose_col]
        missing_cols = [col for col in cols_needed if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns in DataFrame: {', '.join(missing_cols)}. "
                f"Use LeeCarter(age_col, year_col, actual_col, expose_col) "
                f"to specify columns."
            )

        # grouping the data
        logger.info("grouping data by age and year")
        lc_df = (
            df.groupby([self.age_col, self.year_col], observed=True)[
                [self.actual_col, self.expose_col]
            ]
            .sum()
            .reset_index()
        )
        logger.info(
            f"calculating qx_raw rates using {self.actual_col} and {self.expose_col}"
        )
        lc_df["qx_raw"] = np.where(
            lc_df[self.expose_col] == 0,
            0,
            lc_df[self.actual_col] / lc_df[self.expose_col],
        )
        logger.info(
            f"floored {(lc_df['qx_raw'] <= 0).sum()} rates "
            f"to 0.000001 and capped {len(lc_df[lc_df['qx_raw'] >= 1])} rates "
            f"to 0.999999."
        )
        lc_df["qx_raw"] = lc_df["qx_raw"].clip(lower=0.000001, upper=0.999999)
        self.lc_df = lc_df
        logger.info(f"crude_df shape: {self.lc_df.shape}")

        return self.lc_df

    def fit(self, lc_df: pd.DataFrame, interval: Optional[int] = None) -> pd.DataFrame:
        """
        Fit the LeeCarter model from a crude_df which will add the qx_lc rates.

        There are 4 parameters when fitting the LeeCarter model:
          - a_x (age effect)
          - k_t (time trend)
          - b_x (age effect over time trend)
          - b_x_k_t (matrix multiply)

        Parameters
        ----------
        lc_df : pd.DataFrame
            A DataFrame containing crude mortality rates for a given population.
            - rows: year
            - columns: age
        interval : int, optional
            The interval of ages for each iteration of fit.
            Default is None which uses the entire range of ages in the data.

        Returns
        -------
        lc_df : pd.DataFrame
            A DataFrame containing the LeeCarter mortality rates.

        """
        # checks if models have data needed
        if self.year_col not in lc_df.columns or self.age_col not in lc_df.columns:
            raise ValueError(f"{self.age_col} and {self.year_col} are required")

        # initialize the variables
        logger.info("creating Lee Carter model with qx_raw rates...")
        crude_pivot = lc_df.pivot(
            index=self.year_col, columns=self.age_col, values="qx_raw"
        )
        a_x = {}
        k_t = {}
        b_x = {}
        b_x_k_t = {}
        predictions_list = []

        year_start = crude_pivot.index.min()
        year_end = crude_pivot.index.max()
        age_start = int(crude_pivot.columns.min())
        age_end = int(crude_pivot.columns.max())
        ages = crude_pivot.columns

        if interval is None and self.interval is not None:
            interval = self.interval
        elif interval is None:
            # interval would equal data range
            interval = len(ages)

        logger.info(f"age range: {age_start}, {age_end}")
        logger.info(f"year range: {year_start}, {year_end}")
        logger.info(f"creating `{len(ages) // interval}` intervals")

        for i in range(0, len(ages), interval):
            interval_ages = ages[i : i + interval]
            interval_pivot = crude_pivot[interval_ages]
            interval_age_range = f"{interval_ages[0]}-{interval_ages[-1]}"
            logger.debug(f"interval age range: {interval_age_range}")

            # qx is the mortality matrix
            log_qx = np.log(interval_pivot)

            # ax is the age effect (average mortality rate by age)
            logger.debug("calculating a_x")
            a_x[interval_age_range] = log_qx.mean(axis=0)

            # kt is the time trend
            logger.debug("calculating k_t")
            k_t[interval_age_range] = (log_qx - a_x[interval_age_range]).sum(axis=1)
            e1 = (log_qx - a_x[interval_age_range]).multiply(
                k_t[interval_age_range], axis="index"
            )
            e2 = e1.sum(axis=0)
            e3 = k_t[interval_age_range] ** 2
            e4 = e3.sum()

            # bx is the rate of change of age due to time trend
            logger.debug("calculating b_x")
            b_x[interval_age_range] = e2 / e4

            # matrix multiply for b_x_k_t
            logger.debug("calculating b_x_k_t")
            b_x_k_t_df = pd.DataFrame(
                np.outer(b_x[interval_age_range], k_t[interval_age_range])
            )
            b_x_k_t_df = b_x_k_t_df.transpose()
            b_x_k_t_df.index = interval_pivot.index
            b_x_k_t_df.columns = interval_pivot.columns
            b_x_k_t[interval_age_range] = b_x_k_t_df

            # calculate qx_lc
            logger.debug("calculating qx_lc = exp(a_x + b_x * k_t)")
            qx_log_lc = a_x[interval_age_range].values + b_x_k_t_df.values
            qx_log_lc = pd.DataFrame(
                qx_log_lc, index=interval_pivot.index, columns=interval_pivot.columns
            )
            qx_lc = np.exp(qx_log_lc)

            # append predictions to list
            predictions_list.append(
                qx_lc.reset_index().melt(
                    id_vars=self.year_col, var_name=self.age_col, value_name="qx_lc"
                )
            )

        # merge predictions back into lc_df
        logger.info("adding qx_lc to lc_df")
        predictions = pd.concat(predictions_list, axis=0)
        lc_df = pd.merge(
            lc_df,
            predictions,
            on=[self.year_col, self.age_col],
            how="left",
        ).astype({self.age_col: "int32", self.year_col: "int32"})

        # saving variables in class
        self.a_x = a_x
        self.k_t = k_t
        self.b_x = b_x
        self.b_x_k_t = b_x_k_t
        self.lc_df = lc_df
        self.interval = interval

        return lc_df

    def forecast(self, years: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Forecast the mortality rates using deterministic random walk.

        Parameters
        ----------
        years : int
            The amount of years to forecast LeeCarter model
        seed : int, optional
            The seed for the random number generator

        Returns
        -------
        lcf_df : pd.DataFrame
            A DataFrame containing the forecasted LeeCarter mortality rates.

        """
        # checks if models have data needed
        if self.lc_df is None:
            raise ValueError(
                "model is not fitted use fit method please use fit() method"
            )

        # initialize the variables
        variance = 0
        k_t_i = {}
        forecast_list = []
        a_x = self.a_x
        k_t = self.k_t
        b_x = self.b_x
        b_x_k_t = self.b_x_k_t

        logger.info("forecasting qx_lc using deterministic random walk...")
        for interval_age_range in a_x.keys():
            # year columns
            year_cols = list(
                range(
                    k_t[interval_age_range].index[-1] + 1,
                    k_t[interval_age_range].index[-1] + years + 1,
                )
            )
            # average change in k_t
            mu = (
                k_t[interval_age_range].iloc[-1] - k_t[interval_age_range].iloc[0]
            ) / len(k_t[interval_age_range])

            # random walk
            rng = np.random.default_rng(seed=seed)
            k_t_i[interval_age_range] = (
                k_t[interval_age_range].iloc[-1]
                + mu * np.arange(1, years + 1)
                + rng.normal(scale=variance, size=years)
            )

            # qx_lc forecast
            b_x_k_t_i = pd.DataFrame(
                np.outer(b_x[interval_age_range], k_t_i[interval_age_range])
            )
            b_x_k_t_i = b_x_k_t_i.transpose()
            qx_log_lc = a_x[interval_age_range].values + b_x_k_t_i.values
            qx_lc = np.exp(qx_log_lc)

            # append forecasts to list
            forecast = pd.DataFrame(
                qx_lc,
                index=year_cols,
                columns=b_x_k_t[interval_age_range].columns,
            )
            forecast.index.name = self.year_col
            forecast_list.append(
                forecast.reset_index().melt(
                    id_vars=self.year_col, var_name=self.age_col, value_name="qx_lc"
                )
            )

        # dataframe with forecast
        lcf_df = pd.concat(forecast_list, axis=0)

        # storing variables in class
        self.k_t_i = k_t_i

        return lcf_df

    def map(
        self,
        df: pd.DataFrame,
        age_col: Optional[str] = None,
        year_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Map the mortality rates from the Lee Carter model.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing the data to predict.
        age_col : str, optional
            The column name for the attained age
        year_col : str, optional
            The column name for the observation year

        Returns
        -------
        lc_df : pd.DataFrame
            A DataFrame containing the predicted mortality rates.

        """
        if age_col is None:
            age_col = self.age_col
        if year_col is None:
            year_col = self.year_col
        lc_df = self.lc_df

        # checks if models have data needed
        if lc_df is None:
            raise ValueError(
                "model is not fitted use fit method please use fit() method"
            )
        if year_col not in lc_df.columns or age_col not in lc_df.columns:
            raise ValueError(f"{age_col} and {year_col} are required")

        # map rates to df
        logger.info("mapping qx_lc to df")
        lc_df = lc_df.rename(columns={self.age_col: age_col, self.year_col: year_col})
        lc_df = pd.merge(
            df,
            lc_df[[age_col, year_col, "qx_lc"]],
            on=[age_col, year_col],
            how="left",
            suffixes=("_old", ""),
        )
        if "qx_lc_old" in lc_df.columns:
            lc_df.drop(columns=["qx_lc_old"], inplace=True)

        return lc_df


class CBD:
    """
    Create a Cairns, Blake, Dowd (CBD) model.

    A CBD model needs the data to be structured with the attained age
    and observation year. The model will need "qx_raw". If not in dataframe
    then actual and exposure will need to be used and the structure_df method
    will calculate "qx_raw".

    reference:
    - https://www.actuaries.org/AFIR/Colloquia/Rome2/Cairns_Blake_Dowd.pdf
    """

    def __init__(
        self,
        age_col: str = "attained_age",
        year_col: str = "observation_year",
        actual_col: str = "death_claim_amount",
        expose_col: str = "amount_exposed",
        interval: Optional[int] = None,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        age_col : str, optional
            The column name for the attained age
        year_col : str, optional
            The column name for the observation year
        actual_col : str, optional
            The column name for the actual values
        expose_col : str, optional
            The column name for the exposure values
        interval : int, optional
            The interval for age groups

        """
        logger.info("initialized CBD")
        self.age_col = age_col
        self.year_col = year_col
        self.actual_col = actual_col
        self.expose_col = expose_col
        self.interval = interval
        # calculations
        self.age_diff = None
        self.age_columns = None
        self.k_t_1 = None
        self.k_t_2 = None
        self.cbd_df = None
        # forecast
        self.k_1_f = None
        self.k_2_f = None

    def structure_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Structure the data for the CBD model.

        The CBD model requires the data to be the mortality rates with
        the columns as the attained age and the rows as the observation year.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing the data to structure.

        Returns
        -------
        cbd_df : pd.DataFrame
            CBD data frame

        """
        # check if columns are in the dataframe
        cols_needed = [self.age_col, self.year_col, self.actual_col, self.expose_col]
        missing_cols = [col for col in cols_needed if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns in DataFrame: {', '.join(missing_cols)}. "
                f"Use CBD(age_col, year_col, actual_col, expose_col) "
                f"to specify column names."
            )

        # grouping the data
        logger.info("grouping data by age and year")
        cbd_df = (
            df.groupby([self.age_col, self.year_col], observed=True)[
                [self.actual_col, self.expose_col]
            ]
            .sum()
            .reset_index()
        )
        logger.info(
            f"calculating qx_raw rates using {self.actual_col} and {self.expose_col}"
        )
        cbd_df["qx_raw"] = np.where(
            cbd_df[self.expose_col] == 0,
            0,
            cbd_df[self.actual_col] / cbd_df[self.expose_col],
        )
        logger.info(
            f"floored {(cbd_df['qx_raw'] <= 0).sum()} rates "
            f"to 0.000001 and capped {len(cbd_df[cbd_df['qx_raw'] >= 1])} rates "
            f"to 0.999999."
        )
        cbd_df["qx_raw"] = cbd_df["qx_raw"].clip(lower=0.000001, upper=0.999999)
        self.cbd_df = cbd_df
        logger.info(f"cbd_df shape: {self.cbd_df.shape}")

        return self.cbd_df

    def fit(
        self,
        cbd_df: pd.DataFrame,
        interval: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get the forecasted mortality rates.

        Parameters
        ----------
        cbd_df : pd.DataFrame
            A DataFrame containing crude mortality rates for a given population.
        interval : int, optional
            The interval for age groups

        Returns
        -------
        cbd_df : pd.DataFrame
            A DataFrame containing the forecasted mortality rates.
            - rows: year
            - columns: age

        """
        logger.info("creating CBD model with qx_raw rates...")
        crude_pivot = cbd_df.pivot(
            index=self.year_col, columns=self.age_col, values="qx_raw"
        )
        k_t_1 = {}
        k_t_2 = {}
        age_diff = {}
        age_columns = {}
        predictions_list = []

        year_start = crude_pivot.index.min()
        year_end = crude_pivot.index.max()
        ages = crude_pivot.columns
        age_start = int(ages.min())
        age_end = int(ages.max())

        if interval is None and self.interval is not None:
            interval = self.interval
        elif interval is None:
            # interval would equal data range
            interval = len(ages)

        logger.info(f"age range: {age_start}, {age_end}")
        logger.info(f"year range: {year_start}, {year_end}")
        logger.info(f"creating `{len(ages) // interval}` intervals")

        # error if there is only one age
        if interval < 2 or len(ages) % interval == 1:
            logger.error("age range must have more than one age")
            return None

        for i in range(0, len(ages), interval):
            interval_ages = ages[i : i + interval]
            interval_pivot = crude_pivot[interval_ages]
            interval_age_range = f"{interval_ages[0]}-{interval_ages[-1]}"
            interval_mean_age = interval_ages.to_series().mean()
            age_columns[interval_age_range] = interval_pivot.columns
            logger.debug(f"interval age range: {interval_age_range}")

            # qx_logit is the mortality matrix
            logger.debug("calculating qx_logit")
            qx_logit = self._logit(interval_pivot)

            # k_t_1 is the age effect (average mortality rate by age)
            logger.debug("calculating k_t_1 = mean rate per year")
            k_t_1[interval_age_range] = qx_logit.mean(axis=1)

            # k_t_2 is the slope component
            logger.debug(
                "calculating k_t_2 = e1 / e2 \n"
                "e1 = Σ((age - age_mean) * qx_logit) \n"
                "e2 = Σ((age - age_mean)^2)"
            )
            age_diff[interval_age_range] = interval_ages - interval_mean_age
            e1 = (age_diff[interval_age_range] * qx_logit).sum(axis=1)
            e2 = (age_diff[interval_age_range].values ** 2).sum()
            k_t_2[interval_age_range] = e1 / e2

            # qx_logit
            logger.debug("calculating qx_logit_cbd = k_t_1 + (age - age_mean) * k_t_2")
            qx_logit_cbd = k_t_1[interval_age_range].values[:, np.newaxis] + (
                age_diff[interval_age_range].values
                * k_t_2[interval_age_range].values[:, np.newaxis]
            )
            qx_logit_cbd = pd.DataFrame(
                qx_logit_cbd, index=qx_logit.index, columns=qx_logit.columns
            )

            # qx_cbd
            logger.debug(
                "calculating qx_cbd = exp(qx_logit_cbd) / (1 + exp(qx_logit_cbd))"
            )
            qx_cbd = np.exp(qx_logit_cbd) / (1 + np.exp(qx_logit_cbd))

            # append predictions to list
            predictions_list.append(
                qx_cbd.reset_index().melt(
                    id_vars=self.year_col, var_name=self.age_col, value_name="qx_cbd"
                )
            )

        # merge predictions back into cbd_df
        logger.info("adding qx_cbd to cbd_df")
        predictions = pd.concat(predictions_list, axis=0)
        cbd_df = pd.merge(
            cbd_df,
            predictions,
            on=[self.year_col, self.age_col],
            how="left",
        ).astype({self.age_col: "int32", self.year_col: "int32"})

        # saving variables in class
        self.k_t_1 = k_t_1
        self.k_t_2 = k_t_2
        self.age_diff = age_diff
        self.age_columns = age_columns
        self.cbd_df = cbd_df
        self.interval = interval

        return cbd_df

    def forecast(self, years: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Forecast the mortality rates using deterministic random walk.

        Parameters
        ----------
        years : int
            The amount of years to forecast CBD model
        seed : int, optional
            The seed for the random number generator

        Returns
        -------
        cbd_df : pd.DataFrame
            A DataFrame containing the forecasted CBD mortality rates.

        """
        # checks if models have data needed
        if self.cbd_df is None:
            raise ValueError(
                "model is not fitted use fit method please use fit() method"
            )

        # initialize the variables
        variance = 0
        forecast_list = []
        k_t_1 = self.k_t_1
        k_t_2 = self.k_t_2
        age_diff = self.age_diff
        age_columns = self.age_columns

        logger.info("forecasting qx_cbd using deterministic random walk...")
        for interval_age_range in k_t_1.keys():
            # year columns
            year_cols = list(
                range(
                    k_t_1[interval_age_range].index[-1] + 1,
                    k_t_1[interval_age_range].index[-1] + years + 1,
                )
            )
            # average change in k_t_1 and k_t_2
            mu = [
                (k_t_1[interval_age_range].iloc[-1] - k_t_1[interval_age_range].iloc[0])
                / len(k_t_1[interval_age_range]),
                (k_t_2[interval_age_range].iloc[-1] - k_t_2[interval_age_range].iloc[0])
                / len(k_t_2[interval_age_range]),
            ]

            # random walk
            rng = np.random.default_rng(seed=seed)
            k_1_f = (
                k_t_1[interval_age_range].iloc[-1]
                + mu[0] * np.arange(1, years + 1)
                + rng.normal(scale=variance, size=years)
            )
            k_1_f = pd.Series(data=k_1_f, index=year_cols)
            k_1_f.index.name = self.year_col
            k_2_f = (
                k_t_2[interval_age_range].iloc[-1]
                + mu[1] * np.arange(1, years + 1)
                + rng.normal(scale=variance, size=years)
            )
            k_2_f = pd.Series(data=k_2_f, index=year_cols)
            k_2_f.index.name = self.year_col

            # qx_logit
            logger.debug("calculating qx_logit_cbd = k_t_1 + (age - age_mean) * k_t_2")
            qx_logit_cbd = k_1_f.values[:, np.newaxis] + (
                age_diff[interval_age_range].values * k_2_f.values[:, np.newaxis]
            )
            qx_logit_cbd = pd.DataFrame(
                qx_logit_cbd, index=year_cols, columns=age_columns[interval_age_range]
            )

            # qx_cbd
            logger.debug(
                "calculating qx_cbd = exp(qx_logit_cbd) / (1 + exp(qx_logit_cbd))"
            )
            qx_cbd = np.exp(qx_logit_cbd) / (1 + np.exp(qx_logit_cbd))

            # append forecasts to list
            forecast = pd.DataFrame(
                qx_cbd,
                index=year_cols,
                columns=age_columns[interval_age_range],
            )
            forecast.index.name = self.year_col
            forecast_list.append(
                forecast.reset_index().melt(
                    id_vars=self.year_col, var_name=self.age_col, value_name="qx_cbd"
                )
            )

        # dataframe with forecast
        cbdf_df = pd.concat(forecast_list, axis=0)

        # saving variables in class
        self.k_1_f = k_1_f
        self.k_2_f = k_2_f

        return cbdf_df

    def map(
        self,
        df: pd.DataFrame,
        age_col: Optional[str] = None,
        year_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Map the mortality rates from the CBD model.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing the data to predict.
        age_col : str, optional
            The column name for the attained age
        year_col : str, optional
            The column name for the observation year

        Returns
        -------
        cbd_df : pd.DataFrame
            A DataFrame containing the predicted mortality rates.

        """
        if age_col is None:
            age_col = self.age_col
        if year_col is None:
            year_col = self.year_col
        cbd_df = self.cbd_df

        # checks if models have data needed
        if cbd_df is None:
            raise ValueError(
                "model is not fitted use fit method please use fit() method"
            )
        if year_col not in cbd_df.columns or age_col not in cbd_df.columns:
            raise ValueError(f"{age_col} and {year_col} are required")

        # map rates to df
        logger.info("mapping qx_cbd to df")
        cbd_df = cbd_df.rename(columns={self.age_col: age_col, self.year_col: year_col})
        cbd_df = pd.merge(
            df,
            cbd_df[[age_col, year_col, "qx_cbd"]],
            on=[age_col, year_col],
            how="left",
            suffixes=("_old", ""),
        )
        if "qx_cbd_old" in cbd_df.columns:
            cbd_df.drop(columns=["qx_cbd_old"], inplace=True)

        return cbd_df

    def _logit(self, a: float) -> float:
        """
        Logit function.

        Parameters
        ----------
        a : float
            The value

        Returns
        -------
        logit : float
            The logit value

        """
        return np.log(a / (1 - a))


class GAMPy(BaseEstimator, RegressorMixin):
    """
    Create a GAM model - pygam wrapper.

    The BaseEstimator and RegressorMixin classes are used to interface with
    scikit-learn with certain functions.

    source function:
    https://github.com/dswah/pyGAM

    limitations:
    - does not seem to be actively maintained

    """

    def __init__(
        self,
    ) -> None:
        """Initialize the model."""
        self.X = None
        self.y = None
        self.weights = None
        self.spline_dict = None
        self.unfit_model = None
        self.model = None
        self.is_fitted_ = False

    def setup_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series = None,
        distribution: str = "binomial",
        link: str = "logit",
        spline_dict: Optional[dict] = None,
        alpha: float = 0,
        save: bool = True,
        **kwargs,
    ) -> Any:
        """
        Set up the GAM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        weights : pd.Series, optional
            The weights
        distribution : str, optional
            The distribution to use
        link : str, optional
            The link function to use
        spline_dict : dict, optional
            The dictionary of the splines to use for the GAM model
            example:
                {
                    "column_1": {"df": 12, "degree": 3},
                    "column_2": {"df": 10, "degree": 3},
                }
            defaults:
              - df: 10
              - degree: 3
              - constraints: None
              - drop: True
            notes:
              - having a higher degree of freedom will allow the alpha search to limit
              how much the model can penalize the features. Too few degrees of freedom
              and the model may not model the complexity well.
              - it's better to drop the initial spline column as it would create both a
              linear and non-linear relationship which may introduce multicollinearity
        alpha : float, optional (default=0)
            The alpha value for the GAM model
        save : bool, optional
            Save the variables in the class
        kwargs : dict, optional
            Additional keyword arguments to apply to the model

        Returns
        -------
        unfit_model : GAM
            The GAM model

        """
        logger.info(
            f"setup GAM model with pygam and `{distribution}` distribution "
            f"with `{link}` link"
        )

        # create terms
        terms = self.get_terms(X=X, spline_dict=spline_dict)

        # creating the model
        unfit_model = pygam.GAM(
            terms=terms,
            distribution=distribution,
            link=link,
            lam=alpha,
            fit_intercept=False,
            **kwargs,
        )

        # save the variables
        if save:
            self.X = X
            self.y = y
            self.weights = weights
            self.spline_dict = spline_dict
            self.unfit_model = unfit_model

        return unfit_model

    def fit(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        weights: pd.Series = None,
    ) -> Any:
        """
        Fit the GAM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        weights : pd.Series, optional
            The weights

        Returns
        -------
        model : GAM
            The GAM model

        """
        # check if variables are saved
        if (X is None and self.X is None) or (y is None and self.y is None):
            raise ValueError("Need to provide X and y or save the variables")
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if weights is None:
            weights = self.weights

        # fit model
        logger.info("fiting the model")
        model = self.unfit_model.fit(X=X, y=y, weights=weights)

        self.model = model
        self.is_fitted_ = True

        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target.

        Parameters
        ----------
        X : pd.DataFrame
            The features

        Returns
        -------
        predictions : np.ndarray
            The predictions

        """
        if not self.is_fitted_:
            raise ValueError("model is not fitted use fit method")

        if self.model is None:
            raise ValueError("please create a model first")

        predictions = np.array(self.model.predict(X))

        return predictions

    def get_terms(self, X: pd.DataFrame, spline_dict: Optional[dict] = None) -> str:
        """
        Get the terms for the GAM model.

        Assumes that all columns are linear except for the columns in the spline_dict.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        spline_dict : dict, optional
            The dictionary of the splines to use for the GAM model

        Returns
        -------
        term_list : list
            The terms

        """
        term_list = []
        for i, col in enumerate(X.columns):
            if spline_dict and col in spline_dict:
                term_list.append(
                    s(
                        i,
                        n_splines=spline_dict[col].get("df", 5),
                        spline_order=spline_dict[col].get("degree", 3),
                    )
                )
            else:
                term_list.append(l(i))

        terms = TermList(*term_list)

        return terms


class GAMStats(BaseEstimator, RegressorMixin):
    """
    Create a GAM model - statsmodels wrapper.

    The BaseEstimator and RegressorMixin classes are used to interface with
    scikit-learn with certain functions.

    source function:
    https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html#

    limitations:
    - does not seem to support weights

    """

    def __init__(
        self,
    ) -> None:
        """Initialize the model."""
        self.X = None
        self.y = None
        self.weights = None
        self.spline_dict = None
        self.r_style = None
        self.mapping = None
        self.unfit_model = None
        self.model = None
        self.smoother = None
        self.is_fitted_ = False

    def setup_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series = None,
        family: sm.families = None,
        spline_dict: Optional[dict] = None,
        alpha: float = 0,
        save: bool = True,
        **kwargs,
    ) -> Any:
        """
        Set up the GAM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        weights : pd.Series, optional
            The weights
        family : sm.families, optional
            The family to use for the GAM model
        spline_dict : dict, optional
            The dictionary of the splines to use for the GAM model
            example:
                {
                    "column_1": {"df": 12, "degree": 3},
                    "column_2": {"df": 10, "degree": 3},
                }
            defaults:
              - df: 10
              - degree: 3
              - constraints: None
              - drop: True
            function:
              - https://www.statsmodels.org/stable/generated/statsmodels.gam.smooth_basis.BSplines.html
            notes:
              - having a higher degree of freedom will allow the alpha search to limit
              how much the model can penalize the features. Too few degrees of freedom
              and the model may not model the complexity well.
              - it's better to drop the initial spline column as it would create both a
              linear and non-linear relationship which may introduce multicollinearity
        alpha : float, optional (default=0)
            The alpha value for the GAM model
        save : bool, optional
            Save the variables in the class
        kwargs : dict, optional
            Additional keyword arguments to apply to the model

        Returns
        -------
        unfit_model : GAM
            The GAM model

        """
        if family is None:
            family = sm.families.Binomial()
        logger.info(f"setup GAM model with statsmodels and {type(family)} family...")

        # create the smoother
        smoother, X_model = self.create_smoother(X=X, spline_dict=spline_dict)

        # creating the model
        # using either r-style or python-style formula
        if self.r_style:
            formula = self.get_formula(X=X, y=y, smoother=smoother)
            model_data = pd.concat([y, X], axis=1)
            unfit_model = GLMGam.from_formula(
                formula=formula,
                data=model_data,
                family=family,
                freq_weights=weights,
                smoother=smoother,
                alpha=alpha,
                **kwargs,
            )
        else:
            unfit_model = GLMGam(
                endog=y,
                exog=X_model,
                family=family,
                freq_weights=weights,
                smoother=smoother,
                alpha=alpha,
                **kwargs,
            )

        # save the variables
        if save:
            self.X = X
            self.y = y
            self.weights = weights
            self.spline_dict = spline_dict
            self.unfit_model = unfit_model
            self.smoother = smoother

        return unfit_model

    def create_smoother(
        self, X: pd.DataFrame, spline_dict: Optional[dict] = None
    ) -> Any:
        """
        Create the smoother for the GAM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        spline_dict : dict, optional
            The dictionary of the splines to use for the GAM model

        Returns
        -------
        smoother : BSplines
            The smoother for the GAM model
        X_model : pd.DataFrame
            The features without the spline columns

        """
        if spline_dict is None:
            spline_dict = self.spline_dict

        # create the splines and get the attributes
        # drop the initial spline column by default
        spline_cols = list(spline_dict.keys())
        splines = X[spline_cols]
        attributes = spline_dict[next(iter(spline_dict))].keys()
        attr_lists = {f"{attr}_list": [] for attr in attributes}
        for spline in spline_cols:
            for attr in attributes:
                attr_lists[f"{attr}_list"].append(spline_dict[spline][attr])
        df_list = attr_lists.get("df_list", [10] * len(spline_cols))
        degree_list = attr_lists.get("degree_list", [3] * len(spline_cols))
        drop_list = attr_lists.get("drop_list", [True] * len(spline_cols))
        drop_cols = [spline_cols[i] for i, drop in enumerate(drop_list) if drop]

        # create the smoother
        smoother = BSplines(splines, df=df_list, degree=degree_list)
        X_model = X.drop(columns=drop_cols)
        logger.info(f"created splines for `{smoother.variable_names}`")

        return smoother, X_model

    def search_alpha(self, sample: bool = True, k_folds: int = 5, **kwargs) -> float:
        """
        Search for the best alpha value for the GAM model.

        function:
            - https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.select_penweight_kfold.html#statsmodels.gam.generalized_additive_model.GLMGam.select_penweight_kfold

        Parameters
        ----------
        sample : bool
            Sample dataset to speed up search
        k_folds : int, optional
            The number of folds to use for the search
        kwargs : dict, optional
            Additional keyword arguments to apply to the search

        Returns
        -------
        alpha_best : float
            The best alpha value

        """
        if self.unfit_model is None:
            raise ValueError("please create a model first")

        unfit_model = self.unfit_model
        k_smooths = unfit_model.k_smooths
        logger.info(f"searching for best alpha with `{k_folds}` k_folds")

        # sample dataset to 10k rows for faster search
        if sample and len(self.X) > 10000:
            logger.info("sampling dataset for faster search")
            sample_X = self.X.sample(10000, random_state=42)
            sample_y = self.y.loc[sample_X.index]
            sample_weights = (
                self.weights.loc[sample_X.index] if self.weights is not None else None
            )
            sample_gam = suppress_logs(self.setup_model)(
                X=sample_X,
                y=sample_y,
                weights=sample_weights,
                family=unfit_model.family,
                spline_dict=self.spline_dict,
                save=False,
            )
            unfit_model = sample_gam

        # automate search
        alpha_grid = [np.logspace(-3, 3, 25) for _ in range(k_smooths)]
        alpha_best, _ = unfit_model.select_penweight_kfold(
            alphas=alpha_grid, k_folds=k_folds
        )
        logger.info(f"best alpha value: {alpha_best}")
        self.unfit_model.alpha = np.array(alpha_best).ravel()

        return alpha_best

    def fit(
        self,
        **kwargs,
    ) -> Any:
        """
        Fit the GAM model.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keyword arguments to apply to the model

        Returns
        -------
        model : GAM
            The GAM model

        """
        if kwargs.get("maxiter") is None:
            kwargs["maxiter"] = 100

        # fit model
        logger.info("fiting the model")
        model = self.unfit_model.fit(maxiter=kwargs["maxiter"])

        self.model = model
        self.is_fitted_ = True

        # effective degrees of freedom from penalty
        adf = len(self.smoother.col_names) + 1
        mean_exposure = (
            np.mean(self.unfit_model.freq_weights)
            if self.unfit_model.freq_weights is not None
            else 1.0
        )
        edf = self.model.edf[
            [col in self.smoother.col_names for col in list(self.model.params.index)]
        ].sum()
        edf_normalized = edf / mean_exposure
        logger.info(
            f"`{adf}` degrees of freedom for smoother with `{edf_normalized:.2f}` "
            f"being effective"
        )

        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target.

        Parameters
        ----------
        X : pd.DataFrame
            The features

        Returns
        -------
        predictions : np.ndarray
            The predictions

        """
        if not self.is_fitted_:
            raise ValueError("model is not fitted use fit method")

        if self.model is None:
            raise ValueError("please create a model first")

        smoother_cols = self.smoother.variable_names
        non_smoother_cols = [col for col in X.columns if col not in smoother_cols]
        predictions = np.array(
            self.model.predict(exog=X[non_smoother_cols], exog_smooth=X[smoother_cols])
        )

        return predictions

    def get_formula(self, X: pd.DataFrame, y: pd.Series, smoother: Any) -> str:
        """
        Get the formula for the GAM model.

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        smoother : Any
            The smoother with the splines

        Returns
        -------
        formula : str
            The formula

        """
        if self.mapping:
            # categorical and linear
            cat_pass_keys = {
                key: value
                for key, value in self.mapping.items()
                if value["type"] == "cat_pass" and key not in self.smoother_cols
            }
            other_keys = {
                key: value
                for key, value in self.mapping.items()
                if value["type"] != "cat_pass" and key not in self.smoother_cols
            }
            non_categorical_part = " + ".join(other_keys) if other_keys else ""
            categorical_part = (
                " + ".join([f"C({key})" for key in cat_pass_keys])
                if cat_pass_keys
                else ""
            )

            if non_categorical_part and categorical_part:
                formula = f"{y.name} ~ {non_categorical_part} + {categorical_part}"
            elif non_categorical_part:
                formula = f"{y.name} ~ {non_categorical_part}"
            elif categorical_part:
                formula = f"{y.name} ~ {categorical_part}"
            else:
                formula = f"{y.name} ~ 1"
        else:
            # assumes all linear
            non_smoother_cols = [
                col for col in X.columns if col not in self.smoother_cols
            ]
            formula = f"{y.name} ~ {' + '.join(non_smoother_cols)}"

        logger.warning(f"Caution - Not thorougly tested. R-style formula: {formula}")

        return formula


def calc_likelihood_ratio(full_model: Any, reduced_model: Any) -> dict:
    """
    Calculate the likelihood ratio.

    In statistics, the likelihood-ratio test assesses the goodness of fit
    of two competing statistical models.

    Parameters
    ----------
    full_model : model
        The full model
    reduced_model : model
        The reduced model


    Returns
    -------
    likelihood_dict: dict
        The likelihood ratio and p-value

    References
    ----------
    https://en.wikipedia.org/wiki/Likelihood-ratio_test

    """
    # validation checks
    if not hasattr(full_model, "llf") or not hasattr(reduced_model, "llf"):
        raise ValueError("models do not have a log-likelihood attribute")
    if full_model.df_model <= reduced_model.df_model:
        raise ValueError("full model has less or equal degrees of freedom than reduced")

    # calculate the likelihood ratio
    logger.info("calculating likelihood ratio")
    lr = 2 * (full_model.llf - reduced_model.llf)
    degrees_of_freedom_diff = full_model.df_model - reduced_model.df_model
    p_value = chi2.sf(lr, degrees_of_freedom_diff)

    # create the dictionary
    likelihood_dict = {
        "likelihood_ratio": lr,
        "degrees_of_freedom_diff": degrees_of_freedom_diff,
        "p_value": p_value,
    }

    return likelihood_dict
