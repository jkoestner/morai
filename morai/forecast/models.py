"""Creates models for forecasting mortality rates."""

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.base import BaseEstimator, RegressorMixin

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


class GLM(BaseEstimator, RegressorMixin):
    """
    Create a GLM model.

    The BaseEstimator and RegressorMixin classes are used to interface with
    scikit-learn with certain functions.
    """

    def __init__(
        self,
    ):
        """Initialize the model."""
        self.r_style = None
        self.mapping = None
        self.model = None
        self.is_fitted_ = False

    def fit(
        self, X, y, weights=None, family=None, r_style=False, mapping=None, **kwargs
    ):
        """
        Fit the GLM model.

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

    def predict(self, X, manual=False):
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

    def get_formula(self, X, y):
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

        logger.info(f"using R-style formula: {formula}")

        return formula

    def get_odds(self, display=False):
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

    def get_feature_contributions(self, X, y, weights=None, base_features=None):
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

    def _setup_model(self, X, y, weights=None, family=None, **kwargs):
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

    def __init__(self, model):
        """
        Initialize the model wrapper.

        Parameters
        ----------
        model : object
            The model to wrap

        """
        self.model = model

    def get_features(self):
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

    def get_importance(self):
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

    def check_predict(self):
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
        age_col="attained_age",
        year_col="observation_year",
        actual_col="death_claim_amount",
        expose_col="amount_exposed",
    ):
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

        """
        logger.info("initialized LeeCarter")
        self.age_col = age_col
        self.year_col = year_col
        self.actual_col = actual_col
        self.expose_col = expose_col
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
        df,
    ):
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
            lc_df[self.actual_col] == 0,
            0,
            lc_df[self.actual_col] / lc_df[self.expose_col],
        )
        logger.info(
            f"there were {len(lc_df[lc_df['qx_raw']>1])} rates "
            f"over 1 that were capped."
        )
        lc_df["qx_raw"] = lc_df["qx_raw"].clip(upper=1)
        self.lc_df = lc_df
        logger.info(f"crude_df shape: {self.lc_df.shape}")

        return self.lc_df

    def fit(self, lc_df):
        """
        Fit the LeeCarter model from a crude_df which will add the qx_lc rates.

        Parameters
        ----------
        lc_df : pd.DataFrame
            A DataFrame containing crude mortality rates for a given population.
            - rows: year
            - columns: age

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

        year_start = crude_pivot.index.min()
        year_end = crude_pivot.index.max()
        age_start = int(crude_pivot.columns.min())
        age_end = int(crude_pivot.columns.max())

        logger.info(f"age range: {age_start}, {age_end}")
        logger.info(f"year range: {year_start}, {year_end}")

        # qx is the mortality matrix
        log_qx = np.log(crude_pivot)

        # ax is the age effect (average mortality rate by age)
        logger.debug("calculating a_x")
        a_x = log_qx.mean(axis=0)
        self.a_x = a_x

        # kt is the time trend
        logger.debug("calculating k_t")
        self.k_t = (log_qx - self.a_x).sum(axis=1)
        e1 = (log_qx - self.a_x).multiply(self.k_t, axis="index")
        e2 = e1.sum(axis=0)
        e3 = self.k_t**2
        e4 = e3.sum()

        # bx is the rate of change of age due to time trend
        logger.debug("calculating b_x")
        b_x = e2 / e4
        self.b_x = b_x

        # matrix multiply for b_x_k_t
        logger.debug("calculating b_x_k_t")
        b_x_k_t = pd.DataFrame(np.outer(self.b_x, self.k_t))
        b_x_k_t = b_x_k_t.transpose()
        b_x_k_t.index = crude_pivot.index
        b_x_k_t.columns = crude_pivot.columns
        self.b_x_k_t = b_x_k_t

        # calculate qx_lc
        logger.info("calculating qx_lc = exp(a_x + b_x * k_t)")
        qx_log_lc = a_x.values + b_x_k_t.values
        qx_log_lc = pd.DataFrame(
            qx_log_lc, index=crude_pivot.index, columns=crude_pivot.columns
        )
        qx_lc = np.exp(qx_log_lc)

        # adding predictions to lc_df
        logger.info("adding qx_lc to lc_df")
        lc_df = pd.merge(
            lc_df,
            qx_lc.reset_index().melt(
                id_vars=self.year_col, var_name=self.age_col, value_name="qx_lc"
            ),
            on=[self.year_col, self.age_col],
            how="left",
        ).astype({self.age_col: "int32", self.year_col: "int32"})
        self.lc_df = lc_df

        return lc_df

    def forecast(self, years, seed=None):
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
        year_cols = list(range(self.k_t.index[-1] + 1, self.k_t.index[-1] + years + 1))

        logger.info("forecasting qx_lc using deterministic random walk...")
        # average change in k_t
        mu = (self.k_t.iloc[-1] - self.k_t.iloc[0]) / len(self.k_t)

        # random walk
        rng = np.random.default_rng(seed=seed)
        k_t_i = (
            self.k_t.iloc[-1]
            + mu * np.arange(1, years + 1)
            + rng.normal(scale=variance, size=years)
        )
        self.k_t_i = k_t_i

        # qx_lc forecast
        b_x_k_t_i = pd.DataFrame(np.outer(self.b_x, k_t_i))
        b_x_k_t_i = b_x_k_t_i.transpose()
        qx_log_lc = self.a_x.values + b_x_k_t_i.values
        qx_lc = np.exp(qx_log_lc)

        # dataframe with forecast
        lcf_df = pd.DataFrame(
            qx_lc,
            index=year_cols,
            columns=self.b_x_k_t.columns,
        )
        lcf_df.index.name = self.year_col
        lcf_df = lcf_df.reset_index().melt(
            id_vars=self.year_col, var_name=self.age_col, value_name="qx_lc"
        )

        return lcf_df

    def map(self, df, age_col=None, year_col=None):
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
        age_col="attained_age",
        year_col="observation_year",
        actual_col="death_claim_amount",
        expose_col="amount_exposed",
    ):
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

        """
        logger.info("initialized CBD")
        self.age_col = age_col
        self.year_col = year_col
        self.actual_col = actual_col
        self.expose_col = expose_col
        # calculations
        self.age_diff = None
        self.ages = None
        self.k_t_1 = None
        self.k_t_2 = None
        self.cbd_df = None
        # forecast
        self.k_1_f = None
        self.k_2_f = None

    def structure_df(
        self,
        df,
    ):
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
            cbd_df[self.actual_col] == 0,
            0,
            cbd_df[self.actual_col] / cbd_df[self.expose_col],
        )
        logger.info(
            f"there were {len(cbd_df[cbd_df['qx_raw']>1])} rates "
            f"over 1 that were capped."
        )
        cbd_df["qx_raw"] = cbd_df["qx_raw"].clip(upper=1)
        self.cbd_df = cbd_df
        logger.info(f"cbd_df shape: {self.cbd_df.shape}")

        return self.cbd_df

    def fit(self, cbd_df):
        """
        Get the forecasted mortality rates.

        Parameters
        ----------
        cbd_df : pd.DataFrame
            A DataFrame containing crude mortality rates for a given population.

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

        year_start = crude_pivot.index.min()
        year_end = crude_pivot.index.max()
        ages = crude_pivot.columns
        self.ages = ages
        age_start = int(ages.min())
        age_end = int(ages.max())
        age_mean = ages.to_series().mean()

        logger.info(f"age range: {age_start}, {age_end}")
        logger.info(f"average age: {age_mean}")
        logger.info(f"year range: {year_start}, {year_end}")

        # qx_logit is the mortality matrix
        logger.debug("calculating qx_logit")
        qx_logit = self._logit(crude_pivot)

        # k_t_1 is the age effect (average mortality rate by age)
        logger.debug("calculating k_t_1 = mean rate per year")
        k_t_1 = qx_logit.mean(axis=1)
        self.k_t_1 = k_t_1

        # k_t_2 is the slope component
        logger.debug(
            "calculating k_t_2 = e1 / e2 \n"
            "e1 = Σ((age - age_mean) * qx_logit) \n"
            "e2 = Σ((age - age_mean)^2)"
        )
        age_diff = ages - age_mean
        self.age_diff = age_diff
        e1 = (age_diff * qx_logit).sum(axis=1)
        e2 = (age_diff.values**2).sum()
        k_t_2 = e1 / e2
        self.k_t_2 = k_t_2

        # qx_logit
        logger.debug("calculating qx_logit_cbd = k_t_1 + (age - age_mean) * k_t_2")
        qx_logit_cbd = k_t_1.values[:, np.newaxis] + (
            age_diff.values * k_t_2.values[:, np.newaxis]
        )
        qx_logit_cbd = pd.DataFrame(
            qx_logit_cbd, index=qx_logit.index, columns=qx_logit.columns
        )

        # qx_cbd
        logger.debug("calculating qx_cbd = exp(qx_logit_cbd) / (1 + exp(qx_logit_cbd))")
        qx_cbd = np.exp(qx_logit_cbd) / (1 + np.exp(qx_logit_cbd))

        # adding predictions to cbd_df
        logger.info("adding qx_cbd to cbd_df")
        cbd_df = pd.merge(
            cbd_df,
            qx_cbd.reset_index().melt(
                id_vars=self.year_col, var_name=self.age_col, value_name="qx_cbd"
            ),
            on=[self.year_col, self.age_col],
            how="left",
        ).astype({self.age_col: "int32", self.year_col: "int32"})
        self.cbd_df = cbd_df

        return cbd_df

    def forecast(self, years, seed=None):
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
        year_cols = list(
            range(self.k_t_1.index[-1] + 1, self.k_t_1.index[-1] + years + 1)
        )

        logger.info("forecasting qx_cbd using deterministic random walk...")
        # average change in k_t_1 and k_t_2
        mu = [
            (self.k_t_1.iloc[-1] - self.k_t_1.iloc[0]) / len(self.k_t_1),
            (self.k_t_2.iloc[-1] - self.k_t_2.iloc[0]) / len(self.k_t_2),
        ]

        # random walk
        rng = np.random.default_rng(seed=seed)
        k_1_f = (
            self.k_t_1.iloc[-1]
            + mu[0] * np.arange(1, years + 1)
            + rng.normal(scale=variance, size=years)
        )
        k_1_f = pd.Series(data=k_1_f, index=year_cols)
        k_1_f.index.name = self.year_col
        k_2_f = (
            self.k_t_2.iloc[-1]
            + mu[1] * np.arange(1, years + 1)
            + rng.normal(scale=variance, size=years)
        )
        k_2_f = pd.Series(data=k_2_f, index=year_cols)
        k_2_f.index.name = self.year_col
        self.k_1_f = k_1_f
        self.k_2_f = k_2_f

        # qx_logit
        logger.debug("calculating qx_logit_cbd = k_t_1 + (age - age_mean) * k_t_2")
        qx_logit_cbd = k_1_f.values[:, np.newaxis] + (
            self.age_diff.values * k_2_f.values[:, np.newaxis]
        )
        qx_logit_cbd = pd.DataFrame(qx_logit_cbd, index=year_cols, columns=self.ages)

        # qx_cbd
        logger.debug("calculating qx_cbd = exp(qx_logit_cbd) / (1 + exp(qx_logit_cbd))")
        qx_cbd = np.exp(qx_logit_cbd) / (1 + np.exp(qx_logit_cbd))

        # dataframe with forecast
        cbdf_df = pd.DataFrame(
            qx_cbd,
            index=year_cols,
            columns=self.ages,
        )
        cbdf_df.index.name = self.year_col
        cbdf_df = cbdf_df.reset_index().melt(
            id_vars=self.year_col, var_name=self.age_col, value_name="qx_cbd"
        )

        return cbdf_df

    def map(self, df, age_col=None, year_col=None):
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

    def _logit(self, a):
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


def calc_likelihood_ratio(full_model, reduced_model):
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
