"""Archived models for forecasting mortality rates."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import pygam
import rpy2.robjects as ro
import statsmodels.api as sm
from pygam import l, s
from pygam.terms import TermList
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.gam.api import BSplines, GLMGam

from morai.utils import custom_logger
from morai.utils.custom_logger import suppress_logs

logger = custom_logger.setup_logging(__name__)


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


class GAMR(BaseEstimator, RegressorMixin):
    """
    Create a GAM model - R mcgv wrapper.

    The BaseEstimator and RegressorMixin classes are used to interface with
    scikit-learn with certain functions.

    source function:
    https://cran.r-project.org/web/packages/mgcv/

    limitations:
    - requires R

    """

    def __init__(
        self,
    ) -> None:
        """Initialize the model."""
        self.X = None
        self.y = None
        self.weights = None
        self.spline_dict = None
        self.family = None
        self.link = None
        self.unfit_model = None
        self.model = None
        self.is_fitted_ = False

    def setup_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series = None,
        distribution: str = "quasibinomial",
        link: str = "logit",
        spline_dict: Optional[dict] = None,
        extra_text: Optional[str] = "",
        save: bool = True,
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
            notes:
              - having a higher degree of freedom will allow the alpha search to limit
              how much the model can penalize the features. Too few degrees of freedom
              and the model may not model the complexity well.
        extra_text : str, optional
            Extra text to add to the model
        save : bool, optional
            Save the variables in the class

        Returns
        -------
        unfit_model : GAM
            The GAM model

        """
        logger.info(
            f"setup GAM model with mgcv and `{distribution}` distribution "
            f"with `{link}` link"
        )

        # activate pandas2ri and import R packages
        pandas2ri.activate()
        importr("base")
        importr("mgcv")

        # clean up r environment
        ro.r("rm(list=ls())")

        # check shape of inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if weights is not None and X.shape[0] != weights.shape[0]:
            raise ValueError("X and weights must have the same number of rows")

        # initiate variables
        self.spline_dict = spline_dict
        self.family = distribution
        self.link = link

        # setting up model
        X = self._clean_data(X)
        formula = self.get_formula(X=X, y=y, spline_dict=spline_dict)
        ro.globalenv["data"] = pd.concat([X, y], axis=1)
        ro.globalenv["weights"] = weights if weights is not None else ro.r("NULL")
        ro.globalenv["formula"] = ro.r(formula)
        ro.globalenv["family"] = ro.r(f"{distribution}(link={link})")

        # fit the model
        model_text = (
            f"model <- bam(formula, data=data, weights=weights, family=family"
            f", drop.intercept=TRUE{extra_text})"
        )
        logger.info("fitting the model:")
        logger.info(f"> {model_text}`")
        ro.r(model_text)
        logger.info("model fitted")
        self.is_fitted_ = True

        model = ro.globalenv["model"]

        # save the variables
        if save:
            self.X = X
            self.y = y
            self.weights = weights
            self.model = model

        return None

    def get_formula(
        self, X: pd.DataFrame, y: pd.Series, spline_dict: Optional[dict] = None
    ) -> str:
        """
        Get the formula for the GAM model.

        Assumes that all columns are linear except for the columns in the spline_dict.

        Example:
          y ~ s(age) + faceband + gender + constant

        Parameters
        ----------
        X : pd.DataFrame
            The features
        y : pd.Series
            The target
        spline_dict : dict, optional
            The dictionary of the splines to use for the GAM model

        Returns
        -------
        formula : str
            The formula

        """
        var_list = []
        for col in X.columns:
            if spline_dict and col in spline_dict:
                bs_value = (
                    spline_dict[col]["bs"]
                    if spline_dict[col].get("bs") is not None
                    else "ps"
                )
                bs = f", bs='{bs_value}'"
                k = (
                    f", k={spline_dict[col]['df']}"
                    if spline_dict[col].get("df") is not None
                    else ""
                )
                m = (
                    f", m={spline_dict[col]['degree']}"
                    if "degree" in spline_dict[col] and bs_value == "ps"
                    else ""
                )
                sp = (
                    f", sp={spline_dict[col]['alpha']}"
                    if "alpha" in spline_dict[col]
                    else ""
                )

                var_list.append(f"s(`{col}`{k}{m}{sp}{bs})")
            else:
                var_list.append(f"`{col}`")

        formula = f"{y.name} ~ " + "+".join(var_list)
        logger.info("formula:")
        logger.info(f"> {formula}")

        return formula

    def summary(self, expand: bool = False) -> str:
        """
        Print the summary of the model.

        This is used in place of R's summary:
        > ro.r("print(summary(model))")

        Parameters
        ----------
        expand : bool, optional
            Whether to expand the smooth terms

        Returns
        -------
        summary_text : str
            The summary of the model

        """
        if not self.is_fitted_:
            raise ValueError("model is not fitted use fit method")

        if self.model is None:
            raise ValueError("please create a model first")

        # initialize variables
        coefs = None
        smooth = None
        expand_smooth = None

        # get coefficients
        coefs = pandas2ri.rpy2py_dataframe(
            self.ro("as.data.frame(summary(model)$p.table)")
        )
        smooth = pandas2ri.rpy2py_dataframe(
            self.ro("as.data.frame(summary(model)$s.table)")
        )
        # get smooth alpha values
        alpha_vals = []
        num_smooths = int(self.ro("length(model$smooth)")[0])
        sp_vector = np.array(self.ro("model$sp"))
        for i in range(num_smooths):
            sp_i = float(self.ro(f"model$smooth[[{i + 1}]]$sp")[0])
            if sp_i == -1:
                sp_i = sp_vector[i]
            alpha_vals.append(sp_i)
        smooth["alpha"] = alpha_vals

        # populate summary stats
        summary_stats = {
            "family": str(self.ro("model$family$family")[0]),
            "link": str(self.ro("model$family$link")[0]),
            "adj_r_squared": float(self.ro("summary(model)$r.sq")[0]),
            "deviance_explained": float(self.ro("summary(model)$dev.expl")[0]),
            "scale_estimate": float(self.ro("summary(model)$scale")[0]),
            "fREML": float(self.ro("summary(model)$sp.criterion")[0]),
            "n_obs": int(self.ro("length(model$y)")[0]),
            "weights": not np.allclose(np.array(self.ro("model$prior.weights")), 1.0),
        }

        if expand:
            coef_names = list(self.ro("names(model$coefficients)"))
            coef_vals = np.array(self.ro("model$coefficients"))
            expand_smooth = pd.DataFrame(
                {
                    "name": coef_names,
                    "coef": coef_vals,
                }
            )
            expand_smooth = expand_smooth[
                expand_smooth["name"].str.contains("s\\(", regex=True)
            ]

        summary_text = "\n".join(
            [
                "Generalized Additive Model (mgcv) Summary",
                "=" * 70,
                f"Family                : {summary_stats['family']}",
                f"Link                  : {summary_stats['link']}",
                f"Number of Observations: {summary_stats['n_obs']}",
                f"Adjusted R-squared    : {summary_stats['adj_r_squared']:.3f}",
                f"Deviance Explained    : {summary_stats['deviance_explained']:.1%}",
                f"Scale Estimate        : {summary_stats['scale_estimate']:.4g}",
                f"fREML                 : {summary_stats['fREML']:.4g}",
                "",
                "Parametric Coefficients:",
                coefs.to_string(),
                "",
                "Smooth Terms:",
                smooth.to_string(),
                "",
                "Expanded Smooth Terms:",
                expand_smooth.to_string() if expand else "",
            ]
        )

        return summary_text

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

        X = self._clean_data(X)
        ro.globalenv["newdata"] = X
        predictions = ro.r("predict(model, newdata=newdata, type='response')")
        logger.info("predicted rates")

        return predictions

    def ro(self, input: Any) -> ro:
        """
        Return the R object.

        Parameters
        ----------
        input : Any
            The input to the R object

        Returns
        -------
        r_object : ro
            The R object

        """
        r_object = ro.r(input)

        return r_object

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data.

        Parameters
        ----------
        X : pd.DataFrame
            The features

        Returns
        -------
        X : pd.DataFrame
            The cleaned features

        """
        X.columns = [
            col.replace(":", "_")
            .replace(",", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            for col in X.columns
        ]

        return X


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
