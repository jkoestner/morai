"""R models for forecasting mortality rates."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator, RegressorMixin

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


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
        self.formula = None
        self.model = None
        self.coefs = None
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
        coefs = pandas2ri.rpy2py_dataframe(
            ro.r("as.data.frame(summary(model)$p.table)")
        )["Estimate"]

        # save the variables
        if save:
            self.X = X
            self.y = y
            self.weights = weights
            self.model = model
            self.coefs = coefs
            self.formula = formula

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
        coefs = self.coefs
        smooth = None
        expand_smooth = None

        # get coefficients
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
                "Formula:",
                self.formula,
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
