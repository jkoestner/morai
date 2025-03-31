"""Collection of exploratory data analysis (eda) tools."""

import itertools
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def correlation(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    numeric: bool = True,
    method: str = "pearson",
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create a correlation matrix for the DataFrame.

    Correlation describes the strength of a relationship between two variables. The
    method used to calculate the correlation is different for different data types.
      - num to num: use "pearson", "kendall", or "spearman"
      - cat to cat: use "cramers_v"

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    features : list, optional
        The features to use for the plot. Default is to use all features.
    numeric : bool, optional
        Whether to use only numeric features or categorical. Default is True.
    method : str, optional
        The method to use for the correlation. Default is "pearson".
        The 3 correlation metrics are "pearson", "kendall", and "spearman"
           - pearson: best for linear relationships and normally distributed data.
           - kendall: best for small datasets and ordinal data, focusing
                      on the order of data points.
           - spearman: best for monotonic relationships and ranked/ordinal data.
    **kwargs : dict
        Additional keyword arguments to pass to corr.

    Returns
    -------
    corr : pd.DataFrame
        The correlation matrix plot.

    """
    df = df.copy()
    if features is None:
        features = df.columns

    # get the feature data types
    cat_features = df[features].select_dtypes(exclude=[np.number]).columns.to_list()
    num_features = df[features].select_dtypes(include=[np.number]).columns.to_list()

    # create the correlation matrix for numeric or categorical features
    if numeric:
        logger.info(
            f"Creating correlation matrix for numeric features "
            f"using method: '{method}'."
        )
        corr = pd.DataFrame(index=num_features, columns=num_features)
        corr = df[num_features].corr(method=method, **kwargs)

    else:
        logger.info(
            "Creating correlation matrix for categorical features using cramers_v. "
        )
        corr = pd.DataFrame(index=cat_features, columns=cat_features)
        for col1 in cat_features:
            for col2 in cat_features:
                if col1 == col2:
                    corr.loc[col1, col2] = 1.0
                else:
                    confusion_matrix = pd.crosstab(df[col1], df[col2])
                    corr.loc[col1, col2] = cramers_v(confusion_matrix)

    return corr


def mutual_info(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_jobs: Optional[int] = None,
    display: bool = True,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate mutual information.

    The values range from 0 to 1, where 0 means no association and 1 means
    strong. There are times where the mutual info can be greater than 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    features : list, optional
        List of features to calculate the mutual information. If None, all
        features are used.
    n_jobs : int, optional
        Number of parallel jobs to run. If None, the computation is sequential.
        n_jobs=-1 means using all processors.
    display : bool, optional
        Whether to display the figure or not.
    threshold : float, optional
        The threshold to use for highlighting in the mutual information plot.
        Default is 0.5.

    Returns
    -------
    mi_matrix : pd.DataFrame
        DataFrame with the mutual information values.

    References
    ----------
    - https://en.wikipedia.org/wiki/Mutual_information

    """
    df = df.copy()
    if features is None:
        features = df.columns
    df = df[features]

    # label encode the categorical variables
    for col in df.select_dtypes(exclude=["number"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # calculate the mutual information for each feature
    num_features = len(features)
    combinations = num_features * (num_features - 1) // 2
    logger.info(
        f"Calculating the mutual information for '{num_features}' "
        f"columns using 'regression' that will result in "
        f"'{combinations}' combinations."
    )

    def compute_mi(
        col_i: str, col_j: str, df: pd.DataFrame
    ) -> Tuple[str, str, np.ndarray]:
        logger.debug(f"Calculating the mutual information for '{col_i}' and '{col_j}'.")
        mi_value = mutual_info_regression(
            np.copy(df[col_i].values).reshape(-1, 1), np.copy(df[col_j])
        )
        return (col_i, col_j, mi_value)

    # parallelize computation if n_jobs provided
    if n_jobs is not None:
        if n_jobs == -1:
            n_jobs = cpu_count()
        logger.info(f"Using up to '{n_jobs}' parallel jobs for the computation.")
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_mi)(col_i, col_j, df)
            for col_i, col_j in tqdm(
                list(itertools.combinations(features, 2)),
                desc="Processing",
                unit="combo",
            )
        )
    else:
        logger.info("Using sequential computation for the mutual information.")
        results = [
            compute_mi(col_i, col_j, df)
            for col_i, col_j in tqdm(
                list(itertools.combinations(features, 2)),
                desc="Processing",
                unit="combo",
            )
        ]

    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col_i, col_j, mi_value in results:
        mi_matrix.loc[col_j, col_i] = mi_value

    return mi_matrix


def cramers_v(confusion_matrix: Union[pd.DataFrame, np.ndarray]) -> float:
    """
    Calculate Cramer's V for categorical-categorical association.

    Values range from 0 to 1, where 0 means no association and 1 means strong.

    Parameters
    ----------
    confusion_matrix : pd.DataFrame
        The confusion matrix for the categorical variables.

    Returns
    -------
    cramers_v : float
        The Cramer's V value.

    References
    ----------
    - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    """
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
    return cramers_v


def gvif(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    constant_col: str = "constant",
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    Calculate the generalized variance inflation factor (GVIF) for the features.

    A GVIF > 5 or GVIF > 10 are common thresholds for multicollinearity. The formula
    calculates the VIF for each feature using a linear regression model when the feature
    is numeric.

    VIF = 1 / (1 - R^2)
    GVIF = VIF ** (1 / (2 * degrees_of_freedom))

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    features : list, optional
        The features to use for the VIF calculation. Default is to use all features.
    constant_col : str, optional
        The constant column name to use for the VIF calculation. Default is "constant".
        VIF needs to be calculated with an intercept due to using a linear
        regression model.
    numeric_only : bool, optional
        Whether to use only numeric features or all. Default is True.

    Returns
    -------
    gvif : pd.DataFrame
        The GVIF values for the features.

    References
    ----------
    - https://en.wikipedia.org/wiki/Variance_inflation_factor
    - Fox, John. 2016. Applied Regression Analysis and Generalized Linear Models
    - https://github.com/cran/car/blob/master/R/vif.R

    """
    df = df.copy()
    if features is None:
        features = df.columns
    else:
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in the DataFrame.")
    df = df[features]
    # add constant column if not present
    if constant_col not in df.columns:
        logger.debug(f"Adding a constant column '{constant_col}'.")
        df[constant_col] = 1
        features.append(constant_col)
    if constant_col not in features:
        features.append(constant_col)

    # get the features to use
    features_num = df[features].select_dtypes(include=[np.number]).columns
    features_cat = df[features].select_dtypes(exclude=[np.number]).columns
    if numeric_only:
        if len(features_cat) > 0:
            logger.warning(
                f"Found '{len(features_cat)}' categorical features that will "
                f"be ignored."
            )
        df = df[features_num]
        features = features_num
    else:
        if len(features_cat) > 0:
            logger.info(
                f"Converting '{len(features_cat)}' categorical features to "
                f"dummy variables."
            )
        df = pd.get_dummies(df, columns=features_cat, dtype="int8", drop_first=True)

    # check nas
    if df.isnull().sum().sum() > 0:
        raise ValueError("The DataFrame contains missing values.")

    # calculate gvif for each feature
    logger.info(f"Calculating the GVIF for '{len(features)-1}' features.")
    gvif_data = []
    features_exclude_constant = [f for f in features if f not in constant_col]
    for feature in features_exclude_constant:
        logger.debug(f"Calculating GVIF for '{feature}'")

        if feature in features_num:
            X = df.loc[:, df.columns != feature]
            y = df[feature]

            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            gvif_value = 1 / (1 - r2)

        elif feature in features_cat:
            pattern = re.compile(rf"^{feature}_")
            dummy_features = [col for col in df.columns if pattern.match(col)]
            other_columns = [
                col for col in df.columns if col not in [*dummy_features, constant_col]
            ]

            def make_positive_semi_definite(
                matrix: np.ndarray, epsilon: float = 1e-10
            ) -> np.ndarray:
                return matrix + np.eye(matrix.shape[0]) * epsilon

            A = make_positive_semi_definite(df[dummy_features].corr().to_numpy())
            B = make_positive_semi_definite(df[other_columns].corr().to_numpy())
            C = make_positive_semi_definite(
                df[dummy_features + other_columns].corr().to_numpy()
            )

            det_A = np.linalg.det(A)
            det_B = np.linalg.det(B)
            det_C = np.linalg.det(C)

            degrees_of_freedom = len(dummy_features)
            gvif_unadjust = (det_A * det_B) / det_C
            gvif_value = gvif_unadjust ** (1 / (2 * degrees_of_freedom))
        else:
            raise ValueError(f"Feature '{feature}' not found in the DataFrame.")

        gvif_data.append({"feature": feature, "gvif": gvif_value})

    gvif = pd.DataFrame(gvif_data)

    return gvif
