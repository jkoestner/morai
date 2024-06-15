"""Collection of exploratory data analysis (eda) tools."""

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def cramers_v(confusion_matrix):
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


def correlation(
    df,
    features=None,
    numeric=True,
    method="pearson",
    **kwargs,
):
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
    fig : Figure
        The chart

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


def mutual_info(df, features=None, n_jobs=None, display=True, threshold=0.5):
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

    def compute_mi(col_i, col_j, df):
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
            for i, col_i in enumerate(df.columns)
            for j, col_j in enumerate(df.columns[i + 1 :], start=i + 1)
        )
    else:
        logger.info("Using sequential computation for the mutual information.")
        results = [
            compute_mi(col_i, col_j, df)
            for i, col_i in enumerate(df.columns)
            for j, col_j in enumerate(df.columns[i + 1 :], start=i + 1)
        ]

    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col_i, col_j, mi_value in results:
        mi_matrix.loc[col_j, col_i] = mi_value

    return mi_matrix
