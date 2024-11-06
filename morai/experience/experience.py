"""Experience study model."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def normalize(
    df: pd.DataFrame,
    features: List[str],
    numerator: str,
    denominator: Optional[str] = None,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize a column (numerator) based on a number of features.

    Normalizing over the features is a crude method to adjust data for differences
    in the feature groups. when the features are independent, this method is
    appropriate. When the features are not independent, this method will blend
    the effects.

    Creates a new column with the normalized numerator with the suffix '_norm'.

    Tip:
    -----
    When normalizing using a denominator, you should use the denominator of what
    calculated the rate. For example, if you are normalizing a mortality rate,
    you should use the exposure as the denominator.

    a/o = use exposure as denominator
    a/e = use expected as denominator

    Example:
    --------
    Male = 100 deaths / 1000 exposures = 0.1
    Female = 100 deaths / 500 exposures = 0.2
    Normalization would be:
    Male = 100 / .75 = 133.3, rate = .133
    Female = 50 / 1.5 = 66.6, rate = .133

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    features : list
        List of columns to group by.
    numerator : str
        Column to normalize.
    denominator : str, optional
        Weighting column.
    suffix : str, optional
        Additional suffix for the normalized column if doing multiple normalizations.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional column for normalized values with prefix '_norm'.


    """
    numerator_norm = f"{numerator}_norm"
    if suffix is not None:
        numerator_norm = f"{numerator_norm}_{suffix}"

    # calculate the relative risk
    df = calc_relative_risk(
        df=df, features=features, numerator=numerator, denominator=denominator
    )

    # normalize the numerator
    df[numerator_norm] = df[numerator] / df["risk"]
    df = df.drop(columns=["risk"])

    return df


def calc_relative_risk(
    df: pd.DataFrame,
    features: List[str],
    numerator: str,
    denominator: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate relative risk of a column (numerator) based on a number of features.

    The relative risk is calculated as the average rate for the feature group
    divided by the average rate for all groups. The relative risk is weighted by
    the denominator if provided.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    features : list
        List of columns to group by.
    numerator : str
        Column to normalize.
    denominator : str, optional
        Weighting column.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional column for relative risk 'risk'


    """
    if denominator is None:
        df["temp_one"] = 1
        denominator = "temp_one"

    # calculated the relative risk for each feature group
    grouped_df = (
        df.groupby(features, observed=True)[[numerator, denominator]]
        .sum()
        .reset_index()
    )
    total_ratio = grouped_df[numerator].sum() / grouped_df[denominator].sum()
    grouped_df["risk"] = (grouped_df[numerator] / grouped_df[denominator]) / total_ratio

    # merge risk back to the original dataframe
    df = df.merge(grouped_df[[*features, "risk"]], on=features, how="left")
    if "temp_one" in df.columns:
        df = df.drop(columns=["temp_one"])

    return df


def calc_variance(
    df: pd.DataFrame,
    rate_col: str,
    exposure_col: str,
    amount_col: Optional[str] = None,
) -> pd.Series:
    """
    Calculate the variance of a binomial distribution.

    variance = amount^2 * exposure * rate * (1 - rate)

    Notes
    -----
    Needs to be based on seriatim data and not aggregated data.

    Reference
    ---------
    https://www.soa.org/resources/tables-calcs-tools/table-development/
    page 59

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.
    amount_col : str, optional
        Column name of the face amount.

    Returns
    -------
    variance : pd.Series
        Series with the variance values.

    """
    # check the columns exist
    missing_cols = [col for col in [rate_col, exposure_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )
    amount = 1 if amount_col is None else df[amount_col]

    # calculate the variance
    variance = amount**2 * df[exposure_col] * df[rate_col] * (1 - df[rate_col])

    return variance


def calc_moments(
    df: pd.DataFrame,
    rate_col: str,
    exposure_col: str,
    amount_col: Optional[str] = None,
    sffx: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate the moment variables of a binomial distribution.

    moment_1 = amount * exposure * rate
    moment_2_p1 = amount^2 * exposure * rate
    moment_2_p2 = amount^2 * exposure * rate^2
    moment_3_p1 = amount^3 * exposure * rate
    moment_3_p2 = amount^3 * exposure * rate^2
    moment_3_p3 = amount^3 * exposure * rate^3

    mean = moment_1
    variance = (moment_2_p1 - moment_2_p2)
    skewness = -(moment_3_p1 - 3 * moment_3_p2 + 2 * moment_3_p3) / variance ** 1.5

    Notes
    -----
    Needs to be based on seriatim data and not aggregated data.

    skewnes can also be calculated as:
    skewness = (2 * rate - 1) / (amount^2 * exposure * rate * (1 - rate)) ^ 0.5

    Reference
    ---------
    https://en.wikipedia.org/wiki/Moment_(mathematics)
    https://proofwiki.org/wiki/Skewness_of_Binomial_Distribution

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    rate_col : str
        Column name of the rate.
    exposure_col : str
        Column name of the exposure.
    amount_col : str, optional
        Column name of the face amount.
    sffx : str, optional
        Suffix for the moment columns.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns for variance measures.


    """
    # check the columns exist
    missing_cols = [col for col in [rate_col, exposure_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {', '.join(missing_cols)} in the DataFrame."
        )
    if sffx is None:
        sffx = ""
    else:
        sffx = f"_{sffx}"
        logger.info(f"Adding the label: '{sffx}' to the moment columns.")
    amount = 1 if amount_col is None else df[amount_col]

    # calculate the moments
    logger.info(
        "Calculating moments for the binomial distribution, need to be seriatim data."
    )
    moment_1 = amount * df[exposure_col] * df[rate_col]
    moment_2_p1 = amount**2 * df[exposure_col] * df[rate_col]
    moment_2_p2 = amount**2 * df[exposure_col] * df[rate_col] ** 2
    moment_3_p1 = amount**3 * df[exposure_col] * df[rate_col]
    moment_3_p2 = amount**3 * df[exposure_col] * df[rate_col] ** 2
    moment_3_p3 = amount**3 * df[exposure_col] * df[rate_col] ** 3

    # add the moments to the dataframe
    df[f"moment{sffx}_1"] = moment_1
    df[f"moment{sffx}_2_p1"] = moment_2_p1
    df[f"moment{sffx}_2_p2"] = moment_2_p2
    df[f"moment{sffx}_3_p1"] = moment_3_p1
    df[f"moment{sffx}_3_p2"] = moment_3_p2
    df[f"moment{sffx}_3_p3"] = moment_3_p3

    return df


def calc_qx_exp_ae(
    model_data: pd.DataFrame,
    predictions: pd.Series,
    model_name: str,
    exposure_col: str,
    actual_col: str,
) -> pd.DataFrame:
    """
    Add to the model data the qx, expected amount, and ae.

    Parameters
    ----------
    model_data : pd.DataFrame
        DataFrame with the model data.
    predictions : pd.DataFrame
        DataFrame with the predictions.
    model_name : str
        Name of the model.
    exposure_col : str
        Column name of the exposure.
    actual_col : str
        Column name of the actual values.

    Returns
    -------
    model_data : pd.DataFrame
        DataFrame with additional columns for the model data.

    """
    model_data[f"qx_{model_name}"] = predictions
    model_data[f"exp_amt_{model_name}"] = (
        model_data[f"qx_{model_name}"] * model_data[exposure_col]
    )
    model_data[f"ae_{model_name}"] = np.where(
        model_data[exposure_col] == 0,
        0,
        np.where(
            model_data[f"exp_amt_{model_name}"] == 0,
            1,
            model_data[actual_col] / model_data[f"exp_amt_{model_name}"],
        ),
    )
    return model_data


def calc_whl(
    rates: Union[pd.Series, np.ndarray],
    weights: Union[pd.Series, np.ndarray],
    horizontal_order: int,
    horizontal_lambda: float,
    horizontal_expo: float = 0,
    vertical_order: int = 0,
    vertical_lambda: float = 0,
    vertical_expo: float = 0,
    normalize_weights: bool = True,
) -> np.ndarray:
    """
    Calculate the whittaker-henderson-lowrie (whl) smooth rate.

    The whl smoothes the raw rates by minimizing a difference equation.

    equation:
    q = ∑(w * (y - y_hat)^2) + λ * ∑((Δ^k * y_hat)^2)
    - y = raw rate
    - y_hat = graduated rate
    - w = weight
    - λ = lambda smoothing parameter
    - Δ^k = kth difference operator

    This equation is solved using a matrix equation.

    Caveats
    -------
    - Does not include the standard table Lowrie modification.

    Notes
    -----
    - The function can handle both one-dimensional and two-dimensional data arrays.
    - It includes the exponential rate parameter for the differences which is useful
    for mortality rates and is the Lowrie modification.
    - It is recommended to normalize the weights to sum to the number of data points,
    so that there isn't a bias in the smoothing.

    References
    ----------
    - http://www.howardfamily.ca/graduation/index.html
    - https://arxiv.org/pdf/2306.06932
    - 34 TSA 329: https://www.soa.org/globalassets/assets/library/research/transactions-of-society-of-actuaries/1982/january/tsa82v3414.pdf

    Parameters
    ----------
    rates : pd.Series
        Observed raw data points (1D or 2D array).
    weights : pd.Series
        Weights associated with the raw data (same shape as raw_data)
    horizontal_order : int
        Order of differencing in the horizontal direction (or primary dimension).
    horizontal_lambda : float
        Smoothing parameter for horizontal differences.
    horizontal_expo : float
        Exponential rate parameter for horizontal differences.
    vertical_order : int
        Order of differencing in the vertical direction (0 for 1D data).
    vertical_lambda : float
        Smoothing parameter for vertical differences (0 for 1D data).
    vertical_expo : float
        Exponential rate parameter for vertical differences.
    normalize_weights : bool
        Whether to normalize weights to sum to the number of data points.

    Returns
    -------
    graduated_values : np.ndarray
        Array of smoothed rates

    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve

    rates = np.array(rates, dtype=float)
    weights = np.array(weights, dtype=float)

    # dimensions of the data
    if rates.ndim == 1:
        N = rates.size
        rows = N
        cols = 1
        y = rates.copy()
        weights = weights.copy()
    elif rates.ndim == 2:
        rows, cols = rates.shape
        N = rows * cols
        if rates.shape != weights.shape:
            raise ValueError("rates and weights must have the same shape.")
        y = rates.flatten()
        weights = weights.flatten()
    else:
        raise ValueError("raw_data must be a 1D or 2D array.")

    # normalize weights to number of data points
    if normalize_weights:
        total_weight = np.sum(weights)
        weights = weights / total_weight * N

    # weight matrix
    weight_matrix = diags(weights, 0, shape=(N, N))

    # difference matrix
    def build_difference_matrix(
        size: int, order: int, expo_rate: float = 0, axis: int = 0
    ) -> np.ndarray:
        """
        Build the difference matrix for a given order and exponential rate.

        Parameters
        ----------
        size : int
            The size of the matrix.
        order : int
            The order of differencing.
        expo_rate : float, optional
            The exponential rate parameter (default is 0).
        axis : int, optional
            The axis along which to build the difference matrix (default is 0).

        Returns
        -------
        np.ndarray
            The difference matrix.

        """
        from scipy.special import comb

        if order == 0:
            return None

        # initialize
        data = []
        row_indices = []
        col_indices = []

        # 1 dimension
        if rates.ndim == 1:
            num_diffs = size - order
            indices_list = [np.arange(i, i + order + 1) for i in range(num_diffs)]
        # 2 dimension (horizontal)
        elif axis == 0:
            num_diffs = (cols - order) * rows
            indices_list = []
            for i in range(rows):
                for j in range(cols - order):
                    base_idx = i * cols + j
                    indices = base_idx + np.arange(order + 1)
                    indices_list.append(indices)
        # 2 dimension (vertical)
        else:
            num_diffs = (rows - order) * cols
            indices_list = []
            for i in range(rows - order):
                for j in range(cols):
                    base_idx = i * cols + j
                    indices = base_idx + np.arange(0, (order + 1) * cols, cols)
                    indices_list.append(indices)

        # compute expo factors
        if expo_rate != 0:
            exp_factors = np.exp(expo_rate * np.arange(size))
        else:
            exp_factors = np.ones(size)

        # build the difference matrix
        for idx, indices in enumerate(indices_list):
            coeffs = np.array([(-1) ** k * comb(order, k) for k in range(order + 1)])
            if expo_rate != 0:
                coeffs *= exp_factors[indices]
            row_indices.extend([idx] * (order + 1))
            col_indices.extend(indices)
            data.extend(coeffs)

        diff_matrix = csr_matrix(
            (data, (row_indices, col_indices)), shape=(len(indices_list), size)
        )
        return diff_matrix

    # difference matrices
    diff_matrix_horizontal = build_difference_matrix(
        N, horizontal_order, horizontal_expo, axis=0
    )
    diff_matrix_vertical = None
    if rates.ndim == 2 and vertical_order > 0:
        diff_matrix_vertical = build_difference_matrix(
            N, vertical_order, vertical_expo, axis=1
        )

    # lambda matrices
    lambda_matrix_horizontal = (
        horizontal_lambda * (diff_matrix_horizontal.T @ diff_matrix_horizontal)
        if diff_matrix_horizontal is not None
        else 0
    )
    lambda_matrix_vertical = (
        vertical_lambda * (diff_matrix_vertical.T @ diff_matrix_vertical)
        if diff_matrix_vertical is not None
        else 0
    )

    # solve for y_hat: A * y_hat = b
    A = weight_matrix + lambda_matrix_horizontal + lambda_matrix_vertical
    b = weights * y
    y_hat = spsolve(A.tocsr(), b)

    # reshape
    if rates.ndim == 1:
        graduated_values = y_hat
        # graduated_values = pd.Series(y_hat, name="y_hat")
    else:
        graduated_values = y_hat.reshape(rates.shape)

    return graduated_values
