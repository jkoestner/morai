"""Graduation functions."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import expit, logit

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def whl(
    rates: Union[pd.Series, np.ndarray],
    horizontal_order: int,
    horizontal_lambda: float,
    weights: Union[pd.Series, np.ndarray] = None,
    horizontal_expo: float = 0,
    vertical_order: int = 0,
    vertical_lambda: float = 0,
    vertical_expo: float = 0,
    normalize_weights: bool = True,
    standard_rates: Union[pd.Series, np.ndarray] = None,
    standard_weights: Union[pd.Series, np.ndarray] = None,
    blending_factor: float = 0,
) -> np.ndarray:
    """
    Calculate the whittaker-henderson-lowrie (whl) smooth rate.

    The whl smoothes the raw rates by minimizing a difference equation.

    equation:
    q = (1 - l) ∑(w * (y - y_hat)^2)
        + λ * ∑((Δ^k * y_hat - r * Δ^(k - 1) * y_hat)^2)
        + (l) ∑(w_std * (y_std - y_hat)^2)
    - y = raw rate
    - y_hat = graduated rate
    - w = weight
    - λ = lambda smoothing parameter
    - Δ^k = kth difference operator
    - l = blending factor
    - r = exponential rate parameter
    - w_std = standard table weight
    - y_std = standard table rate

    This equation is solved using a matrix equation.

    Caveats
    -------
    - The standard table Lowrie modification has not been fully tested.

    Notes
    -----
    - The function can handle both one-dimensional and two-dimensional data arrays.
    - It includes the exponential rate parameter for the differences which is useful
    for mortality rates and is the Lowrie modification.
    - It is recommended to normalize the weights to sum to the number of data points,
    so that there isn't a bias in the smoothing.
    - If using standard weights, one method for calculating the weight is to use
    the max(average_expo - expo, 0). This will give higher weight to standard table
    with lower exposure.

    References
    ----------
    - http://www.howardfamily.ca/graduation/index.html
    - https://arxiv.org/pdf/2306.06932
    - 34 TSA 329: https://www.soa.org/globalassets/assets/library/research/transactions-of-society-of-actuaries/1982/january/tsa82v3414.pdf

    Parameters
    ----------
    rates : pd.Series
        Observed raw data points (1D or 2D array).
    horizontal_order : int
        Order of differencing in the horizontal direction (or primary dimension).
    horizontal_lambda : float
        Smoothing parameter for horizontal differences.
    weights : pd.Series
        Weights associated with the raw data (same shape as raw_data)
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
    standard_rates : pd.Series, optional
        Standard rates to blend with the raw rates.
    standard_weights : pd.Series, optional
        Weights for the standard rates.
    blending_factor : float
        Blending factor for the standard rates. The higher the factor, the more
        the standard rates are used.

    Returns
    -------
    graduated_whl : np.ndarray
        Array of smoothed rates

    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve

    rates = np.array(rates, dtype=float)
    if weights is None:
        weights = np.ones_like(rates)
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

    # blending to a standard table
    if blending_factor > 0:
        if standard_rates is None or standard_weights is None:
            raise ValueError(
                "standard_rates and standard_weights must be provided "
                "when blending_factor > 0."
            )

        standard_rates = np.array(standard_rates, dtype=float)
        standard_weights = np.array(standard_weights, dtype=float)

        if rates.shape != standard_rates.shape or rates.shape != standard_weights.shape:
            raise ValueError(
                "standard_rates and standard_weights must have "
                "the same shape as rates."
            )

        if rates.ndim == 1:
            y_std = standard_rates.copy()
            w_std = standard_weights.copy()
        elif rates.ndim == 2:
            y_std = standard_rates.flatten()
            w_std = standard_weights.flatten()

        # normalize standard weights
        if normalize_weights:
            total_std_weight = np.sum(w_std)
            w_std = w_std / total_std_weight * N
    else:
        y_std = np.zeros_like(y)
        w_std = np.zeros_like(weights)

    # weight matrix
    weights_obs_matrix = diags((1 - blending_factor) * weights, 0, shape=(N, N))
    weights_std_matrix = diags(blending_factor * w_std, 0, shape=(N, N))
    weight_matrix = weights_obs_matrix + weights_std_matrix

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
    b = (1 - blending_factor) * weights * y + blending_factor * w_std * y_std
    y_hat = spsolve(A.tocsr(), b)

    # reshape
    if rates.ndim == 1:
        graduated_whl = y_hat
    else:
        graduated_whl = y_hat.reshape(rates.shape)

    return graduated_whl


def exponential(
    rates: np.ndarray,
    weights: np.ndarray,
    ages: np.ndarray,
    expo_low: int,
    expo_high: int,
    calc_ages: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Graduate data by fitting a logistic model using weighted least squares.

    Gompertz provides a general form for the force of mortality (mu_x) or
    mortality rates (q_x). The model is given by: mu_x = B * C^x, where
    mortality is exponentially increasing with age.

    Kannisto proposed a modification to the Gompertz model by including a
    decelaration component.

    Parameters
    ----------
    rates : array_like
        Raw mortality rates (qx) corresponding to the ages.
    weights : array_like
        Weights associated with each age (e.g., exposures or deaths).
    ages : array_like
        Array of ages corresponding to the raw mortality rates.
    expo_low : int
        Lowest exponent in the model (integer).
    expo_high : int
        Highest exponent in the model (integer).
    calc_ages : array_like, optional
        Ages for which to calculate the graduated mortality rates.
        If None, uses the input ages.

    Returns
    -------
    graduated_qx : ndarray
        Graduated mortality rates corresponding to calc_ages.

    Notes
    -----
    - It uses weighted least squares to estimate the parameters.
    - The weights are normalized within the function.

    References
    ----------
    - http://www.howardfamily.ca/graduation/index.html

    """
    # initialize variables
    ages = np.asarray(ages, dtype=float)
    if calc_ages is None:
        calc_ages = ages.copy()
    else:
        calc_ages = np.asarray(calc_ages, dtype=float)
    ages = ages + 0.5
    calc_ages = calc_ages + 0.5
    weights = np.asarray(weights, dtype=float)
    exponents = np.arange(expo_low, expo_high + 1)

    # checks
    if len(ages) != len(rates) or len(ages) != len(weights):
        raise ValueError("Ages, qx_raw, and weights must have the same length.")

    # convert qx to mu_x
    rates = np.asarray(rates, dtype=float)
    mu_x = -np.log(1 - rates)
    y = logit(mu_x)

    # normalize weights
    weights = weights / np.sum(weights) * len(weights)

    # solve for parameters
    ages_matrix = np.column_stack([ages**e for e in exponents])
    calc_ages_matrix = np.column_stack([(calc_ages) ** e for e in exponents])
    weights_matrix = np.diag(weights)
    A = ages_matrix.T @ weights_matrix @ ages_matrix
    B = ages_matrix.T @ weights_matrix @ y
    params = solve(A, B)

    # calculate graduated qx
    graduated_mu = expit(calc_ages_matrix @ params)
    graduated_qx = 1 - np.exp(-graduated_mu)

    return graduated_qx
