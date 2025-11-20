# scripts/correlations.py

import numpy as np


def exponential_smoothing_correlation(returns, window, tau):
    """
    Compute exponentially smoothed correlation matrix over a rolling window.

    Parameters
    ----------
    returns : DataFrame
        Asset returns (rows: time, columns: assets).
    window : int
        Window size (number of observations) used for each local correlation.
    tau : float
        Decay parameter for the exponential weights.

    Returns
    -------
    DataFrame correlation matrix.
    """
    n = len(returns)
    if n < window:
        return returns.corr()

    corr_list = [
        returns.iloc[t : t + window].corr()
        for t in range(n - window + 1)
    ]

    weights = np.exp((np.arange(1, len(corr_list) + 1) - tau) / tau)
    weights = weights / weights.sum()

    corr_smoothed = sum(w * c for w, c in zip(weights, corr_list))
    return corr_smoothed


def binarize_corr(corr, theta):
    """
    Binarize a correlation matrix: 1 if rho >= theta, 0 otherwise.

    Returns a DataFrame with the same index/columns.
    """
    return (corr >= theta).astype(int)
