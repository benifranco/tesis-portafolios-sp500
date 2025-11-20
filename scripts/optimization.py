# scripts/optimization.py

import warnings

import numpy as np
from scipy.optimize import minimize

from inputs.experiment_config import EXPERIMENT_PARAMS


# Global options for the SLSQP solver, taken from the experiment config
_SOLVER_OPTS = {
    "ftol": EXPERIMENT_PARAMS.get("ftol", 1e-9),
    "maxiter": EXPERIMENT_PARAMS.get("maxiter", 2000),
}


def _min_var_long_only(cov_matrix):
    """
    Minimum-variance portfolio with constraints:
    - weights >= 0
    - weights <= 0.25 (or 1.0 if n < 4)
    - sum(weights) = 1
    """
    n = cov_matrix.shape[0]
    upper_bound = 0.25 if n >= 4 else 1.0

    def objective(w):
        return float(w.dot(cov_matrix).dot(w))

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, upper_bound)] * n
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SOLVER_OPTS,
    )

    if not result.success:
        msg = "MinVar (long-only, max %.0f%%) failed: %s" % (
            upper_bound * 100,
            result.message,
        )
        raise ValueError(msg)

    return result.x


def _max_sharpe_long_only(mu, cov_matrix, rf_daily):
    """
    Maximum Sharpe portfolio with constraints:
    - weights >= 0
    - weights <= 0.25 (or 1.0 if n < 4)
    - sum(weights) = 1
    """
    n = len(mu)
    upper_bound = 0.25 if n >= 4 else 1.0

    def negative_sharpe(w):
        variance = float(w.dot(cov_matrix).dot(w))
        if variance <= 0:
            return np.inf
        excess_ret = float(w.dot(mu) - rf_daily)
        return -(excess_ret / np.sqrt(variance))

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, upper_bound)] * n
    x0 = np.ones(n) / n

    result = minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SOLVER_OPTS,
    )

    if not result.success:
        msg = "MaxSharpe (long-only, max %.0f%%) failed: %s" % (
            upper_bound * 100,
            result.message,
        )
        raise ValueError(msg)

    return result.x


def min_var_portfolio(
    returns,
    rf=0.0,
    freq=252,
    lambda_reg=1e-3,
    regular_type="ridge",
    no_short=False,
    warn_negative=False,
):
    """
    Compute the minimum-variance portfolio.

    regular_type="shrinkage" ⇒ Σ_hat = Σ + λ·I with λ = λ0 · tr(Σ)/n
    """
    cov = returns.cov().values
    n = cov.shape[0]

    if regular_type == "shrinkage":
        lambda_reg = lambda_reg * np.trace(cov) / n

    cov_reg = cov + lambda_reg * np.eye(n)

    if no_short:
        weights = _min_var_long_only(cov_reg)
    else:
        inv = np.linalg.pinv(cov_reg)
        ones = np.ones(n)
        weights = inv.dot(ones) / (ones.dot(inv).dot(ones))

    if warn_negative and (weights < -1e-8).any():
        warnings.warn(
            "Negative weights detected under a no-short constraint.",
            RuntimeWarning,
        )

    mu = returns.mean().values
    variance = float(weights.dot(cov).dot(weights))
    rf_daily = rf / freq
    ret_p = float(weights.dot(mu))
    if variance > 0:
        sharpe = (ret_p - rf_daily) / np.sqrt(variance) * np.sqrt(freq)
    else:
        sharpe = np.nan

    return weights, variance, sharpe


def frontier_max_sharpe(mu, cov, rf=0.0, freq=252, n_points=None):
    """
    Compute the maximum Sharpe ratio on the Markowitz frontier using a grid
    of target returns.

    Returns
    -------
    best_sr : float
        Maximum Sharpe ratio found.
    best_weights : ndarray
        Portfolio weights achieving best_sr.
    """
    if n_points is None:
        n_points = EXPERIMENT_PARAMS.get("frontier_n_points", 500)

    inv = np.linalg.pinv(cov)
    ones = np.ones_like(mu)

    A = ones.dot(inv).dot(ones)
    B = ones.dot(inv).dot(mu)
    C = mu.dot(inv).dot(mu)
    D = A * C - B * B

    if abs(D) < 1e-12:
        raise ValueError("Efficient frontier is ill-conditioned.")

    rf_daily = rf / freq

    R_min = min(mu.min(), rf_daily)
    R_max = mu.max()

    best_sr = -np.inf
    best_weights = None

    for R in np.linspace(R_min, R_max, n_points):
        top_vec = (C - B * R) * ones + (A * R - B) * mu
        w = inv.dot(top_vec) / D
        var = float(w.dot(cov).dot(w))
        if var <= 0:
            continue
        sr = (R - rf_daily) / np.sqrt(var) * np.sqrt(freq)
        if sr > best_sr:
            best_sr = sr
            best_weights = w

    return best_sr, best_weights


def max_sharpe_portfolio(
    returns,
    rf=0.0,
    freq=252,
    lambda_reg=1e-3,
    no_short=False,
    warn_negative=False,
):
    """
    Compute the maximum Sharpe ratio portfolio.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    cov_reg = cov + lambda_reg * np.eye(len(mu))
    rf_daily = rf / freq

    if no_short:
        weights = _max_sharpe_long_only(mu, cov_reg, rf_daily)
    else:
        inv = np.linalg.pinv(cov_reg)
        top = inv.dot(mu - rf_daily)
        if top.sum() <= 0:
            _, weights = frontier_max_sharpe(mu, cov, rf, freq)
        else:
            weights = top / top.sum()

    if warn_negative and (weights < -1e-8).any():
        warnings.warn(
            "Negative weights detected under a no-short constraint.",
            RuntimeWarning,
        )

    variance = float(weights.dot(cov).dot(weights))
    if variance > 0:
        sharpe = (weights.dot(mu) - rf_daily) / np.sqrt(variance)
        sharpe *= np.sqrt(freq)
    else:
        sharpe = np.nan

    return weights, variance, sharpe
