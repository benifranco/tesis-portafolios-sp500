# scripts/metrics.py

import numpy as np
import pandas as pd

from inputs.experiment_config import FREQ


def portfolio_returns(weights, tickers, returns):
    """
    Build a portfolio return series from weights and asset returns.

    Parameters
    ----------
    weights : array-like
        Portfolio weights for each ticker.
    tickers : list-like
        Ticker names corresponding to the weights.
    returns : DataFrame
        Daily returns for all assets (columns are tickers).

    Returns
    -------
    Series of portfolio daily returns.
    """
    w = pd.Series(weights, index=tickers)

    # Keep only assets with complete data for the period
    df = returns[tickers].dropna(axis=1, how="any")

    # Align weights with remaining columns and re-normalize
    w = w.loc[df.columns]
    w = w / w.sum()

    return df.dot(w)


def evaluate(ret, freq=FREQ):
    """
    Compute simple performance statistics for a daily return series.

    Metrics:
    - total return over the period
    - annualized Sharpe ratio (rf = 0)
    - variance of daily returns
    - maximum drawdown

    Returns
    -------
    dict with keys: "return", "sharpe", "variance", "max_dd20".
    """
    if ret.empty:
        return {
            "return": np.nan,
            "sharpe": np.nan,
            "variance": np.nan,
            "max_dd20": np.nan,
        }

    total_return = (1.0 + ret).prod() - 1.0
    mu = ret.mean()
    sigma = ret.std()
    sharpe = (mu / sigma) * np.sqrt(freq) if sigma > 0 else np.nan

    wealth = (1.0 + ret).cumprod()
    drawdown = (wealth / wealth.cummax() - 1.0).min()

    return {
        "return": total_return,
        "sharpe": sharpe,
        "variance": ret.var(),
        "max_dd20": drawdown,
    }
