# scripts/spectral_sensitivity.py

"""
Sensitivity analysis for spectral clustering random seed.

This module runs a reduced experiment where:
- The number of communities is fixed (k from SENSITIVITY_CONFIG["k"], e.g. k=10),
- The correlation threshold is fixed (theta from SENSITIVITY_CONFIG["theta"], e.g. 0),
- Only MaxSharpe-MaxSharpe portfolios are considered,
- The spectral clustering random_state is varied over a list of seeds.

For each (year, seed, scenario), the script:
1. Builds the correlation network and runs spectral clustering with that seed.
2. Selects one representative per community via a MaxSharpe rule.
3. Builds a MaxSharpe portfolio with the selected assets.
4. Evaluates the portfolio (Sharpe, variance, total return, max drawdown).
5. Computes ARI between this clustering and the baseline clustering with seed=0.

Results are stored in:
    output/csv/spectral_seed_sensitivity.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from inputs.experiment_config import EXPERIMENT_CONFIG, EXPERIMENT_PARAMS, SENSITIVITY_CONFIG
from scripts.data_loading import load_close_prices
from scripts.correlations import exponential_smoothing_correlation, binarize_corr
from scripts.graph_clustering import graph_from_adj, spectral_communities
from scripts.optimization import max_sharpe_portfolio
from scripts.metrics import portfolio_returns, evaluate


def _align_labels(nodes_ref, labels_ref, nodes_seed, labels_seed):
    """
    Align cluster labels for ARI computation.

    Parameters
    ----------
    nodes_ref : list
        Node order from the baseline clustering (seed=0).
    labels_ref : array-like
        Cluster labels corresponding to nodes_ref.
    nodes_seed : list
        Node order from the current clustering (other seed).
    labels_seed : array-like
        Cluster labels corresponding to nodes_seed.

    Returns
    -------
    labels_ref_aligned : list
    labels_seed_aligned : list
        Labels for the intersection of nodes, in the same order,
        ready to be passed to adjusted_rand_score.
    """
    dict_ref = {n: lbl for n, lbl in zip(nodes_ref, labels_ref)}
    dict_seed = {n: lbl for n, lbl in zip(nodes_seed, labels_seed)}

    common_nodes = sorted(set(dict_ref.keys()) & set(dict_seed.keys()))
    labels_ref_aligned = [dict_ref[n] for n in common_nodes]
    labels_seed_aligned = [dict_seed[n] for n in common_nodes]

    return labels_ref_aligned, labels_seed_aligned


def _select_representatives_maxsharpe(returns, rf, freq, communities):
    """
    Select one representative per community using a MaxSharpe rule.

    Parameters
    ----------
    returns : DataFrame
        Daily return series (rows = dates, columns = tickers).
    rf : float
        Annual risk-free rate.
    freq : int
        Number of trading days per year.
    communities : dict
        Mapping cluster_label -> list of tickers in that community.

    Returns
    -------
    list of str
        List of selected tickers (one per community) using MaxSharpe.
    """
    rf_daily = rf / freq
    selected = []

    for grp in communities.values():
        tickers = [t for t in grp if t in returns.columns]
        if not tickers:
            continue

        mu = returns[tickers].mean()
        sigma = returns[tickers].std().replace(0, np.nan)

        sharpe_local = ((mu - rf_daily) / sigma) * np.sqrt(freq)
        sharpe_local = sharpe_local.replace([np.inf, -np.inf], np.nan)

        if sharpe_local.dropna().empty:
            continue

        best = sharpe_local.idxmax()
        selected.append(best)

    return selected


def run_spectral_sensitivity(close_prices):
    """
    Run the spectral clustering seed sensitivity experiment.

    This function:
    - Restricts to years defined in SENSITIVITY_CONFIG["years"],
    - Uses a fixed theta and k from SENSITIVITY_CONFIG,
    - Loops over a set of seeds from SENSITIVITY_CONFIG["seeds"],
    - For each (year, seed, scenario) builds a MaxSharpe-MaxSharpe portfolio,
    - Computes ARI vs the baseline seed=0 clustering,
    - Stores all results into a CSV file.
    """
    config = EXPERIMENT_CONFIG
    params = EXPERIMENT_PARAMS
    sens = SENSITIVITY_CONFIG

    years = sens["years"]
    theta = sens["theta"]
    k = sens["k"]
    seeds = sens["seeds"]

    rf = config["rf"]
    freq = config["freq"]
    scenarios = config["scenarios"]

    min_days = params.get("min_days_per_year", 125)
    min_assets = params.get("min_assets_per_year", 2)

    records = []

    for year in tqdm(years, desc="Sensitivity years"):
        # Filter prices for this year and keep only assets with full data
        prices_y = close_prices[close_prices.index.year == year].dropna(axis=1, how="any")

        if len(prices_y) < min_days or prices_y.shape[1] < min_assets:
            continue

        returns_y = prices_y.pct_change().iloc[1:]

        # Correlation and network
        corr = exponential_smoothing_correlation(
            returns_y,
            window=params["window_corr"],
            tau=params["tau_corr"],
        )
        adj = binarize_corr(corr, theta)
        G = graph_from_adj(adj)

        # Baseline clustering with seed=0
        communities_ref, nodes_ref, labels_ref = spectral_communities(
            G,
            k,
            return_labels=True,
            random_state=0,
        )

        for seed in seeds:
            # Clustering with current seed
            communities, nodes_seed, labels_seed = spectral_communities(
                G,
                k,
                return_labels=True,
                random_state=seed,
            )

            # ARI vs baseline clustering
            y_ref, y_seed = _align_labels(nodes_ref, labels_ref, nodes_seed, labels_seed)
            ari = adjusted_rand_score(y_ref, y_seed)

            # Representative selection (MaxSharpe per community)
            selected = _select_representatives_maxsharpe(returns_y, rf, freq, communities)
            if len(selected) < 2:
                continue

            for scen in scenarios:
                code = scen["code"]
                no_short = scen["no_short"]

                # Build MaxSharpe portfolio on the selected assets
                try:
                    w, var_p, sr_p = max_sharpe_portfolio(
                        returns_y[selected],
                        rf=rf,
                        freq=freq,
                        lambda_reg=params["lambda_reg"],
                        no_short=no_short,
                        warn_negative=no_short,
                    )
                except Exception as e:
                    # If optimization fails, skip this configuration
                    print("Warning: optimization failed for year {}, seed {}, scenario {}: {}".format(
                        year, seed, code, e
                    ))
                    continue

                ret_series = portfolio_returns(w, selected, returns_y)
                mets = evaluate(ret_series)

                records.append({
                    "year": year,
                    "theta": theta,
                    "k": k,
                    "seed": seed,
                    "scenario": code,
                    "n_selected_assets": len(selected),
                    "ari_vs_seed0": ari,
                    "sharpe": mets.get("sharpe", np.nan),
                    "variance": mets.get("variance", np.nan),
                    "total_return": mets.get("return", np.nan),
                    "max_dd20": mets.get("max_dd20", np.nan),
                })

    df = pd.DataFrame(records)

    output_dir = config["output_dir"]
    csv_subdir = config["csv_subdir"]
    csv_path = output_dir / csv_subdir / "spectral_seed_sensitivity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    print("Sensitivity results saved to:", csv_path)

    return df


def main():
    """
    Entry point to run the sensitivity analysis as a script:

        python -m scripts.spectral_sensitivity
    """
    data_path = EXPERIMENT_CONFIG["data_path"]
    close_prices = load_close_prices(data_path)
    run_spectral_sensitivity(close_prices)


if __name__ == "__main__":
    main()
