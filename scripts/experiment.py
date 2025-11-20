# scripts/experiment.py

import numpy as np
import pandas as pd
from tqdm import tqdm

from inputs.experiment_config import (
    YEARS,
    RF,
    FREQ,
    THETA_GRID,
    K_GRID,
    SCENARIOS,
    EXPERIMENT_PARAMS,
    OUTPUT_DIR,
    CSV_SUBDIR,
)

from scripts.correlations import exponential_smoothing_correlation, binarize_corr
from scripts.graph_clustering import graph_from_adj, spectral_communities
from scripts.optimization import min_var_portfolio, max_sharpe_portfolio
from scripts.metrics import portfolio_returns, evaluate
from scripts.plotting import save_weight_histogram


def run_experiment(close_prices):
    """
    Run the main experiment over all years, theta values, k values and scenarios.

    Parameters
    ----------
    close_prices : DataFrame
        Close prices with Date as index and tickers as columns.

    Returns
    -------
    df_all : DataFrame
        Full-portfolio results (All-MinVar, All-MaxSharpe).
    spectral_results : dict
        Keys: (theta, k)
        Values: list of dicts with spectral portfolio results for each year
                and scenario, including ratio_var and ratio_sharpe.
    """
    min_days = EXPERIMENT_PARAMS.get("min_days_per_year", 125)
    min_assets = EXPERIMENT_PARAMS.get("min_assets_per_year", 2)
    lambda_reg = EXPERIMENT_PARAMS.get("lambda_reg", 1e-3)
    regular_type = EXPERIMENT_PARAMS.get("regularization_type", "shrinkage")
    window_corr = EXPERIMENT_PARAMS.get("window_corr", 125)
    tau_corr = EXPERIMENT_PARAMS.get("tau_corr", 125)

    all_rows = []
    spectral_results = {}

    for year in tqdm(YEARS, desc="Years"):
        price_y = close_prices[close_prices.index.year == year].dropna(
            axis=1, how="any"
        )

        if len(price_y) < min_days or price_y.shape[1] < min_assets:
            continue

        ret_y = price_y.pct_change().iloc[1:]

        # -----------------------------------------------------------------
        # 6-A. Baseline (full universe, All-MinVar / All-MaxSharpe)
        # -----------------------------------------------------------------
        baseline_metrics = {}

        for scenario in SCENARIOS:
            code = scenario["code"]
            no_short = scenario["no_short"]
            suf = scenario["suffix"]

            w_min, _, _ = min_var_portfolio(
                ret_y,
                rf=RF,
                freq=FREQ,
                lambda_reg=lambda_reg,
                regular_type=regular_type,
                no_short=no_short,
                warn_negative=no_short,
            )
            save_weight_histogram(w_min, "all_minvar", year, code)
            eval_min = evaluate(portfolio_returns(w_min, price_y.columns, ret_y))

            w_max, _, _ = max_sharpe_portfolio(
                ret_y,
                rf=RF,
                freq=FREQ,
                lambda_reg=lambda_reg,
                no_short=no_short,
                warn_negative=no_short,
            )
            save_weight_histogram(w_max, "all_maxsharpe", year, code)
            eval_max = evaluate(portfolio_returns(w_max, price_y.columns, ret_y))

            baseline_metrics[(code, "MinVar")] = eval_min
            baseline_metrics[(code, "MaxSharpe")] = eval_max

            all_rows.append(
                {
                    "year": year,
                    "method": "All-MinVar{}".format(suf),
                    **eval_min,
                }
            )
            all_rows.append(
                {
                    "year": year,
                    "method": "All-MaxSharpe{}".format(suf),
                    **eval_max,
                }
            )

        # -----------------------------------------------------------------
        # 6-B. Spectral clustering + community-based selection
        # -----------------------------------------------------------------
        corr = exponential_smoothing_correlation(
            ret_y,
            window=window_corr,
            tau=tau_corr,
        )

        for theta in THETA_GRID:
            G = graph_from_adj(binarize_corr(corr, theta))

            for k in K_GRID:
                comms = spectral_communities(G, k)
                sel_min, sel_max, sel_ret = [], [], []

                for grp in comms.values():
                    tickers_group = [t for t in grp if t in ret_y.columns]
                    if not tickers_group:
                        continue

                    # MinVar representative inside the community
                    var_series = ret_y[tickers_group].var()
                    sel_min.append(var_series.idxmin())

                    # MaxSharpe representative inside the community
                    mu_group = ret_y[tickers_group].mean()
                    sigma_group = ret_y[tickers_group].std().replace(0, np.nan)
                    sharpe_local = ((mu_group - RF / FREQ) / sigma_group) * np.sqrt(FREQ)
                    sel_max.append(sharpe_local.idxmax())

                    # MaxReturn representative inside the community (new)
                    sel_ret.append(mu_group.idxmax())

                for scenario in SCENARIOS:
                    code = scenario["code"]
                    no_short = scenario["no_short"]
                    suf = scenario["suffix"]

                    combos = [
                        ("MinVar-MinVar", sel_min, min_var_portfolio, "MinVar"),
                        ("MinVar-MaxSharpe", sel_min, max_sharpe_portfolio, "MaxSharpe"),
                        ("MaxSharpe-MinVar", sel_max, min_var_portfolio, "MinVar"),
                        ("MaxSharpe-MaxSharpe", sel_max, max_sharpe_portfolio, "MaxSharpe"),
                        ("MaxReturn-MinVar", sel_ret, min_var_portfolio, "MinVar"),
                        ("MaxReturn-MaxSharpe", sel_ret, max_sharpe_portfolio, "MaxSharpe"),
                    ]

                    for label, tickers_sel, builder, baseline_key in combos:
                        if len(tickers_sel) < 2:
                            continue

                        w, var_p, sr_p = builder(
                            ret_y[tickers_sel],
                            rf=RF,
                            freq=FREQ,
                            lambda_reg=lambda_reg,
                            no_short=no_short,
                            warn_negative=no_short,
                        )
                        mets = evaluate(portfolio_returns(w, tickers_sel, ret_y))

                        base_metrics = baseline_metrics.get((code, baseline_key))
                        if base_metrics is None:
                            ratio_var = np.nan
                            ratio_sharpe = np.nan
                        else:
                            ref_var = base_metrics["variance"]
                            ref_sharpe = base_metrics["sharpe"]

                            if ref_var is None or np.isnan(ref_var) or ref_var == 0:
                                ratio_var = np.nan
                            else:
                                ratio_var = mets["variance"] / ref_var

                            if (
                                mets["sharpe"] is None
                                or np.isnan(mets["sharpe"])
                                or mets["sharpe"] == 0
                            ):
                                ratio_sharpe = np.nan
                            else:
                                ratio_sharpe = ref_sharpe / mets["sharpe"]

                        row = {
                            "year": year,
                            "method": "{}{}".format(label, suf),
                            "theta": theta,
                            "k": k,
                            "ratio_var": ratio_var,
                            "ratio_sharpe": ratio_sharpe,
                            "weights": ",".join("{:.4f}".format(x) for x in w),
                        }
                        row.update(mets)

                        spectral_results.setdefault((theta, k), []).append(row)

    df_all = pd.DataFrame(all_rows)
    return df_all, spectral_results


def save_experiment_results(df_all, spectral_results):
    """
    Save the experiment results to CSV files:

    - output/csv/all_portfolio.csv
    - output/csv/spectral_th{theta}_k{k}.csv  for each (theta, k)
    """
    csv_root = OUTPUT_DIR / CSV_SUBDIR
    csv_root.mkdir(parents=True, exist_ok=True)

    # Full portfolios
    df_all.to_csv(csv_root / "all_portfolio.csv", index=False)

    # Spectral portfolios, one file per (theta, k)
    for (theta, k), rows in spectral_results.items():
        if not rows:
            continue
        df_spec = pd.DataFrame(rows)
        theta_str = "{:.1f}".format(theta)  # stable naming, e.g. -0.2, 0.0, 0.5
        filename = csv_root / "spectral_th{}_k{}.csv".format(theta_str, k)
        df_spec.to_csv(filename, index=False)
