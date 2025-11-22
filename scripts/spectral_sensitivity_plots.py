# scripts/spectral_sensitivity_plots.py

"""
Plotting utilities for the spectral clustering seed sensitivity analysis.

This module reads:
    output/csv/spectral_seed_sensitivity.csv

and produces several figures that help visualize:
- The stability of spectral clustering w.r.t. the random seed (ARI),
- The variability of portfolio Sharpe ratio across seeds,
- The relation between clustering stability (ARI) and performance (Sharpe),
- The time series of Sharpe ratio across years, for different seeds and k,
  with seed=0 highlighted and (optionally) the full-universe All-MaxSharpe
  portfolio overlaid for comparison.

It also reads:
    output/csv/all_portfolio.csv

to produce time series of MinVar vs MaxSharpe portfolios on the full universe.

Figures are saved under:
    output/images/sensitivity/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_sensitivity_results(csv_path=None):
    """
    Load the seed sensitivity results from CSV.

    Parameters
    ----------
    csv_path : str or Path or None
        Path to the 'spectral_seed_sensitivity.csv' file.
        If None, it is assumed to be:
            output/csv/spectral_seed_sensitivity.csv

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns such as:
        year, theta, k, seed, scenario, ari_vs_seed0, sharpe, variance, ...
    """
    if csv_path is None:
        csv_path = Path("output") / "csv" / "spectral_seed_sensitivity.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError("Sensitivity CSV not found at: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    return df


def _ensure_output_dir():
    """
    Ensure that the sensitivity image directory exists.

    Returns
    -------
    out_dir : Path
        Path to 'output/images/sensitivity'.
    """
    out_dir = Path("output") / "images" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------
# ARI-related plots
# ---------------------------------------------------------------------
def plot_ari_heatmaps_by_k(df):
    """
    For each k, create a heatmap of ARI (year x seed).

    This shows how stable the clustering is with respect to the random seed.
    Years on the y-axis are subsampled so the labels do not overlap.
    """
    out_dir = _ensure_output_dir()

    ks = sorted(df["k"].unique())
    for k in ks:
        df_k = df[df["k"] == k]

        # Pivot to year x seed matrix
        pivot = df_k.pivot_table(
            index="year", columns="seed", values="ari_vs_seed0", aggfunc="mean"
        ).sort_index()

        years = pivot.index.to_numpy()
        seeds = pivot.columns.tolist()
        n_years = len(years)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, label="ARI vs seed=0")

        # x-axis: all seeds, rotated
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_xticklabels(seeds, rotation=90)

        # y-axis: subsample years to avoid overlapping labels
        max_labels = 12
        step = max(1, n_years // max_labels)
        y_idx = np.arange(0, n_years, step)
        ax.set_yticks(y_idx)
        ax.set_yticklabels(years[y_idx])

        ax.set_xlabel("Seed")
        ax.set_ylabel("Year")
        ax.set_title("ARI heatmap (k = {})".format(k))

        fname = out_dir / "ari_heatmap_k{}.png".format(k)
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print("Saved:", fname)


def plot_ari_mean_by_year(df):
    """
    Plot ARI mean vs year, with one curve per k.

    This summarizes the stability of the clustering across seeds for each k.
    """
    out_dir = _ensure_output_dir()

    # Compute mean ARI per (year, k)
    grp = df.groupby(["year", "k"])["ari_vs_seed0"].mean().reset_index()
    ks = sorted(grp["k"].unique())

    plt.figure(figsize=(8, 5))
    for k in ks:
        sub = grp[grp["k"] == k]
        plt.plot(sub["year"], sub["ari_vs_seed0"], marker="o", label="k={}".format(k))

    plt.xlabel("Year")
    plt.ylabel("Mean ARI vs seed=0")
    plt.title("Mean ARI by year for different k")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    fname = out_dir / "ari_mean_by_year_k_all.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)


# ---------------------------------------------------------------------
# Sharpe distribution / variability / ARI relation
# ---------------------------------------------------------------------
def plot_sharpe_boxplots_by_year(df, scenario_code="short"):
    """
    For each k, create boxplots of Sharpe vs year (for a given scenario).

    This shows how much the portfolio Sharpe ratio varies with the seed.
    """
    out_dir = _ensure_output_dir()

    df_s = df[df["scenario"] == scenario_code]
    ks = sorted(df_s["k"].unique())

    for k in ks:
        sub = df_s[df_s["k"] == k]

        years = sorted(sub["year"].unique())
        data = []
        for y in years:
            vals = sub[sub["year"] == y]["sharpe"].dropna().values
            data.append(vals)

        plt.figure(figsize=(12, 5))

        # Color scheme for boxplots
        boxprops = dict(facecolor="#a6cee3", edgecolor="#1f78b4", linewidth=1.5)
        medianprops = dict(color="#ff7f00", linewidth=2)
        whiskerprops = dict(color="#1f78b4", linewidth=1.2)
        capprops = dict(color="#1f78b4", linewidth=1.2)
        flierprops = dict(
            marker="o",
            markerfacecolor="#fb9a99",
            markeredgecolor="none",
            markersize=4,
            alpha=0.6,
        )

        plt.boxplot(
            data,
            patch_artist=True,
            showfliers=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=flierprops,
        )

        # Rotated year labels to avoid overlap
        plt.xticks(
            np.arange(1, len(years) + 1),
            years,
            rotation=45,
            ha="right",
        )

        plt.xlabel("Year")
        plt.ylabel("Sharpe ratio (annualized)")
        plt.title(
            "Sharpe distribution across seeds (k = {}, scenario = {})".format(
                k, scenario_code
            )
        )
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        fname = out_dir / "sharpe_boxplots_by_year_k{}_{}.png".format(k, scenario_code)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


def plot_sharpe_std_by_year(df, scenario_code="short"):
    """
    Plot the standard deviation of Sharpe across seeds vs year, with one curve per k.

    This is a compact way to show how sensitive performance is to the seed.
    """
    out_dir = _ensure_output_dir()

    df_s = df[df["scenario"] == scenario_code]
    grp = df_s.groupby(["year", "k"])["sharpe"].std().reset_index()
    ks = sorted(grp["k"].unique())

    plt.figure(figsize=(8, 5))
    for k in ks:
        sub = grp[grp["k"] == k]
        plt.plot(sub["year"], sub["sharpe"], marker="o", label="k={}".format(k))

    plt.xlabel("Year")
    plt.ylabel("Std of Sharpe across seeds")
    plt.title("Sharpe variability vs year (scenario = {})".format(scenario_code))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    fname = out_dir / "sharpe_std_by_year_k_all_{}.png".format(scenario_code)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)


def plot_sharpe_vs_ari(df, scenario_code="short"):
    """
    Scatter plot of Sharpe vs ARI, colored by k.

    Each point corresponds to a (year, k, seed) configuration. This helps
    visualize whether lower stability (lower ARI) is associated with worse
    portfolio performance (lower Sharpe).
    """
    out_dir = _ensure_output_dir()

    df_s = df[df["scenario"] == scenario_code].copy()
    ks = sorted(df_s["k"].unique())

    plt.figure(figsize=(8, 6))
    for k in ks:
        sub = df_s[df_s["k"] == k]
        plt.scatter(
            sub["ari_vs_seed0"],
            sub["sharpe"],
            label="k={}".format(k),
            alpha=0.7,
        )

    plt.xlabel("ARI vs seed=0")
    plt.ylabel("Sharpe ratio (annualized)")
    plt.title("Sharpe vs ARI (scenario = {})".format(scenario_code))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    fname = out_dir / "sharpe_vs_ari_scatter_{}.png".format(scenario_code)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)


# ---------------------------------------------------------------------
# Sharpe time series across seeds, with optional full portfolio overlay
# ---------------------------------------------------------------------
def plot_sharpe_timeseries_by_seed(
    df,
    scenario_code="short",
    baseline_seed=0,
    include_full=True,
    all_portfolio_path=None,
):
    """
    Plot time series of Sharpe ratio by year for each k, across seeds.

    For each k, we:
    - Pivot the data to a matrix [year x seed] of Sharpe ratios.
    - Plot the envelope (min/max across seeds) as a shaded band.
    - Plot all seeds as light, gray "shadow" lines.
    - Highlight the baseline seed (default: 0) as a thick blue line.
    - Optionally overlay the Sharpe ratio of the full-universe All-MaxSharpe
      portfolio (from output/csv/all_portfolio.csv) for visual comparison.

    If include_full is False, the full-universe line is skipped and the
    resulting image filename ends with "_no_full".
    """
    out_dir = _ensure_output_dir()

    # Filter by scenario (e.g., only short-selling allowed or no-short)
    df_s = df[df["scenario"] == scenario_code].copy()

    # Sort unique k values
    ks = sorted(df_s["k"].unique())

    # Default location of full-portfolio CSV, if not provided
    if all_portfolio_path is None:
        all_portfolio_path = Path("output") / "csv" / "all_portfolio.csv"
    all_portfolio_path = Path(all_portfolio_path)

    df_full = None
    if include_full:
        if all_portfolio_path.exists():
            try:
                df_full = pd.read_csv(all_portfolio_path)
            except Exception as e:
                print("Warning: could not read full portfolio CSV:", e)
                df_full = None
        else:
            print(
                "Warning: full portfolio CSV not found at: {}".format(all_portfolio_path)
            )

    for k in ks:
        df_k = df_s[df_s["k"] == k]

        # Build pivot: rows = year, columns = seed, values = Sharpe
        pivot = df_k.pivot_table(
            index="year", columns="seed", values="sharpe", aggfunc="mean"
        ).sort_index()

        years = pivot.index.values
        seeds = pivot.columns.tolist()

        plt.figure(figsize=(10, 5))

        # Envelope (min/max across seeds) as shaded band
        min_sh = pivot.min(axis=1).values
        max_sh = pivot.max(axis=1).values
        plt.fill_between(
            years,
            min_sh,
            max_sh,
            color="lightgray",
            alpha=0.4,
            label="range over seeds",
        )

        # Plot all non-baseline seeds as thin, gray lines
        for seed in seeds:
            if seed == baseline_seed:
                continue
            series = pivot[seed].values
            plt.plot(
                years,
                series,
                color="gray",
                alpha=0.3,
                linewidth=0.8,
            )

        # Plot the baseline seed as a thick blue line
        if baseline_seed in seeds:
            base_series = pivot[baseline_seed].values
            plt.plot(
                years,
                base_series,
                color="tab:blue",
                linewidth=2.5,
                marker="o",
                label="MaxSharpe (seed = {})".format(baseline_seed),
            )

        # Overlay full-universe All-MaxSharpe portfolio, if requested
        if include_full and df_full is not None and "method" in df_full.columns:
            if scenario_code == "no-short":
                suffix = "_NS"
            else:
                suffix = ""

            mask = df_full["method"] == "All-MaxSharpe{}".format(suffix)
            if mask.any():
                full_sharpe_by_year = (
                    df_full.loc[mask].groupby("year")["sharpe"].mean()
                )
                full_series = full_sharpe_by_year.reindex(years)

                plt.plot(
                    years,
                    full_series.values,
                    color="tab:orange",
                    linewidth=2.5,
                    linestyle="--",
                    marker="s",
                    label="Full portfolio (All-MaxSharpe{})".format(suffix),
                )

        plt.xlabel("Year")
        plt.ylabel("Sharpe ratio (annualized)")
        plt.title(
            "Sharpe ratio across seeds (k = {}, scenario = {})".format(
                k, scenario_code
            )
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        # Different suffix for filenames, depending on whether the full portfolio is drawn
        tag = "with_full" if include_full and df_full is not None else "no_full"
        fname = out_dir / "sharpe_timeseries_seeds_k{}_{}_{}.png".format(
            k, scenario_code, tag
        )
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


# ---------------------------------------------------------------------
# Time series for MinVar vs MaxSharpe on the full universe
# ---------------------------------------------------------------------
def plot_all_portfolio_minvar_maxsharpe_timeseries(
    all_portfolio_path=None,
    scenario_code="no-short",
    metric="sharpe",
):
    """
    Plot time series for All-MinVar vs All-MaxSharpe from all_portfolio.csv.

    Parameters
    ----------
    all_portfolio_path : str or Path or None
        Path to 'all_portfolio.csv'. If None, it is assumed to be:
            output/csv/all_portfolio.csv
    scenario_code : {'short', 'no-short'}
        Which scenario to use. For 'short' we look for methods
        All-MinVar and All-MaxSharpe; for 'no-short' we look for
        All-MinVar_NS and All-MaxSharpe_NS.
    metric : str
        Column to plot on the y-axis, typically 'sharpe' or 'variance'.

    The resulting figure compares how MinVar and MaxSharpe evolve over time
    on the full universe of assets for the chosen scenario.
    """
    out_dir = _ensure_output_dir()

    if all_portfolio_path is None:
        all_portfolio_path = Path("output") / "csv" / "all_portfolio.csv"
    all_portfolio_path = Path(all_portfolio_path)

    if not all_portfolio_path.exists():
        print("all_portfolio.csv not found at:", all_portfolio_path)
        return

    df_full = pd.read_csv(all_portfolio_path)

    if metric not in df_full.columns:
        print("Metric '{}' not found in all_portfolio.csv".format(metric))
        return

    if scenario_code == "no-short":
        suffix = "_NS"
    else:
        suffix = ""

    method_min = "All-MinVar{}".format(suffix)
    method_max = "All-MaxSharpe{}".format(suffix)

    mask_min = df_full["method"] == method_min
    mask_max = df_full["method"] == method_max

    if not mask_min.any() or not mask_max.any():
        print("Methods {} or {} not found in all_portfolio.csv".format(
            method_min, method_max
        ))
        return

    # Aggregate by year in case there are multiple rows per year
    min_by_year = df_full.loc[mask_min].groupby("year")[metric].mean()
    max_by_year = df_full.loc[mask_max].groupby("year")[metric].mean()

    # Align indices
    years = sorted(set(min_by_year.index) | set(max_by_year.index))
    min_series = min_by_year.reindex(years)
    max_series = max_by_year.reindex(years)

    plt.figure(figsize=(10, 5))
    plt.plot(
        years,
        min_series.values,
        marker="o",
        linewidth=2,
        label="All-MinVar{}".format(suffix),
    )
    plt.plot(
        years,
        max_series.values,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="All-MaxSharpe{}".format(suffix),
    )

    plt.xlabel("Year")
    plt.ylabel(metric.capitalize())
    plt.title(
        "Full-universe portfolios ({}): MinVar vs MaxSharpe".format(scenario_code)
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    fname = out_dir / "all_portfolio_{}_timeseries_{}.png".format(
        scenario_code, metric
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():
    """
    Entry point to generate all sensitivity plots.

    Usage (from project root):

        python -m scripts.spectral_sensitivity_plots
    """
    df = load_sensitivity_results()

    # ARI-related plots (independent of scenario)
    plot_ari_heatmaps_by_k(df)
    plot_ari_mean_by_year(df)

    # Sharpe-related plots for both scenarios:
    #   - 'short'    : short-selling allowed
    #   - 'no-short' : long-only (the interesting one for the thesis)
    for scenario_code in ["short", "no-short"]:
        plot_sharpe_boxplots_by_year(df, scenario_code=scenario_code)
        plot_sharpe_std_by_year(df, scenario_code=scenario_code)
        plot_sharpe_vs_ari(df, scenario_code=scenario_code)

        # Version WITHOUT the full-universe All-MaxSharpe line
        plot_sharpe_timeseries_by_seed(
            df,
            scenario_code=scenario_code,
            baseline_seed=0,
            include_full=False,
        )

        # Version WITH the full-universe All-MaxSharpe line
        plot_sharpe_timeseries_by_seed(
            df,
            scenario_code=scenario_code,
            baseline_seed=0,
            include_full=True,
        )

    # Time series for full-universe MinVar vs MaxSharpe (both scenarios)
    for scenario_code in ["short", "no-short"]:
        plot_all_portfolio_minvar_maxsharpe_timeseries(
            scenario_code=scenario_code,
            metric="sharpe",  # you can change to 'variance' if you like
        )


if __name__ == "__main__":
    main()
