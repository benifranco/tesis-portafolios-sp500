# scripts/spectral_sensitivity_plots.py

"""
Plotting utilities for the spectral clustering seed sensitivity analysis.

This module reads:
    output/csv/spectral_seed_sensitivity.csv

and produces several figures that help visualize:
- The stability of spectral clustering w.r.t. the random seed (ARI),
- The variability of portfolio Sharpe ratio across seeds,
- The relation between clustering stability (ARI) and performance (Sharpe),
- The time series of Sharpe ratio across years, for different seeds and k.

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

        years = pivot.index.tolist()
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
        ax.set_yticklabels([years[i] for i in y_idx])

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

        # Nicer color scheme for boxplots
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


def plot_sharpe_timeseries_by_seed(df, scenario_code="short", baseline_seed=0):
    """
    Plot time series of Sharpe ratio by year for each k, across seeds.

    For each k, we:
    - Pivot the data to a matrix [year x seed] of Sharpe ratios.
    - Plot all seeds as light, gray "shadow" lines.
    - Highlight the baseline seed (default: 0) as a thick black line.

    This produces one figure per k.
    """
    out_dir = _ensure_output_dir()

    # Filter by scenario (e.g., only short-selling allowed or no-short)
    df_s = df[df["scenario"] == scenario_code].copy()

    # Sort unique k values
    ks = sorted(df_s["k"].unique())

    for k in ks:
        df_k = df_s[df_s["k"] == k]

        # Build pivot: rows = year, columns = seed, values = Sharpe
        pivot = df_k.pivot_table(
            index="year", columns="seed", values="sharpe", aggfunc="mean"
        ).sort_index()

        years = pivot.index.values
        seeds = pivot.columns.tolist()

        plt.figure(figsize=(8, 5))

        # Plot all non-baseline seeds as thin, gray lines
        for seed in seeds:
            if seed == baseline_seed:
                continue
            series = pivot[seed].values
            plt.plot(
                years,
                series,
                color="gray",
                alpha=0.4,
                linewidth=1.0,
            )

        # Plot the baseline seed as a thick black line
        if baseline_seed in seeds:
            base_series = pivot[baseline_seed].values
            plt.plot(
                years,
                base_series,
                color="black",
                linewidth=2.5,
                marker="o",
                label="seed = {}".format(baseline_seed),
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

        fname = out_dir / "sharpe_timeseries_seeds_k{}_{}.png".format(
            k, scenario_code
        )
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


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
        plot_sharpe_timeseries_by_seed(
            df, scenario_code=scenario_code, baseline_seed=0
        )


if __name__ == "__main__":
    main()
