# scripts/experiment_plots.py

"""
Plotting utilities for the main experiment (no sensitivity analysis).

This module assumes that:
- The main experiment has been run,
- CSV files exist under: OUTPUT_DIR / CSV_SUBDIR
    * all_portfolio.csv
    * spectral_th{theta}_k{k}.csv

It produces figures such as:
- Number of stocks with complete data per year,
- Time series of Sharpe ratio (full vs reduced) for fixed theta and k,
- Time series of variance (full vs reduced),
- Heatmaps of Sharpe and variance ratios as functions of (theta, k),
- Trade-off plots between Sharpe and variance ratios,
- Sharpe / variance vs k at a fixed theta (e.g. theta = 0.0).

Figures are saved under:
    OUTPUT_DIR / IMAGES_SUBDIR / "experiment"
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inputs.experiment_config import (
    OUTPUT_DIR,
    CSV_SUBDIR,
    IMAGES_SUBDIR,
    EXPERIMENT_CONFIG,
    SCENARIOS,
)
from scripts.data_loading import load_close_prices


# ---------------------------------------------------------------------
# Helpers for paths
# ---------------------------------------------------------------------

def _csv_root():
    """
    Return the directory where experiment CSV files are stored.
    """
    root = OUTPUT_DIR / CSV_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _images_root():
    """
    Return the directory where main experiment figures will be stored.
    """
    root = OUTPUT_DIR / IMAGES_SUBDIR / "experiment"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------
# Helper para localizar archivos spectral_th{theta}_k{k}.csv
# manejando el caso 0.0 / -0.0 y en general flotantes raros.
# ---------------------------------------------------------------------

def _spectral_path(theta, k, csv_dir, atol=1e-8):
    """
    Busca el archivo spectral_th*_k{k}.csv cuyo theta (en el nombre)
    está a distancia <= atol del valor dado.

    Devuelve Path o None si no lo encuentra.
    """
    pattern = re.compile(r"^spectral_th(?P<th>-?\d+\.\d+)_k(?P<k>\d+)\.csv$")
    for path in csv_dir.iterdir():
        m = pattern.match(path.name)
        if not m:
            continue
        k_val = int(m.group("k"))
        if k_val != k:
            continue
        th_val = float(m.group("th"))
        if abs(th_val - theta) <= atol:
            return path
    return None


# ---------------------------------------------------------------------
# 1) Data coverage: number of stocks per year
# ---------------------------------------------------------------------

def available_stock_for_year(close_prices, save=True, show=False):
    """
    Plot the number of stocks with complete data per year.

    Parameters
    ----------
    close_prices : DataFrame
        Price matrix with dates as index and tickers as columns.
    save : bool
        If True, save the figure under output/images/experiment/.
    show : bool
        If True, display the figure interactively (plt.show()).
    """
    complete_counts = close_prices.groupby(close_prices.index.year).apply(
        lambda df_year: df_year.notna().all().sum()
    )

    complete_df = complete_counts.reset_index()
    complete_df.columns = ["year", "num_stocks"]

    print(complete_df)

    plt.figure(figsize=(10, 6))
    plt.bar(complete_df["year"], complete_df["num_stocks"])

    plt.title("Number of stocks with complete data per year", pad=15)
    plt.xlabel("Year")
    plt.ylabel("Number of stocks")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(complete_df["year"], rotation=45, ha="right")
    plt.tight_layout()

    if save:
        img_dir = _images_root()
        fname = img_dir / "stocks_per_year.png"
        plt.savefig(fname, dpi=150)
        print("Saved:", fname)

    if show:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------
# 2) Time series: Sharpe (full vs reduced) para theta fijo
#     -> versión casi idéntica a tu script original
# ---------------------------------------------------------------------

def plot_sharpe_timeseries_full_vs_reduced(theta=0.0, ks=None):
    """
    Plot annual Sharpe ratio (full vs reduced) for several k values,
    at a fixed correlation threshold theta.

    One figure per scenario (short / no-short),
    with 1 row and len(ks) columns (subplots), shared y-axis.
    """
    if ks is None:
        ks = [10, 50, 100]

    csv_dir = _csv_root()
    df_all = pd.read_csv(csv_dir / "all_portfolio.csv")
    years = sorted(df_all["year"].unique())
    theta_str = "{:.1f}".format(theta)
    img_dir = _images_root()

    # Para cada escenario, replicamos la lógica de tu código original
    for scenario in SCENARIOS:
        scenario_label = scenario["name"]
        suf = scenario["suffix"]
        code = scenario["code"]

        fig, axes = plt.subplots(1, len(ks), figsize=(18, 4), sharey=True)
        if len(ks) == 1:
            axes = [axes]

        for ax, k in zip(axes, ks):
            # Serie Sharpe del portafolio completo
            df_full = df_all[df_all["method"] == f"All-MaxSharpe{suf}"]
            sharpe_full = df_full.set_index("year").reindex(years)["sharpe"]

            # Serie Sharpe del portafolio reducido
            path_spec = _spectral_path(theta, k, csv_dir)
            if path_spec is None:
                print(
                    f"Warning: spectral file not found for theta={theta_str}, k={k}"
                )
                continue

            df_spec = pd.read_csv(path_spec)
            df_red = df_spec[df_spec["method"] == f"MaxSharpe-MaxSharpe{suf}"]
            sharpe_red = df_red.set_index("year").reindex(years)["sharpe"]

            # Plot
            ax.plot(years, sharpe_full, marker="o", label="Full Portfolio")
            ax.plot(years, sharpe_red, marker="s", label="Reduced Portfolio")
            ax.set_title(f"$k={k}$")
            ax.set_xlabel("Year")
            if ax is axes[0]:
                ax.set_ylabel("Sharpe Ratio")
            ax.legend()
            ax.grid(True)

        fig.suptitle(
            f"Annual Sharpe Ratio Comparison (theta={theta_str}, {scenario_label})",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])

        fname = img_dir / f"sharpe_timeseries_theta{theta_str}_{code}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print("Saved:", fname)


# ---------------------------------------------------------------------
# 3) Time series: Variance (full vs reduced) para theta fijo
#     -> también prácticamente tu script original
# ---------------------------------------------------------------------

def plot_variance_timeseries_full_vs_reduced(theta=0.0, ks=None):
    """
    Plot annual portfolio variance (full vs reduced) for several k values,
    at a fixed correlation threshold theta.

    One figure per scenario, with len(ks) subplots (one per k).
    """
    if ks is None:
        ks = [10, 50, 100]

    csv_dir = _csv_root()
    df_all = pd.read_csv(csv_dir / "all_portfolio.csv")
    years = sorted(df_all["year"].unique())
    theta_str = "{:.1f}".format(theta)
    img_dir = _images_root()

    for scenario in SCENARIOS:
        scenario_label = scenario["name"]
        suf = scenario["suffix"]
        code = scenario["code"]

        fig, axes = plt.subplots(1, len(ks), figsize=(18, 4), sharey=True)
        if len(ks) == 1:
            axes = [axes]

        for ax, k in zip(axes, ks):
            # Serie de varianza del portafolio completo
            df_full = df_all[df_all["method"] == f"All-MinVar{suf}"]
            var_full = df_full.set_index("year").reindex(years)["variance"]

            # Serie de varianza del portafolio reducido
            spec_file = _spectral_path(theta, k, csv_dir)
            if spec_file is None:
                print(
                    f"Warning: spectral file not found for theta={theta_str}, k={k}"
                )
                continue

            df_spec = pd.read_csv(spec_file)
            df_red = df_spec[df_spec["method"] == f"MinVar-MinVar{suf}"]
            var_red = df_red.set_index("year").reindex(years)["variance"]

            # Plot
            ax.plot(years, var_full, marker="o", label="Full Portfolio")
            ax.plot(years, var_red, marker="s", label="Reduced Portfolio")
            ax.set_title(f"$k={k}$")
            ax.set_xlabel("Year")
            if ax is axes[0]:
                ax.set_ylabel("Variance")
            ax.legend()
            ax.grid(True)

        fig.suptitle(
            f"Annual Variance Comparison (theta={theta_str}, {scenario_label})",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])

        fname = img_dir / f"variance_timeseries_theta{theta_str}_{code}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print("Saved:", fname)


# ---------------------------------------------------------------------
# 4) Ratios Sharpe / Var: build summary DataFrame from spectral CSV
# ---------------------------------------------------------------------

def _build_ratio_dataframe():
    """
    Build a DataFrame with average ratio_sharpe and ratio_var
    per (scenario, theta, k), using the spectral CSV files.

    The spectral CSVs already contain ratio_sharpe and ratio_var.
    We aggregate them across years.
    """
    csv_dir = _csv_root()
    pattern = re.compile(r"^spectral_th(?P<th>-?\d+\.\d+)_k(?P<k>\d+)\.csv$")

    records = []

    for path in csv_dir.iterdir():
        m = pattern.match(path.name)
        if not m:
            continue

        theta = float(m.group("th"))
        k = int(m.group("k"))
        df = pd.read_csv(path)

        for scenario in SCENARIOS:
            suf = scenario["suffix"]
            label = scenario["name"]
            code = scenario["code"]

            # Sharpe_full / Sharpe_reduced para MaxSharpe-MaxSharpe
            mask_sr = df["method"] == f"MaxSharpe-MaxSharpe{suf}"
            ratio_sr = df.loc[mask_sr, "ratio_sharpe"].mean()

            # Var_reduced / Var_full para MinVar-MinVar
            mask_vr = df["method"] == f"MinVar-MinVar{suf}"
            ratio_vr = df.loc[mask_vr, "ratio_var"].mean()

            records.append(
                {
                    "scenario_name": label,
                    "scenario_code": code,
                    "theta": theta,
                    "k": k,
                    "ratio_sharpe": ratio_sr,
                    "ratio_var": ratio_vr,
                }
            )

    if not records:
        raise RuntimeError("No spectral_th*_k*.csv files found to build ratio dataframe")

    df_ratios = pd.DataFrame(records)
    return df_ratios


# ---------------------------------------------------------------------
# 5) Heatmaps de ratios como función de (theta, k)
# ---------------------------------------------------------------------

def plot_ratio_heatmaps():
    """
    Create 2x2 heatmaps:

        - Sharpe_full / Sharpe_reduced for each scenario,
        - Var_reduced / Var_full for each scenario,

    with (rows = k, columns = theta). One 2x2 figure saved to disk.
    """
    df = _build_ratio_dataframe()
    img_dir = _images_root()

    scenarios_labels = df["scenario_name"].unique()
    metrics = [
        ("ratio_sharpe", "Sharpe_full / Sharpe_reduced", "Sharpe sensitivity"),
        ("ratio_var", "Var_reduced / Var_full", "Variance sensitivity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    for i, (metric, cb_label, title_base) in enumerate(metrics):
        for j, scenario_label in enumerate(scenarios_labels):
            ax = axes[i, j]
            grp = df[df["scenario_name"] == scenario_label]

            # Agrupar por (k, theta) por si hubiera duplicados
            grp_clean = grp.groupby(["k", "theta"], as_index=False)[metric].mean()

            pivot = (
                grp_clean.pivot(index="k", columns="theta", values=metric)
                .sort_index()
                .sort_index(axis=1)
            )

            im = ax.imshow(pivot.values, origin="lower", aspect="auto")
            ax.set_title(f"{title_base} ({scenario_label})")
            ax.set_xlabel("Threshold theta")
            if j == 0:
                ax.set_ylabel("Number of communities k")

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(
                [f"{t:.1f}" for t in pivot.columns],
                rotation=90,
                ha="center",
            )
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)

            fig.colorbar(im, ax=ax, label=cb_label)

    fig.tight_layout()
    fname = img_dir / "ratio_heatmaps_theta_k.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("Saved:", fname)


# ---------------------------------------------------------------------
# 6) Sharpe / variance vs k at theta = 0.0
# ---------------------------------------------------------------------

def plot_sharpe_vs_k_at_theta0():
    """
    Plot average Sharpe ratio vs k at theta = 0.0, for each scenario:

        - Line with markers for the reduced portfolios (MaxSharpe-MaxSharpe),
        - Horizontal dashed line for the full portfolio (All-MaxSharpe).
    """
    theta = 0.0
    theta_str = "{:.1f}".format(theta)

    csv_dir = _csv_root()
    img_dir = _images_root()

    all_df = pd.read_csv(csv_dir / "all_portfolio.csv")

    # usar helper para encontrar archivos de este theta
    pattern = re.compile(r"^spectral_th(?P<th>-?\d+\.\d+)_k(?P<k>\d+)\.csv$")
    ks = []
    spectral_files = {}
    for path in csv_dir.iterdir():
        m = pattern.match(path.name)
        if not m:
            continue
        th_val = float(m.group("th"))
        k_val = int(m.group("k"))
        if abs(th_val - theta) <= 1e-8:
            ks.append(k_val)
            spectral_files[k_val] = path
    ks = sorted(set(ks))

    for scenario in SCENARIOS:
        suf = scenario["suffix"]
        label = scenario["name"]
        code = scenario["code"]

        sr_full = all_df.loc[
            all_df["method"] == f"All-MaxSharpe{suf}", "sharpe"
        ].mean()

        sr_reduced = []
        for k in ks:
            spec_df = pd.read_csv(spectral_files[k])
            sr_k = spec_df.loc[
                spec_df["method"] == f"MaxSharpe-MaxSharpe{suf}", "sharpe"
            ].mean()
            sr_reduced.append(sr_k)

        plt.figure(figsize=(8, 4))
        plt.plot(ks, sr_reduced, "-o", label=f"Reduced ({label})")
        plt.hlines(
            sr_full,
            ks[0],
            ks[-1],
            colors="red",
            linestyles="--",
            label=f"Full ({label})",
        )

        plt.xlabel("Number of communities k")
        plt.ylabel("Average Sharpe ratio")
        plt.title(f"Sharpe ratio vs k at theta = {theta_str} — {label}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        fname = img_dir / f"sharpe_vs_k_theta0_{code}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


def plot_variance_vs_k_at_theta0():
    """
    Plot average variance vs k at theta = 0.0, for each scenario:

        - Line with markers for the reduced portfolios (MinVar-MinVar),
        - Horizontal dashed line for the full portfolio (All-MinVar).
    """
    theta = 0.0
    theta_str = "{:.1f}".format(theta)

    csv_dir = _csv_root()
    img_dir = _images_root()

    all_df = pd.read_csv(csv_dir / "all_portfolio.csv")

    pattern = re.compile(r"^spectral_th(?P<th>-?\d+\.\d+)_k(?P<k>\d+)\.csv$")
    ks = []
    spectral_files = {}
    for path in csv_dir.iterdir():
        m = pattern.match(path.name)
        if not m:
            continue
        th_val = float(m.group("th"))
        k_val = int(m.group("k"))
        if abs(th_val - theta) <= 1e-8:
            ks.append(k_val)
            spectral_files[k_val] = path
    ks = sorted(set(ks))

    for scenario in SCENARIOS:
        suf = scenario["suffix"]
        label = scenario["name"]
        code = scenario["code"]

        var_full = all_df.loc[
            all_df["method"] == f"All-MinVar{suf}", "variance"
        ].mean()

        var_reduced = []
        for k in ks:
            spec_df = pd.read_csv(spectral_files[k])
            v_k = spec_df.loc[
                spec_df["method"] == f"MinVar-MinVar{suf}", "variance"
            ].mean()
            var_reduced.append(v_k)

        plt.figure(figsize=(8, 4))
        plt.plot(ks, var_reduced, "-o", label=f"Reduced ({label})")
        plt.hlines(
            var_full,
            ks[0],
            ks[-1],
            colors="red",
            linestyles="--",
            label=f"Full ({label})",
        )

        plt.xlabel("Number of communities k")
        plt.ylabel("Average portfolio variance")
        plt.title(f"Variance vs k at theta = {theta_str} — {label}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        fname = img_dir / f"variance_vs_k_theta0_{code}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


# ---------------------------------------------------------------------
# 7) Trade-off scatter: Sharpe_full/Sharpe_reduced vs Var_reduced/Var_full
# ---------------------------------------------------------------------

def plot_tradeoff_scatter():
    """
    Scatter plot of Sharpe_full/Sharpe_reduced vs Var_reduced/Var_full
    for all (theta, k) and scenarios.

    Points are colored by theta. This helps visualize where the reduced
    portfolios achieve lower variance and similar (or better) Sharpe.
    """
    df = _build_ratio_dataframe()
    img_dir = _images_root()

    for scenario_label in df["scenario_name"].unique():
        grp = df[df["scenario_name"] == scenario_label]

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            grp["ratio_var"],
            grp["ratio_sharpe"],
            c=grp["theta"],
            cmap="viridis",
            s=60,
            edgecolor="k",
        )

        plt.axvline(1.0, color="grey", linestyle="--", linewidth=1)
        plt.axhline(1.0, color="grey", linestyle="--", linewidth=1)

        plt.colorbar(sc, label="Threshold theta")
        plt.title(f"Trade-off: Sharpe vs variance ({scenario_label})")
        plt.xlabel("Variance_reduced / Variance_full")
        plt.ylabel("Sharpe_full / Sharpe_reduced")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        safe_name = scenario_label.replace(" ", "_").lower()
        fname = img_dir / f"tradeoff_sharpe_var_{safe_name}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


# ---------------------------------------------------------------------
# 8) Ratios vs theta (mean ± [min, max])
# ---------------------------------------------------------------------

def plot_ratios_vs_theta():
    """
    For each scenario, plot mean ± [min, max] of:

        - Sharpe_full / Sharpe_reduced as a function of theta,
        - Var_reduced / Var_full as a function of theta.

    This summarizes how performance behaves when the correlation threshold
    becomes more or less restrictive.
    """
    df = _build_ratio_dataframe()
    img_dir = _images_root()

    for scenario_label in df["scenario_name"].unique():
        df_sc = df[df["scenario_name"] == scenario_label]

        # Sharpe ratios
        agg_sr = df_sc.groupby("theta")["ratio_sharpe"].agg(
            ["mean", "min", "max"]
        ).reset_index()
        plt.figure(figsize=(8, 4))
        plt.plot(agg_sr["theta"], agg_sr["mean"], "-o", label="Mean Sharpe ratio")
        plt.fill_between(
            agg_sr["theta"],
            agg_sr["min"],
            agg_sr["max"],
            alpha=0.2,
        )
        plt.title(f"Sharpe ratio vs theta ({scenario_label})")
        plt.xlabel("Threshold theta")
        plt.ylabel("Sharpe_full / Sharpe_reduced")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname_sr = img_dir / f"ratios_sharpe_vs_theta_{scenario_label.replace(' ', '_').lower()}.png"
        plt.savefig(fname_sr, dpi=150)
        plt.close()
        print("Saved:", fname_sr)

        # Variance ratios
        agg_vr = df_sc.groupby("theta")["ratio_var"].agg(
            ["mean", "min", "max"]
        ).reset_index()
        plt.figure(figsize=(8, 4))
        plt.plot(agg_vr["theta"], agg_vr["mean"], "-o", label="Mean variance ratio")
        plt.fill_between(
            agg_vr["theta"],
            agg_vr["min"],
            agg_vr["max"],
            alpha=0.2,
        )
        plt.title(f"Variance ratio vs theta ({scenario_label})")
        plt.xlabel("Threshold theta")
        plt.ylabel("Variance_reduced / Variance_full")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname_vr = img_dir / f"ratios_variance_vs_theta_{scenario_label.replace(' ', '_').lower()}.png"
        plt.savefig(fname_vr, dpi=150)
        plt.close()
        print("Saved:", fname_vr)


# ---------------------------------------------------------------------
# 9) Main entrypoint
# ---------------------------------------------------------------------

def main():
    """
    Entry point to generate the main experiment plots.

    Usage (from project root):

        python -m scripts.experiment_plots
    """
    # 1) Plot data coverage (number of stocks per year)
    data_path = EXPERIMENT_CONFIG["data_path"]
    close_prices = load_close_prices(data_path)
    available_stock_for_year(close_prices, save=True, show=False)

    # 2) Time series for Sharpe and variance (theta = 0.0, k in {10, 50, 100})
    plot_sharpe_timeseries_full_vs_reduced(theta=0.0, ks=[10, 50, 100])
    plot_variance_timeseries_full_vs_reduced(theta=0.0, ks=[10, 50, 100])

    # 3) Global ratio heatmaps
    plot_ratio_heatmaps()

    # 4) Sharpe / variance vs k at theta = 0.0
    plot_sharpe_vs_k_at_theta0()
    plot_variance_vs_k_at_theta0()

    # 5) Trade-off scatter and ratios vs theta
    plot_tradeoff_scatter()
    plot_ratios_vs_theta()


if __name__ == "__main__":
    main()
