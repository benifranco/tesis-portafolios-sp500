# inputs/experiment_config.py

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# 1. Base paths
# ---------------------------------------------------------------------

DATA_PATH = Path("data") / "close_prices.csv"
OUTPUT_DIR = Path("output")

# Subfolders inside OUTPUT_DIR
CSV_SUBDIR = Path("csv")
IMAGES_SUBDIR = Path("images")

# Google Drive file ID for the close prices CSV
#DATA_FILE_ID = "1TsOGVI99XbDOf-bE3qwA6cKMOblyxDfX" # Old file

DATA_FILE_ID = "1ZpaVpxRM5WIfUCQrMDFS0zfH-3GLVmLa" # New file with full with 2024 data

# ---------------------------------------------------------------------
# 2. Time horizon and frequency
# ---------------------------------------------------------------------

# Years used in the backtest
YEARS = range(1990, 2025)

# Risk-free rate (annual) and data frequency
RF = 0.0538      # annual risk-free rate
FREQ = 252       # trading days per year

# Minimum data requirements per year
# (used to decide if a year is kept in the experiment)
MIN_DAYS_PER_YEAR = 125
MIN_ASSETS_PER_YEAR = 2

# ---------------------------------------------------------------------
# 3. Parameter grids (theta, k)
# ---------------------------------------------------------------------

# Correlation thresholds used for binarization (theta)
THETA_GRID = np.arange(-0.9, 1.0, 0.1)  # -0.9, -0.8, ..., 0.9

# Number of communities k in spectral clustering
K_GRID = np.arange(5, 101, 5)           # 5, 10, ..., 100

# ---------------------------------------------------------------------
# 4. Trading scenarios
# ---------------------------------------------------------------------
# name     -> label used in tables and plots
# code     -> short identifier (useful for filenames)
# no_short -> True for long-only, False if short-selling is allowed
# suffix   -> suffix used in the "method" field of result tables/CSV

SCENARIOS = [
    {
        "name": "Short-selling",
        "code": "short",
        "no_short": False,
        "suffix": "",
    },
    {
        "name": "No-Short-selling",
        "code": "no-short",
        "no_short": True,
        "suffix": "_NS",
    },
]

# ---------------------------------------------------------------------
# 5. Experiment-level parameters
# ---------------------------------------------------------------------

EXPERIMENT_PARAMS = {
    # Exponentially smoothed correlation
    "window_corr": 125,
    "tau_corr": 125,

    # Minimum sample requirements per year
    "min_days_per_year": MIN_DAYS_PER_YEAR,
    "min_assets_per_year": MIN_ASSETS_PER_YEAR,

    # Covariance matrix regularization
    "regularization_type": "shrinkage",  # or "ridge", "Ledoit-Wolf", etc.
    "lambda_reg": 1e-3,

    # Optimization options for SLSQP
    "ftol": 1e-9,
    "maxiter": 2000,

    # Frontier search for max Sharpe
    "frontier_n_points": 500,

    # Graph / spectral clustering tweaks
    "epsilon_graph": 1e-9,
    "eigen_tol": 1e-4,
    "spectral_random_state": 42,

    # Histograms
    "hist_bins": 20,
}


#----------------------------------------------------------
#Configuration complete only for testing
# ---------------------------------------------------------------------
#  Test configuration (smaller grids, fewer years)

# Years used in the backtest (test mode)
#YEARS = range(2020, 2025)  # 1990, 1991, 1992

# Correlation thresholds (test: only one)
#THETA_GRID = np.array([0.0])

# Number of communities k (test: only one)
#K_GRID = np.array([10])


SENSITIVITY_CONFIG = {
    # Years selected for robustness analysis of spectral clustering
    "years": range(1990, 2024),  # 2020, 2021, 2022
    # Fixed parameters for the sensitivity experiment
    "theta": 0.0,
    "k_list": [5, 10, 15],
    "k": 10,
    # Seeds to explore (spectral clustering random_state)
    "seeds": list(range(20)),  # 0, 1, ..., 19
}


# ---------------------------------------------------------------------
# 6. Master configuration dictionary (optional convenience)
# ---------------------------------------------------------------------

EXPERIMENT_CONFIG = {
    "data_path": DATA_PATH,
    "output_dir": OUTPUT_DIR,
    "csv_subdir": CSV_SUBDIR,
    "images_subdir": IMAGES_SUBDIR,
    "data_file_id": DATA_FILE_ID,
    "years": YEARS,
    "rf": RF,
    "freq": FREQ,
    "theta_grid": THETA_GRID,
    "k_grid": K_GRID,
    "scenarios": SCENARIOS,
    "params": EXPERIMENT_PARAMS,
}


