from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# 1. Base paths
# ---------------------------------------------------------------------

DATA_PATH = Path("data") / "close_prices.csv"
OUTPUT_DIR = Path("output")

# If you want to change the name of subfolders, do it here:
CSV_SUBDIR = Path("csv")
IMAGES_SUBDIR = Path("images")

# ---------------------------------------------------------------------
# 2. Time horizon and data frequency
# ---------------------------------------------------------------------

# Years used in the backtest
YEARS = range(1990, 2024)

# Risk-free rate (annual) and data frequency
RF = 0.0538      # annual risk-free rate
FREQ = 252       # trading days per year

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

    # Covariance matrix regularization
    "regularization_type": "shrinkage",  # or "Ledoit-Wolf", etc.
    "lambda_reg": 1e-3,

    # Optimization options (can be tuned from here)
    "maxiter": 2000,
    "ftol": 1e-9,
}

# ---------------------------------------------------------------------
# 6. Master configuration dictionary
# ---------------------------------------------------------------------

EXPERIMENT_CONFIG = {
    "data_path": DATA_PATH,
    "output_dir": OUTPUT_DIR,
    "csv_subdir": CSV_SUBDIR,
    "images_subdir": IMAGES_SUBDIR,
    "years": YEARS,
    "rf": RF,
    "freq": FREQ,
    "theta_grid": THETA_GRID,
    "k_grid": K_GRID,
    "scenarios": SCENARIOS,
    "params": EXPERIMENT_PARAMS,
}
