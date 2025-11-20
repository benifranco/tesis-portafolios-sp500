# scripts/data_loading.py

from pathlib import Path

import gdown
import matplotlib.pyplot as plt
import pandas as pd

from inputs.experiment_config import DATA_PATH, DATA_FILE_ID


def download_close_prices(file_id=None, dest_path=None):
    """
    Download the close prices CSV from Google Drive using gdown.

    Parameters
    ----------
    file_id : str or None
        Google Drive file ID. If None, uses DATA_FILE_ID from experiment_config.
    dest_path : Path or None
        Destination path for the CSV. If None, uses DATA_PATH.
    """
    if file_id is None:
        file_id = DATA_FILE_ID
    if dest_path is None:
        dest_path = DATA_PATH

    url = "https://drive.google.com/uc?id={}".format(file_id)
    gdown.download(url, str(dest_path), quiet=False)


def load_close_prices(path=None):
    """
    Load daily close prices from CSV.

    Parameters
    ----------
    path : Path or None
        Path to the CSV file. If None, uses DATA_PATH.

    Returns
    -------
    DataFrame with Date as index.
    """
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def available_stock_per_year(close_prices):
    """
    Print and plot the number of stocks with complete data per year.

    This function is mainly for exploratory analysis and sanity checks.
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
    plt.xticks(complete_df["year"], rotation=45)
    plt.tight_layout()
    plt.show()
