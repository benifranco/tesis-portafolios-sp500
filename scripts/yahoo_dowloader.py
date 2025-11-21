"""This script downloads daily close prices from Yahoo Finance for a list of tickers
specified in a JSON file, and saves them to a CSV file in a standardized format."""

from pathlib import Path
import json
import pandas as pd
import yfinance as yf
from inputs.experiment_config import DATA_PATH, START_DATE, END_DATE


# Path to the JSON file with tickers
TICKERS_JSON = Path("inputs") / "tickers.json"


def load_tickers_from_json(json_path=None):
    """
    Load the list of tickers from a JSON file.

    The JSON file is expected to contain a simple list of strings, e.g.:

        [
          "AAPL",
          "MSFT",
          "GOOGL",
          ...
        ]

    Parameters
    ----------
    json_path : Path or None
        Path to the JSON file. If None, uses TICKERS_JSON.

    Returns
    -------
    list of str
        Ticker symbols.
    """
    if json_path is None:
        json_path = TICKERS_JSON

    if not json_path.exists():
        raise FileNotFoundError(
            "Tickers JSON file not found at {}. "
            "Create it first (e.g. with a helper script) "
            "or add it manually.".format(json_path)
        )

    with json_path.open("r", encoding="utf-8") as f:
        tickers = json.load(f)

    if not isinstance(tickers, list) or not tickers:
        raise ValueError(
            "Tickers JSON must be a non-empty list of strings. "
            "Got: {}".format(type(tickers))
        )

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    return unique_tickers


def download_close_prices_from_yahoo(tickers, start_date, end_date):
    """
    Download daily close prices from Yahoo Finance for a list of tickers,
    always using the 'Close' column.

    Parameters
    ----------
    tickers : list of str or str
        Ticker symbols to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    DataFrame
        Index: Date
        Columns: one column per ticker, with close prices.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,   # prices are adjusted, but we still use "Close"
        progress=False,
        group_by="column",
    )

    # Multiple tickers -> MultiIndex columns (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0)
    else:
        # Single ticker -> flat columns (e.g. 'Open', 'High', 'Close', ...)
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.sort_index()
    close.index.name = "Date"
    close = close.reindex(sorted(close.columns), axis=1)

    return close


def save_close_prices(df, csv_path):
    """
    Save the close prices DataFrame to CSV in the expected format.

    Parameters
    ----------
    df : DataFrame
        Index: Date, Columns: tickers.
    csv_path : Path
        Output CSV path.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)


def rebuild_close_prices_csv_from_yahoo():
    """
    High-level helper:
    1) Load tickers from inputs/tickers.json.
    2) Download fresh data from Yahoo Finance for the configured date range.
    3) Save a new close_prices.csv in the expected format.

    This will overwrite the existing DATA_PATH file.
    """
    print("Reading tickers from JSON:", TICKERS_JSON)
    tickers = load_tickers_from_json(TICKERS_JSON)
    print("Found {} tickers.".format(len(tickers)))

    print("Downloading data from Yahoo Finance...")
    df_close = download_close_prices_from_yahoo(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("Downloaded shape:", df_close.shape)
    print("Saving new close_prices.csv to:", DATA_PATH)
    save_close_prices(df_close, DATA_PATH)
    print("Done.")


if __name__ == "__main__":
    rebuild_close_prices_csv_from_yahoo()
