# main.py

from inputs.experiment_config import DATA_PATH, OUTPUT_DIR, CSV_SUBDIR
from scripts.data_loading import download_close_prices, load_close_prices
from scripts.experiment import run_experiment, save_experiment_results


def main():
    if not DATA_PATH.exists():
        print("Data file not found at {}. Downloading...".format(DATA_PATH))
        download_close_prices()

    close_prices = load_close_prices()

    print("Running experiment...")
    df_all, spectral_results = run_experiment(close_prices)
    save_experiment_results(df_all, spectral_results)

    print("Done.")
    print("Full portfolio results saved in:", OUTPUT_DIR / CSV_SUBDIR / "all_portfolio.csv")
    print("Spectral results saved in:", OUTPUT_DIR / CSV_SUBDIR)


if __name__ == "__main__":
    main()
# This is the main entry point for the experiment script.