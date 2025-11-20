# scripts/plotting.py

from pathlib import Path

import matplotlib.pyplot as plt

from inputs.experiment_config import OUTPUT_DIR, IMAGES_SUBDIR, EXPERIMENT_PARAMS


def _weights_dir_for_scenario(scenario_code):
    """
    Build and create (if needed) the directory where weight histograms
    will be stored for a given scenario.
    """
    base = OUTPUT_DIR / IMAGES_SUBDIR / "hist_pesos" / scenario_code
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_weight_histogram(weights, name, year, scenario_code):
    """
    Save a histogram of portfolio weights for a given year and scenario.

    Parameters
    ----------
    weights : array-like
    name : str
        Label for the portfolio (e.g. "all_minvar").
    year : int
    scenario_code : str
        "short" or "no-short".
    """
    bins = EXPERIMENT_PARAMS.get("hist_bins", 20)
    out_dir = _weights_dir_for_scenario(scenario_code)

    plt.hist(weights, bins=bins, alpha=0.7)
    plt.title("{} weights {} ({})".format(name, year, scenario_code))
    filename = out_dir / "{}_{}.png".format(name, year)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close()