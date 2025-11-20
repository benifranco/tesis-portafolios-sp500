# Portfolio Optimization using Spectral Clustering on Financial Networks

This repository contains the code and experiments developed as part of the master's thesis titled:

**"Portfolio Optimization using Spectral Clustering on Correlation Networks"**

The main goal is to explore a portfolio construction strategy based on community detection in networks formed by asset correlations, and to evaluate its performance against traditional portfolios built on the full set of S&P 500 stocks.

---

## Project Objectives

* Apply unsupervised learning techniques (spectral clustering) to financial correlation networks.
* Select representative subsets of assets from detected communities.
* Build minimum variance and maximum Sharpe portfolios using the selected assets.
* Compare the performance of these reduced portfolios against full portfolios, both in short-selling and no-short-selling scenarios.

---

## Experimental Pipeline

The following table summarizes the empirical methodology implemented in the code:

| **Stage**                                                                                                                         | **Configuration**                                                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Analysis period**                                                                                                               | Years from 1990 to 2023                                                                                                                                                    |
| **Asset selection**                                                                                                               | Only S&P 500 stocks with complete data in each year                                                                                                                        |
| **Returns**                                                                                                                       | Daily simple returns computed from adjusted closing prices                                                                                                                 |
| **Correlation estimation**                                                                                                        | Exponentially weighted average with a window of $$W = 125$$ days and decay parameter $$\tau = 125$$                                                                        |
| **Network binarization**                                                                                                          | Correlation threshold: $$\theta \in {-0.9, -0.8, \dots, 0.9}$$; small artificial edges $$\varepsilon = 10^{-9}$$ added to avoid disconnected graphs in the affinity matrix |
| **Community detection**                                                                                                           | Spectral clustering using `SpectralClustering` with `affinity="precomputed"`; number of communities $$k \in {5, 10, \dots, 100}$$                                          |
| **Representative selection**                                                                                                      | One asset per community, selected by: lowest variance (MinVar), highest Sharpe ratio (MaxSharpe), or highest average return (MaxReturn)                                    |
| **Portfolio construction**                                                                                                        | • With short-selling: analytical Markowitz-type solution                                                                                                                   |
| • Without short-selling: constrained optimization with `scipy.optimize.minimize` (SLSQP), including bounds and weight constraints |                                                                                                                                                                            |
| **Regularization**                                                                                                                | Covariance matrix regularized by diagonal (ridge-type) shrinkage $$\Sigma \mapsto \hat{\Sigma} = \Sigma + \lambda I$$                                                      |
| **Performance evaluation**                                                                                                        | Total return, variance, Sharpe ratio (annualized), and maximum drawdown over the sample                                                                                    |
| **Risk-free rate**                                                                                                                | $$r_f = 5.38%$$, consistent with the 3-month U.S. Treasury rate                                                                                                            |

All these elements are controlled through configuration files so that different grids of $$\theta$$, $$k$$, and time windows can be explored without changing the core logic of the code.

---

## Results and Visualization

For each year, the code generates optimized portfolios based on:

* The full universe of eligible S&P 500 stocks.
* Reduced universes built from community representatives under different selection rules (MinVar, MaxSharpe, MaxReturn).

Each portfolio is evaluated using the metrics described above.

* Numerical results are saved as `.csv` files under:

  ```text
  output/csv/
  ```

* Graphs (time series, heatmaps, boxplots, histograms of weights, etc.) are stored under:

  ```text
  output/images/
  ```

The detailed interpretation of these results is discussed in the Results chapter of the thesis. The repository is designed to be reproducible: any combination of $$\theta$$ and $$k$$ from the grids in the configuration file can be run locally to explore additional scenarios.

---

## Requirements and Installation

Install the necessary dependencies with:

```bash
pip install numpy pandas matplotlib networkx scikit-learn scipy tqdm gdown
```

Alternatively, if you are using the `requirements.txt` file in this project:

```bash
pip install -r requirements.txt
```

Make sure you place the input price data in:

```text
data/close_prices.csv
```

before running the main experiment script.

---

## Running the Experiment

From the project root:

```bash
python main.py
```

The script will:

* Load the S&P 500 price data.
* Run the full grid experiment over years, $$\theta$$ thresholds and number of communities $$k$$.
* Build baseline full portfolios and community-based reduced portfolios.
* Save all numerical results and figures in the `output/` directory.

The main experimental settings (years, grids for $$\theta$$ and $$k$$, risk-free rate, scenarios with/without short-selling, etc.) are defined in:

```text
inputs/experiment_config.py
```

---

### To run this code, run an virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# (Linux/Mac: source .venv/bin/activate)
```