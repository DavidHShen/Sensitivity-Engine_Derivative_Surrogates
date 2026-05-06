# Sensitivity-Engine Derivative Surrogates

This repository contains the empirical Python implementation accompanying the paper **Sensitivity-Engine Derivative Surrogates: Greek-Regularized and Boundary-Aware Supervision**.

## Overview

The repository implements a controlled surrogate-learning study for European call options on a fixed maturity-moneyness grid. The empirical design uses public S\&P 500, Treasury-yield, and implied-volatility proxy data to construct daily market states, then generates synthetic option labels from a Black-Scholes-Merton teacher.

Three surrogate specifications are compared:

- **Model A**: price-only multilayer perceptron
- **Model B**: Greek-regularized multi-output multilayer perceptron
- **Model C**: boundary-aware Greek-regularized multi-output multilayer perceptron

The code builds daily states, constructs the teacher panel, trains the three models, evaluates global and local pricing/Greek errors, and runs a five-day local hedging backtest.

## Empirical design

### Input files

The script expects the following CSV files either in the same folder as `Sensitivity-Engine_Derivative_Surrogates.py` or in a directory supplied through `--data-dir`:

- `spx_stooq.csv`
- `DGS3MO.csv`
- `DGS2.csv`
- `DGS10.csv`
- `VIXCLS.csv`
- `VXVCLS.csv`

### Data sources

The required local CSV inputs are sourced from publicly available Stooq and FRED endpoints:

- `spx_stooq.csv`: Stooq S\&P 500 historical daily data, `https://stooq.com/q/d/l/?s=%5Espx&i=d` or `https://stooq.com/q/d/?s=%5Espx`
- `DGS3MO.csv`: FRED 3-Month Treasury yield,   `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO`
- `DGS2.csv`: FRED 2-Year Treasury yield,    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2`
- `DGS10.csv`: FRED 10-Year Treasury yield,    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10`
- `VIXCLS.csv`: FRED CBOE VIX index,    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS`
- `VXVCLS.csv`: FRED CBOE 3-Month Volatility Index,    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXVCLS`

### Fixed grid

The teacher panel is generated on the Cartesian product of:

- maturities in calendar days: `3, 5, 7, 14, 21, 30, 45, 60, 90`
- log-moneyness values: `-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15`

### Boundary region

The local boundary region is defined by

- $\tau \le 30/365$
- $|k| \le 0.05$

### Teacher construction

The teacher is a Black-Scholes-Merton call engine with dividend yield fixed at $q = 0.0$. The maturity-specific teacher volatility is constructed as

$$
\sigma_{\mathrm{teacher}}(\tau)=\alpha(\tau)\ \mathrm{IV}_{\tau}+\bigl(1-\alpha(\tau)\bigr)\mathrm{RV20}
$$

with

- $\alpha(\tau) = 0.8$ for $\tau\le 30/365$
- $\alpha(\tau) = 0.6$ otherwise.

### Targets used in training

- Model A is trained on prices only.
- Models B and C are trained on

$$
y=\bigl(\mathrm{price},\ \sqrt{\lambda_{\Delta}}\ \Delta,\ \sqrt{\lambda_{\Gamma}}\ \mathrm{asinh}(S^{2}\Gamma)\bigr).
$$

The transformed Gamma target stabilizes the second-order channel while preserving an invertible mapping back to ordinary Gamma.

## Default parameter settings

The default command-line parameters are:

- `max_iter = 120`
- `hidden_dim = 64`
- `depth = 2`
- `learning_rate = 1e-3`
- `weight_decay = 1e-5`
- `batch_size = 512`
- `patience = 10`
- `lambda_delta = 1.0`
- `lambda_gamma = 5.0`
- `lambda_boundary = 4.0`
- `boundary_h = 0.02`
- `fd_k_step = 1e-3`
- `seed = 1234`

Model B and Model C use separate validation-selection grids, with Model C receiving the broader boundary-aware search.

## Train / validation / test split

The real-data split is calendar based:

- training dates through `2017-12-31`
- validation dates through `2021-12-31`
- test dates afterward

If that split leaves an empty bucket, the script falls back to a 60/20/20 split over unique dates.

## Outputs

The script writes:

- cleaned daily market states
- full teacher panel
- test-set prediction files for Models A, B, and C
- CSV summary tables for sample counts, global metrics, local-boundary metrics, and hedging metrics
- serialized model artifacts and a manifest file

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run from the repository folder after placing the required CSV files next to `Sensitivity-Engine_Derivative_Surrogates.py`:

```bash
python Sensitivity-Engine_Derivative_Surrogates.py
```

Or specify an explicit data directory:

```bash
python Sensitivity-Engine_Derivative_Surrogates.py --data-dir /path/to/csv_folder
```

A mock-data dry run is also available:

```bash
python Sensitivity-Engine_Derivative_Surrogates.py --mock-data
```

## Repository layout

```text
Sensitivity-Engine_Derivative_Surrogates.py
requirements.txt
.gitignore
README.md
```

## Reference

David Hongkai Shen, *Sensitivity-Engine Derivative Surrogates: Greek-Regularized and Boundary-Aware Supervision*.
