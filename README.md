# Sensitivity-Engine Derivative Surrogates

## Greek- and Structure-Consistent Supervision

This repository contains the manuscript and empirical implementation for **Sensitivity-Engine Derivative Surrogates: Greek- and Structure-Consistent Supervision** by David Hongkai Shen.

The study compares four matched neural surrogate designs:

| Model | Construction |
|---|---|
| A | A scalar normalized-price potential, with all reported sensitivities recovered by automatic differentiation. |
| B | Direct heads for normalized price, Delta, normalized spot Gamma, normalized Vega, Vanna, and normalized Volga. |
| C | Model B plus consistency between direct sensitivity heads and derivatives of the normalized-price potential. |
| D | Model C plus a two-leg asset/strike decomposition with supervised probability heads and price reconstruction. |

The implementation includes point-in-time source alignment, analytic Black--Scholes--Merton teacher labels, finite-difference audits, deterministic matched seeds, early stopping, checkpoint resume, paired inference, consistency diagnostics, and five-day hedging diagnostics.

## Repository contents

```text
.
├── SEDS.py
├── Sensitivity_Engine_Derivative_Surrogates.pdf
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── requirements-tested.txt
├── data/
│   ├── README.md
│   ├── input_manifest_template.csv
│   └── raw/
├── docs/
│   ├── DATA_AVAILABILITY.md
│   └── REPRODUCIBILITY.md
├── scripts/
│   ├── run_smoke.ps1
│   ├── run_smoke.sh
│   └── validate_inputs.py
└── .github/workflows/smoke-test.yml
```

## Installation

Python 3.10 or later is recommended.

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU execution, install the PyTorch build appropriate for the local CUDA environment before installing the remaining dependencies.

## Functional smoke test

The mock-data mode verifies the complete four-model computational path without external market data. It is intentionally small and is not the paper experiment.

### PowerShell

```powershell
.\scripts\run_smoke.ps1
```

### Linux or macOS

```bash
bash scripts/run_smoke.sh
```

The smoke test writes disposable outputs to `outputs/smoke_test/`.

## Empirical inputs

Place the following files in `data/raw/`:

```text
spx_stooq.csv
DGS3MO.csv
DGS2.csv
DGS10.csv
VIXCLS.csv
VXVCLS.csv
```

Validate the files before a full run:

```bash
python scripts/validate_inputs.py --data-dir data/raw --write-manifest data/input_manifest.csv
```

The repository does not redistribute the underlying index and macroeconomic series. See [`data/README.md`](data/README.md) and [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Paper-configuration run

The manuscript reports five matched seeds, a 960-epoch safety cap for Model A, a 320-epoch safety cap for Models B--D, and the common early-stopping rule. The corresponding command is:

```bash
python SEDS.py \
  --data-dir data/raw \
  --output-dir outputs/paper_run \
  --model-a-max-epochs 960 \
  --max-epochs 320
```

For deterministic CUDA execution, set the cuBLAS workspace configuration before starting Python.

### PowerShell

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python SEDS.py `
  --data-dir data/raw `
  --output-dir outputs/paper_run `
  --model-a-max-epochs 960 `
  --max-epochs 320
```

### Linux

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python SEDS.py \
  --data-dir data/raw \
  --output-dir outputs/paper_run \
  --model-a-max-epochs 960 \
  --max-epochs 320
```

Completed models, epoch checkpoints, processed panels, evaluations, and hedging outputs are reusable when the command is rerun with compatible inputs and settings.

## Main output groups

A completed run writes:

- aligned daily market states and the teacher panel;
- source-coverage, data-validation, and derivative-audit tables;
- fitted model files and epoch histories;
- per-seed and summary RMSE/MAE tables;
- paired RMSE inference;
- potential- and decomposition-consistency diagnostics;
- five-day hedging diagnostics;
- LaTeX-ready result tables and macros; and
- a machine-readable run manifest.

Generated model files, caches, raw inputs, and large outputs are excluded from Git by default.

## Reproducibility scope

The repository supports three distinct checks:

1. **Functional reproducibility:** run the complete pipeline with deterministic mock data.
2. **Computational reproducibility:** run the empirical pipeline with files satisfying the documented schemas.
3. **Numerical reproduction of reported tables:** requires the same historical data vintage, software environment, settings, and seeds used for the reported run.

The historical Stooq SPX download currently available may differ in symbol, coverage, or values from the fixed vintage used in the manuscript. A fresh download should therefore not be assumed to reproduce the published numerical tables. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Paper

- [Compiled manuscript](Sensitivity_Engine_Derivative_Surrogates.pdf)

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Licensing

The MIT License in [`LICENSE`](LICENSE) applies only to the software code in this repository.

The manuscript, its typeset PDF, and its scholarly text remain copyright © 2026 David Hongkai Shen unless a separate publication license is stated. The MIT License does not grant rights to third-party market or macroeconomic data. No raw third-party data are distributed in this repository.

See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for the data-availability statement.
