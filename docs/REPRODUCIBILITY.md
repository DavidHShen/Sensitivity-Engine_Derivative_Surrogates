# Reproducibility

## 1. Functional reproducibility

Run the deterministic mock-data smoke test:

```bash
bash scripts/run_smoke.sh
```

or in PowerShell:

```powershell
.\scripts\run_smoke.ps1
```

This exercises data construction, teacher labels, Models A--D, automatic first- and second-order derivatives, evaluation, aggregation, and paper-facing output generation. It is a software check, not an empirical replication.

## 2. Computational reproducibility with independently obtained data

Place the six documented CSV files in `data/raw/`, validate them, and run:

```bash
python scripts/validate_inputs.py --data-dir data/raw --write-manifest data/input_manifest.csv
python SEDS.py \
  --data-dir data/raw \
  --output-dir outputs/paper_run \
  --model-a-max-epochs 960 \
  --max-epochs 320
```

The script records the command-line arguments, model definitions, seeds, dtypes, execution controls, and output paths in `MANIFEST.json`.

## 3. Numerical reproduction of manuscript tables

Numerical identity requires all of the following:

- the same six input-file contents;
- the same date coverage and point-in-time alignment;
- the same seeds and hyperparameters;
- compatible numerical-library and accelerator behavior; and
- the same stopping outcomes or compatible completed-model caches.

The repository does not claim that a newly downloaded Stooq series is identical to the fixed SPX vintage used in the manuscript. Provider changes can alter realized volatility, teacher volatility, the generated panel, stopping paths, and reported metrics.

## Determinism

The script seeds Python, NumPy, and PyTorch and requests deterministic algorithms. CUDA users should set:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Hardware, drivers, PyTorch builds, and low-level kernels can still affect exact bitwise identity. The primary reproducibility target is agreement at the reported statistical precision, conditional on identical data and compatible execution environments.

## Checkpoint and cache behavior

By default, the script can reuse:

- the processed daily state and teacher panel;
- completed fitted models;
- epoch checkpoints;
- evaluation results; and
- hedging state grids and summaries.

Cache compatibility is checked against model settings, data fingerprints, scalers, and test-frame signatures. Delete the selected output/cache directory or use `--no-resume --no-reuse-processed-cache` for a clean run.
