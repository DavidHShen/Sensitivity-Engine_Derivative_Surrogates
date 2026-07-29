#!/usr/bin/env bash
set -euo pipefail

python SEDS.py \
  --mock-data \
  --n-mock-days 160 \
  --seeds 1234 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --max-test-samples 64 \
  --max-epochs 1 \
  --model-a-max-epochs 1 \
  --patience 1 \
  --batch-size 64 \
  --evaluation-batch-size 64 \
  --skip-hedging \
  --teacher-audit-samples 2 \
  --no-save-full-panel \
  --no-resume \
  --no-reuse-processed-cache \
  --output-dir outputs/smoke_test
