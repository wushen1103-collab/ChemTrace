#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"

python scripts/phase0_prepare.py --tasks esol bace --corpus-size "${CORPUS_SIZE:-50000}" --affected-per-task 64 --doses 20 --relations exact random
python scripts/launch_phase0.py --tasks esol bace --conditions clean exact_20x random_20x --seeds 13 29 47 --gpus auto --pretrain-epochs "${PRETRAIN_EPOCHS:-4}" --finetune-epochs "${FINETUNE_EPOCHS:-80}" --batch-size "${BATCH_SIZE:-256}"
python scripts/analyze_phase0.py --pred-dir artifacts/phase0/predictions --out-dir artifacts/phase0/tables
