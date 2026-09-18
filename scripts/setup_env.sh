#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if python3 -m venv .venv; then
  :
elif command -v virtualenv >/dev/null 2>&1; then
  virtualenv .venv
else
  echo "python3-venv is unavailable and virtualenv was not found." >&2
  exit 2
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Keep PyTorch separate so a platform-specific CUDA wheel can be substituted.
python -m pip install -r requirements-train.txt

python - <<'PY'
import importlib.util
mods = ["torch", "rdkit", "pandas", "sklearn", "pyarrow"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing packages after setup: {missing}")
import torch
print("env ok", "torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY
