#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.analysis import analyze_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ChemTrace phase-0 result tables.")
    p.add_argument("--pred-dir", type=Path, default=Path("artifacts/phase0/predictions"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/phase0/tables"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analyze_predictions(args.pred_dir, args.out_dir)


if __name__ == "__main__":
    main()
