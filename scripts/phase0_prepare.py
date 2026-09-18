#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.corpus import build_phase0_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ChemTrace phase-0 splits and paired contamination corpora.")
    p.add_argument("--tasks", nargs="+", default=["esol", "bace"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0"))
    p.add_argument(
        "--external-cache",
        type=Path,
        default=Path("data/external"),
        help="Directory containing optional source SMILES files used to assemble the clean corpus.",
    )
    p.add_argument("--corpus-size", type=int, default=50000)
    p.add_argument("--affected-per-task", type=int, default=64)
    p.add_argument("--doses", nargs="+", type=int, default=[20])
    p.add_argument("--relations", nargs="+", default=["exact", "random"])
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--max-affected-len", type=int, default=158)
    p.add_argument("--affected-filter-relation", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_phase0_artifacts(
        tasks=args.tasks,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        external_cache=args.external_cache,
        corpus_size=args.corpus_size,
        affected_per_task=args.affected_per_task,
        doses=args.doses,
        relations=args.relations,
        seed=args.seed,
        max_affected_len=args.max_affected_len,
        affected_filter_relation=args.affected_filter_relation,
    )


if __name__ == "__main__":
    main()
