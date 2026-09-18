#!/usr/bin/env python
"""Regenerate the paper's fixed-input retrieval comparisons."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "phase0_tf_confirm_top_len158_extra"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproduced")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke test; omit this flag for the paper protocol.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    artifact = args.artifact_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not (artifact / "prepare_meta.json").is_file():
        run([sys.executable, "scripts/unpack_paper_inputs.py", "--output", str(artifact)])

    baseline = output / "retrieval_baseline"
    bootstrap = output / "retrieval_bootstrap"
    common = output / "common_protocol"
    max_clean = "100" if args.quick else "5000"
    common_clean = "100" if args.quick else "1000"
    draws = "200" if args.quick else "20000"
    minhash = "16" if args.quick else "128"

    baseline_command = [
            sys.executable,
            "scripts/audit_retrieval_baselines.py",
            "--artifact-roots",
            str(artifact),
            "--out-dir",
            str(baseline),
            "--max-clean",
            max_clean,
        ]
    if args.quick:
        baseline_command.extend(["--tasks", "bbbp"])
    run(baseline_command)
    run(
        [
            sys.executable,
            "scripts/bootstrap_retrieval_sota.py",
            "--claim-matrices",
            str(baseline / "retrieval_claim_matrix.csv"),
            "--out-dir",
            str(bootstrap),
            "--n-bootstrap",
            draws,
        ]
    )
    common_command = [
            sys.executable,
            "scripts/audit_modern_sota_baselines.py",
            "--artifact-roots",
            str(artifact),
            "--out-dir",
            str(common),
            "--max-clean",
            common_clean,
            "--repeat-seeds",
            "101",
            "103",
            "107",
            "109",
            "113",
            "--minhash-size",
            minhash,
        ]
    if args.quick:
        common_command.extend(["--tasks", "bbbp"])
    run(common_command)
    print(f"Retrieval outputs written to {output}")


if __name__ == "__main__":
    main()
