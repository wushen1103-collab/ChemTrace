#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.analysis import _label_adjusted_did, _parse_name


COND_RE = re.compile(r"(?P<relation>.+)_(?P<dose>\d+)x$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine Phase-0 prediction artifacts against one clean baseline.")
    p.add_argument("--clean-artifact", type=Path, required=True)
    p.add_argument("--artifact-roots", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _condition_parts(condition: str) -> tuple[str, int]:
    m = COND_RE.match(condition)
    if not m:
        return condition, -1
    return m.group("relation"), int(m.group("dose"))


def _load_clean(clean_artifact: Path) -> dict[tuple[str, int], pd.DataFrame]:
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for path in sorted((clean_artifact / "predictions").glob("*.csv")):
        task, condition, seed = _parse_name(path)
        if condition != "clean":
            continue
        frames[(task, seed)] = pd.read_csv(path)
    if not frames:
        raise RuntimeError(f"No clean prediction files found under {clean_artifact / 'predictions'}")
    return frames


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_frames = _load_clean(args.clean_artifact)

    rows = []
    for artifact in args.artifact_roots:
        for path in sorted((artifact / "predictions").glob("*.csv")):
            task, condition, seed = _parse_name(path)
            if condition == "clean":
                continue
            clean = clean_frames.get((task, seed))
            if clean is None:
                raise RuntimeError(f"Missing clean baseline for task={task} seed={seed}")
            contam = pd.read_csv(path)
            relation, dose = _condition_parts(condition)
            c = clean[clean["split"] == "test"][["sample_id", "sample_loss", "affected", "y", "task_type"]].rename(
                columns={"sample_loss": "loss_clean"}
            )
            z = contam[contam["split"] == "test"][["sample_id", "sample_loss"]].rename(
                columns={"sample_loss": "loss_contam"}
            )
            pair = c.merge(z, on="sample_id", how="inner")
            if len(pair) != len(c):
                raise RuntimeError(f"Sample mismatch for {path}: paired={len(pair)} clean_test={len(c)}")
            pair["delta"] = pair["loss_clean"] - pair["loss_contam"]
            aff = pair[pair["affected"].astype(bool)]
            ctl = pair[~pair["affected"].astype(bool)]
            task_type = str(pair["task_type"].iloc[0])
            label_adj = (
                _label_adjusted_did(pair)
                if task_type == "classification"
                else {"did_label_adjusted": np.nan, "did_label_equal": np.nan, "n_labels_adjusted": 0}
            )
            rows.append(
                {
                    "artifact": artifact.name,
                    "task": task,
                    "condition": condition,
                    "relation": relation,
                    "dose": dose,
                    "seed": seed,
                    "n_affected": int(len(aff)),
                    "n_control": int(len(ctl)),
                    "delta_affected": float(aff["delta"].mean()) if len(aff) else np.nan,
                    "delta_control": float(ctl["delta"].mean()) if len(ctl) else np.nan,
                    "did": float(aff["delta"].mean() - ctl["delta"].mean()) if len(aff) and len(ctl) else np.nan,
                    "affected_win_rate": float((aff["delta"] > 0).mean()) if len(aff) else np.nan,
                    "control_win_rate": float((ctl["delta"] > 0).mean()) if len(ctl) else np.nan,
                    **label_adj,
                }
            )

    by_seed = pd.DataFrame(rows).sort_values(["task", "relation", "dose", "seed"])
    by_seed.to_csv(args.out_dir / "phase0_combined_did_by_seed.csv", index=False)
    summary = (
        by_seed.groupby(["task", "relation", "dose"])
        .agg(
            seeds=("seed", "nunique"),
            did_mean=("did", "mean"),
            did_std=("did", "std"),
            did_label_adjusted=("did_label_adjusted", "mean"),
            did_label_equal=("did_label_equal", "mean"),
            affected_win_rate=("affected_win_rate", "mean"),
        )
        .reset_index()
        .sort_values(["task", "relation", "dose"])
    )
    summary.to_csv(args.out_dir / "phase0_combined_dose_summary.csv", index=False)

    wide = summary.pivot_table(index=["task", "dose"], columns="relation", values="did_label_adjusted", aggfunc="mean").reset_index()
    for relation in ["exact", "random", "parent", "stereo", "tautomer"]:
        if relation in wide.columns and "decoy" in wide.columns:
            wide[f"{relation}_minus_decoy"] = wide[relation] - wide["decoy"]
    wide.to_csv(args.out_dir / "phase0_combined_minus_decoy_wide.csv", index=False)

    long_rows = []
    for _, row in wide.iterrows():
        for relation in ["exact", "random", "parent", "stereo", "tautomer"]:
            col = f"{relation}_minus_decoy"
            if relation in row and col in row and pd.notna(row[col]):
                long_rows.append(
                    {
                        "task": row["task"],
                        "dose": int(row["dose"]),
                        "relation": relation,
                        "did_label_adjusted": row[relation],
                        "decoy_did_label_adjusted": row["decoy"],
                        "minus_decoy": row[col],
                    }
                )
    long = pd.DataFrame(long_rows).sort_values("minus_decoy", ascending=False)
    long.to_csv(args.out_dir / "phase0_combined_minus_decoy_long.csv", index=False)

    aggregate = (
        long.groupby(["relation", "dose"])
        .agg(
            tasks=("task", "nunique"),
            mean_minus_decoy=("minus_decoy", "mean"),
            median_minus_decoy=("minus_decoy", "median"),
            positive_tasks=("minus_decoy", lambda x: int((x > 0).sum())),
            mean_adj_did=("did_label_adjusted", "mean"),
        )
        .reset_index()
        .sort_values("mean_minus_decoy", ascending=False)
    )
    aggregate.to_csv(args.out_dir / "phase0_combined_relation_dose_aggregate.csv", index=False)

    best = long.sort_values(["task", "minus_decoy"], ascending=[True, False]).groupby("task", as_index=False).head(3)
    best.to_csv(args.out_dir / "phase0_combined_top3_relation_by_task.csv", index=False)

    compact = []
    for task, tdf in long.groupby("task"):
        top = tdf.loc[tdf["minus_decoy"].idxmax()]
        compact.append(
            {
                "task": task,
                "best_relation": top["relation"],
                "best_dose": int(top["dose"]),
                "best_minus_decoy": top["minus_decoy"],
                "semantic_did_label_adjusted": top["did_label_adjusted"],
                "decoy_did_label_adjusted": top["decoy_did_label_adjusted"],
                "positive_relation_dose_cells": int((tdf["minus_decoy"] > 0).sum()),
            }
        )
    compact_df = pd.DataFrame(compact).sort_values("best_minus_decoy", ascending=False)
    compact_df.to_csv(args.out_dir / "phase0_combined_compact_task_findings.csv", index=False)

    print(f"wrote combined tables to {args.out_dir}")
    print("\nRelation-dose aggregate:")
    print(aggregate.to_string(index=False))
    print("\nCompact task findings:")
    print(compact_df.to_string(index=False))
    print("\nTop 20 relation effects over decoy:")
    print(long.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
