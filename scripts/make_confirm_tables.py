#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge old and extra Phase-0 confirmation seeds into paper tables.")
    p.add_argument(
        "--old-by-seed",
        type=Path,
        default=Path("reports/tables/phase0_combined_refined_shared_clean/phase0_combined_did_by_seed.csv"),
    )
    p.add_argument(
        "--new-by-seed",
        type=Path,
        default=Path("reports/tables/phase0_confirm_top_shared_clean_extra/phase0_combined_did_by_seed.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_confirm_top_combined7"))
    p.add_argument("--file-prefix", default="phase0_confirm_top_combined7")
    return p.parse_args()


def q25(x: pd.Series) -> float:
    return float(x.quantile(0.25))


def q75(x: pd.Series) -> float:
    return float(x.quantile(0.75))


def support_bucket(row: pd.Series) -> str:
    """Conservative triage labels for table drafting, not significance claims."""
    if (
        row.mean_minus_decoy >= 0.02
        and row.positive_seeds >= 5
        and row.mean_semantic_adj_did >= 0.01
        and row.semantic_positive_seeds >= 4
    ):
        return "replicated_positive"
    if row.mean_minus_decoy >= 0.02 and row.positive_seeds >= 5:
        return "decoy_relative_positive"
    if row.mean_minus_decoy > 0.005 and row.positive_seeds >= 4 and row.mean_semantic_adj_did > 0:
        return "directional_positive"
    if abs(row.mean_minus_decoy) < 0.005:
        return "near_null"
    return "weak_or_negative"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.file_prefix

    old = pd.read_csv(args.old_by_seed)
    new = pd.read_csv(args.new_by_seed)
    old["seed_set"] = "old3"
    new["seed_set"] = "extra4"

    selected_sem = new[new["relation"] != "decoy"][["task", "relation", "dose"]].drop_duplicates()
    selected_dose = selected_sem[["task", "dose"]].drop_duplicates()
    all_rows = pd.concat([old, new], ignore_index=True)

    sem = all_rows[all_rows["relation"] != "decoy"].merge(selected_sem, on=["task", "relation", "dose"], how="inner")
    decoy = all_rows[all_rows["relation"] == "decoy"].merge(selected_dose, on=["task", "dose"], how="inner")
    selected_by_seed = pd.concat([sem, decoy], ignore_index=True).sort_values(["task", "dose", "relation", "seed"])
    selected_by_seed.to_csv(args.out_dir / f"{prefix}_did_by_seed.csv", index=False)

    dec = decoy[
        [
            "task",
            "dose",
            "seed",
            "did_label_adjusted",
            "did",
            "affected_win_rate",
            "control_win_rate",
            "seed_set",
        ]
    ].rename(
        columns={
            "did_label_adjusted": "decoy_did_label_adjusted",
            "did": "decoy_did",
            "affected_win_rate": "decoy_affected_win_rate",
            "control_win_rate": "decoy_control_win_rate",
            "seed_set": "decoy_seed_set",
        }
    )
    minus = sem.merge(dec, on=["task", "dose", "seed"], how="left")
    if minus["decoy_did_label_adjusted"].isna().any():
        missing = minus[minus["decoy_did_label_adjusted"].isna()][["task", "relation", "dose", "seed"]]
        raise RuntimeError("Missing decoy rows:\n" + missing.to_string(index=False))

    minus["minus_decoy"] = minus["did_label_adjusted"] - minus["decoy_did_label_adjusted"]
    minus["semantic_positive"] = minus["did_label_adjusted"] > 0
    minus["minus_decoy_positive"] = minus["minus_decoy"] > 0
    minus = minus.sort_values(["task", "relation", "dose", "seed"])
    minus.to_csv(args.out_dir / f"{prefix}_minus_decoy_by_seed.csv", index=False)

    summary = minus.groupby(["task", "relation", "dose"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_semantic_adj_did=("did_label_adjusted", "mean"),
        std_semantic_adj_did=("did_label_adjusted", "std"),
        mean_decoy_adj_did=("decoy_did_label_adjusted", "mean"),
        std_decoy_adj_did=("decoy_did_label_adjusted", "std"),
        mean_minus_decoy=("minus_decoy", "mean"),
        median_minus_decoy=("minus_decoy", "median"),
        std_minus_decoy=("minus_decoy", "std"),
        q25_minus_decoy=("minus_decoy", q25),
        q75_minus_decoy=("minus_decoy", q75),
        min_minus_decoy=("minus_decoy", "min"),
        max_minus_decoy=("minus_decoy", "max"),
        positive_seeds=("minus_decoy_positive", "sum"),
        semantic_positive_seeds=("semantic_positive", "sum"),
    )
    summary["sem_minus_decoy"] = summary["std_minus_decoy"] / summary["n_seeds"].pow(0.5)
    summary["ci95_minus_decoy_halfwidth"] = 1.96 * summary["sem_minus_decoy"]
    summary["positive_seed_rate"] = summary["positive_seeds"] / summary["n_seeds"]
    summary["semantic_positive_seed_rate"] = summary["semantic_positive_seeds"] / summary["n_seeds"]
    summary["support_bucket"] = summary.apply(support_bucket, axis=1)
    summary = summary.sort_values(["mean_minus_decoy", "positive_seed_rate"], ascending=[False, False])
    summary.to_csv(args.out_dir / f"{prefix}_relation_summary.csv", index=False)

    aggregate = summary.groupby(["relation", "dose"], as_index=False).agg(
        tasks=("task", "nunique"),
        mean_minus_decoy=("mean_minus_decoy", "mean"),
        median_minus_decoy=("mean_minus_decoy", "median"),
        positive_tasks=("mean_minus_decoy", lambda x: int((x > 0).sum())),
        replicated_or_directional_tasks=(
            "support_bucket",
            lambda x: int(x.isin(["replicated_positive", "decoy_relative_positive", "directional_positive"]).sum()),
        ),
        mean_semantic_adj_did=("mean_semantic_adj_did", "mean"),
        mean_decoy_adj_did=("mean_decoy_adj_did", "mean"),
    )
    aggregate = aggregate.sort_values(["mean_minus_decoy", "positive_tasks"], ascending=[False, False])
    aggregate.to_csv(args.out_dir / f"{prefix}_relation_aggregate.csv", index=False)

    compact_rows = []
    for task, sub in summary.groupby("task"):
        best = sub.sort_values(["mean_minus_decoy", "positive_seed_rate"], ascending=[False, False]).iloc[0]
        compact_rows.append(
            {
                "task": task,
                "best_relation": best["relation"],
                "best_dose": int(best["dose"]),
                "best_mean_minus_decoy": best["mean_minus_decoy"],
                "best_ci95_halfwidth": best["ci95_minus_decoy_halfwidth"],
                "best_positive_seeds": int(best["positive_seeds"]),
                "best_n_seeds": int(best["n_seeds"]),
                "best_mean_semantic_adj_did": best["mean_semantic_adj_did"],
                "best_mean_decoy_adj_did": best["mean_decoy_adj_did"],
                "best_support_bucket": best["support_bucket"],
                "tested_relation_dose_cells": len(sub),
                "positive_cells": int((sub["mean_minus_decoy"] > 0).sum()),
                "replicated_or_directional_cells": int(
                    sub["support_bucket"].isin(["replicated_positive", "decoy_relative_positive", "directional_positive"]).sum()
                ),
            }
        )
    compact = pd.DataFrame(compact_rows).sort_values("best_mean_minus_decoy", ascending=False)
    compact.to_csv(args.out_dir / f"{prefix}_compact_task_findings.csv", index=False)

    stage = minus.groupby(["seed_set", "task", "relation", "dose"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_minus_decoy=("minus_decoy", "mean"),
        mean_semantic_adj_did=("did_label_adjusted", "mean"),
        mean_decoy_adj_did=("decoy_did_label_adjusted", "mean"),
        positive_seeds=("minus_decoy_positive", "sum"),
    )
    wide = stage.pivot_table(
        index=["task", "relation", "dose"],
        columns="seed_set",
        values=["mean_minus_decoy", "mean_semantic_adj_did", "mean_decoy_adj_did", "positive_seeds", "n_seeds"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{seed_set}" for metric, seed_set in wide.columns]
    wide = wide.reset_index()
    if {"mean_minus_decoy_old3", "mean_minus_decoy_extra4"}.issubset(wide.columns):
        wide["extra4_minus_old3"] = wide["mean_minus_decoy_extra4"] - wide["mean_minus_decoy_old3"]
    wide = wide.sort_values("mean_minus_decoy_extra4" if "mean_minus_decoy_extra4" in wide else "task", ascending=False)
    wide.to_csv(args.out_dir / f"{prefix}_old_vs_new_stability.csv", index=False)

    print(f"wrote {args.out_dir}")
    print("\nCompact findings")
    print(compact.to_string(index=False))
    print("\nTop relation summaries")
    print(summary.head(20).to_string(index=False))
    print("\nRelation aggregate")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
