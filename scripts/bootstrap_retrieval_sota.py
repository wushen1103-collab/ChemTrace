#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEMANTIC_RELATIONS = ("parent", "stereo", "tautomer")
METRIC_COLUMNS = {
    "lattice_changed_recall": "lattice_changed_only_recall",
    "best_exact_changed_recall": "best_exact_changed_only_recall",
    "best_similarity_changed_recall": "best_similarity_changed_only_recall",
    "best_similarity_net_changed_recall": "best_similarity_net_changed_recall",
    "lattice_gain_vs_exact": "lattice_gain_vs_exact_net_changed",
    "lattice_gain_vs_similarity": "lattice_gain_vs_similarity_net_changed",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap relation-typed retrieval gains over exact and generic similarity baselines."
    )
    p.add_argument(
        "--claim-matrices",
        type=Path,
        nargs="*",
        default=None,
        help="retrieval_claim_matrix.csv files. Defaults to sample5k retrieval baseline outputs.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_retrieval_sota_summary"))
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=104729)
    return p.parse_args()


def discover_claim_matrices(paths: list[Path] | None) -> list[Path]:
    if paths:
        return [p for p in paths if p.exists()]
    return sorted(Path("reports/tables").glob("phase0_retrieval_baseline_*sample5k/retrieval_claim_matrix.csv"))


def source_name(path: Path) -> str:
    name = path.parent.name
    return name.replace("phase0_retrieval_baseline_", "")


def finite_semantic_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = df["injection_relation"].isin(SEMANTIC_RELATIONS)
    for col in METRIC_COLUMNS.values():
        keep &= np.isfinite(df[col].astype(float))
    out = df.loc[keep].copy()
    out["source"] = source_name(path)
    out["cluster_unit"] = out["artifact"].astype(str) + "::" + out["task"].astype(str)
    out["lattice_win_vs_exact"] = out["lattice_gain_vs_exact_net_changed"] > 1e-12
    out["lattice_win_vs_similarity"] = out["lattice_gain_vs_similarity_net_changed"] > 1e-12
    out["lattice_tie_vs_similarity"] = out["lattice_gain_vs_similarity_net_changed"].abs() <= 1e-12
    out["lattice_loss_vs_similarity"] = out["lattice_gain_vs_similarity_net_changed"] < -1e-12
    return out


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    for metric, col in METRIC_COLUMNS.items():
        row[metric] = float(df[col].mean())
    row["lattice_win_rate_vs_exact"] = float(df["lattice_win_vs_exact"].mean())
    row["lattice_win_rate_vs_similarity"] = float(df["lattice_win_vs_similarity"].mean())
    row["lattice_tie_rate_vs_similarity"] = float(df["lattice_tie_vs_similarity"].mean())
    row["lattice_loss_rate_vs_similarity"] = float(df["lattice_loss_vs_similarity"].mean())
    return row


def bootstrap_metrics(df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    clusters = sorted(df["cluster_unit"].unique())
    by_cluster = {cluster: part for cluster, part in df.groupby("cluster_unit", sort=False)}
    rows = []
    if len(clusters) <= 1:
        values = df.reset_index(drop=True)
        n = len(values)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            rows.append(compute_metrics(values.iloc[idx]))
    else:
        for _ in range(n_bootstrap):
            picked = rng.choice(clusters, size=len(clusters), replace=True)
            sampled = pd.concat([by_cluster[c] for c in picked], ignore_index=True)
            rows.append(compute_metrics(sampled))
    return pd.DataFrame(rows)


def summarize_group(source: str, relation_group: str, df: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    observed = compute_metrics(df)
    boots = bootstrap_metrics(df, n_bootstrap=n_bootstrap, rng=rng)
    summary = {
        "source": source,
        "relation_group": relation_group,
        "cells": int(len(df)),
        "clusters": int(df["cluster_unit"].nunique()),
    }
    long_rows = []
    for metric, mean in observed.items():
        low, high = np.quantile(boots[metric].to_numpy(), [0.025, 0.975])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_ci_low"] = float(low)
        summary[f"{metric}_ci_high"] = float(high)
        if metric.startswith("lattice_gain"):
            summary[f"{metric}_ci_positive"] = bool(low > 0.0)
        long_rows.append(
            {
                "source": source,
                "relation_group": relation_group,
                "metric": metric,
                "cells": int(len(df)),
                "clusters": int(df["cluster_unit"].nunique()),
                "mean": mean,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return summary, long_rows


def add_ci_strings(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for prefix in [
        "lattice_changed_recall",
        "best_exact_changed_recall",
        "best_similarity_changed_recall",
        "best_similarity_net_changed_recall",
        "lattice_gain_vs_exact",
        "lattice_gain_vs_similarity",
        "lattice_win_rate_vs_similarity",
        "lattice_tie_rate_vs_similarity",
    ]:
        out[f"{prefix}_ci"] = [
            f"{m:.4f} [{lo:.4f}, {hi:.4f}]"
            for m, lo, hi in zip(
                out[f"{prefix}_mean"],
                out[f"{prefix}_ci_low"],
                out[f"{prefix}_ci_high"],
            )
        ]
    return out


def write_markdown(path: Path, summary: pd.DataFrame) -> None:
    view_cols = [
        "source",
        "relation_group",
        "cells",
        "clusters",
        "lattice_changed_recall_ci",
        "best_exact_changed_recall_ci",
        "best_similarity_changed_recall_ci",
        "best_similarity_net_changed_recall_ci",
        "lattice_gain_vs_exact_ci",
        "lattice_gain_vs_similarity_ci",
        "lattice_win_rate_vs_similarity_ci",
        "lattice_tie_rate_vs_similarity_ci",
    ]
    lines = [
        "# Retrieval SOTA Bootstrap",
        "",
        "Cluster bootstrap is grouped by artifact and task; dose-level rows stay together inside a sampled task cluster.",
        "",
        summary[view_cols].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = discover_claim_matrices(args.claim_matrices)
    if not paths:
        raise SystemExit("No retrieval claim matrices found.")

    rng = np.random.default_rng(args.seed)
    frames = [finite_semantic_rows(path) for path in paths]
    all_rows = pd.concat(frames, ignore_index=True)

    summary_rows = []
    long_rows = []
    for source, source_df in all_rows.groupby("source", sort=True):
        for relation_group in ("semantic_all", *SEMANTIC_RELATIONS):
            if relation_group == "semantic_all":
                group = source_df
            else:
                group = source_df[source_df["injection_relation"] == relation_group]
            if group.empty:
                continue
            summary, long_part = summarize_group(
                source=source,
                relation_group=relation_group,
                df=group,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            summary_rows.append(summary)
            long_rows.extend(long_part)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = add_ci_strings(pd.DataFrame(summary_rows)).sort_values(["source", "relation_group"])
    long = pd.DataFrame(long_rows).sort_values(["source", "relation_group", "metric"])
    summary.to_csv(args.out_dir / "retrieval_sota_bootstrap_summary.csv", index=False)
    long.to_csv(args.out_dir / "retrieval_sota_bootstrap_long.csv", index=False)
    write_markdown(args.out_dir / "retrieval_sota_bootstrap.md", summary)

    print(f"wrote bootstrap retrieval SOTA tables to {args.out_dir}")
    print(f"claim matrices={len(paths)} semantic cells={len(all_rows)} bootstrap={args.n_bootstrap}")
    print(
        summary[
            [
                "source",
                "relation_group",
                "cells",
                "clusters",
                "lattice_gain_vs_exact_ci",
                "lattice_gain_vs_similarity_ci",
                "lattice_win_rate_vs_similarity_ci",
                "lattice_tie_rate_vs_similarity_ci",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
