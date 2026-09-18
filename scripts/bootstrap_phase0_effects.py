#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OLD_CLEAN_ARTIFACT = "phase0_tf_class_refined_len158"
OLD_DECOY_ARTIFACT = "phase0_tf_decoy_refined_len158"
EXTRA_ARTIFACT = "phase0_tf_confirm_top_len158_extra"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaffold-cluster bootstrap for ChemTrace semantic-minus-decoy effects.")
    p.add_argument(
        "--minus-by-seed",
        type=Path,
        default=Path("reports/tables/phase0_confirm_top_combined7/phase0_confirm_top_combined7_minus_decoy_by_seed.csv"),
    )
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/tables/phase0_confirm_top_combined7/phase0_confirm_top_combined7_cluster_bootstrap.csv"),
    )
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260831)
    return p.parse_args()


def prediction_path(artifacts_dir: Path, artifact: str, task: str, condition: str, seed: int) -> Path:
    return artifacts_dir / artifact / "predictions" / f"{task}__{condition}__seed{seed}.csv"


def clean_artifact(seed_set: str, semantic_artifact: str) -> str:
    if seed_set == "single":
        return semantic_artifact
    return EXTRA_ARTIFACT if seed_set == "extra4" else OLD_CLEAN_ARTIFACT


def decoy_artifact(seed_set: str, semantic_artifact: str) -> str:
    if seed_set == "single":
        return semantic_artifact
    return EXTRA_ARTIFACT if seed_set == "extra4" else OLD_DECOY_ARTIFACT


def row_value(row: object, name: str, default: object | None = None) -> object:
    if isinstance(row, pd.Series):
        if name in row:
            return row[name]
        if default is not None:
            return default
        raise KeyError(name)
    if hasattr(row, name):
        return getattr(row, name)
    if default is not None:
        return default
    raise AttributeError(name)


def label_adjusted_diff(df: pd.DataFrame, value_col: str) -> float:
    vals = []
    weights = []
    for _, part in df.groupby("y", dropna=True):
        aff = part[part["affected"].astype(bool)]
        ctl = part[~part["affected"].astype(bool)]
        if aff.empty or ctl.empty:
            continue
        vals.append(float(aff[value_col].mean() - ctl[value_col].mean()))
        weights.append(float(len(aff)))
    if not vals:
        return float("nan")
    return float(np.average(np.asarray(vals), weights=np.asarray(weights)))


def load_effect_frame(args: argparse.Namespace, row: pd.Series) -> pd.DataFrame:
    task = str(row_value(row, "task"))
    condition = str(row_value(row, "condition"))
    semantic_artifact = str(row_value(row, "artifact"))
    seed = int(row_value(row, "seed"))
    seed_set = str(row_value(row, "seed_set", "single"))
    dose = int(row_value(row, "dose"))
    decoy_condition = f"decoy_{dose}x"

    cpath = prediction_path(args.artifacts_dir, clean_artifact(seed_set, semantic_artifact), task, "clean", seed)
    spath = prediction_path(args.artifacts_dir, semantic_artifact, task, condition, seed)
    dpath = prediction_path(args.artifacts_dir, decoy_artifact(seed_set, semantic_artifact), task, decoy_condition, seed)
    for path in (cpath, spath, dpath):
        if not path.exists():
            raise FileNotFoundError(path)

    clean = pd.read_csv(cpath)
    sem = pd.read_csv(spath)
    decoy = pd.read_csv(dpath)
    keep = ["sample_id", "y", "affected", "scaffold", "sample_loss", "split"]
    clean = clean[clean["split"] == "test"][keep].rename(columns={"sample_loss": "loss_clean"})
    sem = sem[sem["split"] == "test"][["sample_id", "sample_loss"]].rename(columns={"sample_loss": "loss_semantic"})
    decoy = decoy[decoy["split"] == "test"][["sample_id", "sample_loss"]].rename(columns={"sample_loss": "loss_decoy"})
    pair = clean.merge(sem, on="sample_id", how="inner").merge(decoy, on="sample_id", how="inner")
    pair["semantic_minus_decoy_sample_effect"] = pair["loss_decoy"] - pair["loss_semantic"]
    pair["scaffold_cluster"] = pair["scaffold"].fillna(pair["sample_id"]).astype(str)
    return pair


def bootstrap_condition(frames: dict[int, pd.DataFrame], boot: int, rng: np.random.Generator) -> np.ndarray:
    seed_ids = np.asarray(sorted(frames), dtype=int)
    per_seed_clusters = {}
    for seed in seed_ids:
        df = frames[int(seed)].reset_index(drop=True)
        cluster_to_idx = [idx.to_numpy() for _, idx in df.groupby("scaffold_cluster").groups.items()]
        per_seed_clusters[int(seed)] = (df, cluster_to_idx)

    vals = np.empty(boot, dtype=float)
    vals[:] = np.nan
    for b in range(boot):
        chosen_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_vals = []
        for seed in chosen_seeds:
            df, clusters = per_seed_clusters[int(seed)]
            if not clusters:
                continue
            chosen_clusters = rng.integers(0, len(clusters), size=len(clusters))
            idx = np.concatenate([clusters[int(i)] for i in chosen_clusters])
            val = label_adjusted_diff(df.iloc[idx], "semantic_minus_decoy_sample_effect")
            if np.isfinite(val):
                seed_vals.append(val)
        if seed_vals:
            vals[b] = float(np.mean(seed_vals))
    return vals[np.isfinite(vals)]


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    minus = pd.read_csv(args.minus_by_seed)
    if "relation" in minus.columns:
        minus = minus[minus["relation"] != "decoy"].copy()
    rng = np.random.default_rng(args.seed)
    rows = []

    for (task, relation, dose), sub in minus.groupby(["task", "relation", "dose"], sort=True):
        frames = {int(row.seed): load_effect_frame(args, row) for row in sub.itertuples(index=False)}
        point_vals = [
            label_adjusted_diff(frame, "semantic_minus_decoy_sample_effect")
            for frame in frames.values()
        ]
        boot_vals = bootstrap_condition(frames, args.boot, rng)
        rows.append(
            {
                "task": task,
                "relation": relation,
                "dose": int(dose),
                "n_seeds": len(frames),
                "n_boot": int(args.boot),
                "n_boot_valid": int(len(boot_vals)),
                "point_mean_minus_decoy": float(np.mean(point_vals)),
                "point_std_over_seeds": float(np.std(point_vals, ddof=1)) if len(point_vals) > 1 else float("nan"),
                "boot_mean_minus_decoy": float(np.mean(boot_vals)) if len(boot_vals) else float("nan"),
                "boot_ci_low": float(np.quantile(boot_vals, 0.025)) if len(boot_vals) else float("nan"),
                "boot_ci_high": float(np.quantile(boot_vals, 0.975)) if len(boot_vals) else float("nan"),
                "boot_positive_rate": float((boot_vals > 0).mean()) if len(boot_vals) else float("nan"),
            }
        )

    out = pd.DataFrame(rows).sort_values(["point_mean_minus_decoy", "boot_positive_rate"], ascending=[False, False])
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
