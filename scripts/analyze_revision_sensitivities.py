#!/usr/bin/env python
"""Reanalyze recorded prediction files for the submission revision.

The script does not train a model.  It (i) rewrites the primary estimand in
its algebraically equivalent injected-versus-decoy form, (ii) restricts
non-exact affected sets to canonical-changing records, and (iii) computes
endpoint-clustered global summaries and a centered-bootstrap dispersion test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chemtrace.normalize import canonicalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path("reports/tables/phase0_manuscript_tables/table_downstream_effect_robustness.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/tables/revision_sensitivities"),
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})


def label_adjusted_effect(frame: pd.DataFrame) -> float:
    estimates: list[float] = []
    weights: list[int] = []
    for _, part in frame.groupby("y", dropna=True):
        affected = part[part["affected"]]
        control = part[~part["affected"]]
        if affected.empty or control.empty:
            continue
        estimates.append(float(affected["effect"].mean() - control["effect"].mean()))
        weights.append(len(affected))
    if not estimates:
        return float("nan")
    return float(np.average(np.asarray(estimates), weights=np.asarray(weights)))


@dataclass
class SeedFrame:
    frame: pd.DataFrame
    clusters: list[np.ndarray]


def prepare_seed_frame(frame: pd.DataFrame) -> SeedFrame:
    frame = frame.reset_index(drop=True)
    clusters = [np.asarray(indices, dtype=int) for indices in frame.groupby("cluster", sort=False).indices.values()]
    return SeedFrame(frame=frame, clusters=clusters)


def bootstrap_cell(
    frames: dict[int, pd.DataFrame],
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    prepared = {seed: prepare_seed_frame(frame) for seed, frame in frames.items()}
    seeds = np.asarray(sorted(prepared), dtype=int)
    per_seed = np.asarray([label_adjusted_effect(prepared[int(seed)].frame) for seed in seeds], dtype=float)
    point = float(np.nanmean(per_seed))
    boot = np.full(draws, np.nan, dtype=float)
    for draw in range(draws):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_effects: list[float] = []
        for seed in sampled_seeds:
            item = prepared[int(seed)]
            if not item.clusters:
                continue
            sampled_clusters = rng.integers(0, len(item.clusters), size=len(item.clusters))
            indices = np.concatenate([item.clusters[int(index)] for index in sampled_clusters])
            value = label_adjusted_effect(item.frame.iloc[indices])
            if np.isfinite(value):
                seed_effects.append(value)
        if seed_effects:
            boot[draw] = float(np.mean(seed_effects))
    if np.isnan(boot).any():
        valid = boot[np.isfinite(boot)]
        if valid.size != draws:
            raise RuntimeError(f"Only {valid.size}/{draws} valid bootstrap draws")
    return point, boot


def condition_name(relation: str, dose: int) -> str:
    return f"{relation}_{dose}x"


def prediction_seeds(prediction_dir: Path, task: str, condition: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    prefix = f"{task}__{condition}__seed"
    for path in sorted(prediction_dir.glob(f"{prefix}*.csv")):
        seed = int(path.stem.split("__seed", 1)[1])
        output[seed] = path
    return output


def changed_ids(artifact: Path, task: str, relation: str, dose: int) -> set[str]:
    manifest = artifact / "manifests" / f"{relation}_{dose}x.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    affected = pd.read_csv(artifact / "phase0_affected.csv")
    affected = affected[(affected["task"].astype(str) == task) & as_bool(affected["affected"])].copy()
    originals = dict(zip(affected["sample_id"].astype(str), affected["canonical"].astype(str)))
    changed: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("task")) != task:
                continue
            sample_id = str(record["sample_id"])
            original = originals.get(sample_id)
            if original is None:
                continue
            injected = canonicalize(str(record["injected_smiles"]))
            if injected and injected != original:
                changed.add(sample_id)
    return changed


def load_cell_frames(
    artifact: Path,
    task: str,
    relation: str,
    dose: int,
    restrict_changed: bool,
) -> tuple[dict[int, pd.DataFrame], int, int]:
    prediction_dir = artifact / "predictions"
    semantic = prediction_seeds(prediction_dir, task, condition_name(relation, dose))
    decoy = prediction_seeds(prediction_dir, task, condition_name("decoy", dose))
    seeds = sorted(set(semantic) & set(decoy))
    if not seeds:
        raise RuntimeError(f"No paired predictions for {artifact.name} {task} {relation}/{dose}")
    keep_ids: set[str] | None = None
    if restrict_changed:
        keep_ids = changed_ids(artifact, task, relation, dose)
    frames: dict[int, pd.DataFrame] = {}
    n_affected_original = 0
    n_affected_used = 0
    for seed in seeds:
        sem = pd.read_csv(semantic[seed])
        dec = pd.read_csv(decoy[seed])
        sem = sem[sem["split"].astype(str) == "test"].copy()
        dec = dec[dec["split"].astype(str) == "test"][["sample_id", "sample_loss"]].copy()
        sem["sample_id"] = sem["sample_id"].astype(str)
        dec["sample_id"] = dec["sample_id"].astype(str)
        sem["affected"] = as_bool(sem["affected"])
        pair = sem.merge(dec, on="sample_id", how="inner", suffixes=("_semantic", "_decoy"))
        pair["effect"] = pair["sample_loss_decoy"].astype(float) - pair["sample_loss_semantic"].astype(float)
        original_affected = pair["affected"].copy()
        if restrict_changed:
            assert keep_ids is not None
            is_changed = pair["sample_id"].isin(keep_ids)
            pair = pair[(~original_affected) | is_changed].copy()
            pair["affected"] = pair["affected"] & pair["sample_id"].isin(keep_ids)
        pair["cluster"] = pair["scaffold"].fillna("").astype(str)
        pair.loc[pair["cluster"] == "", "cluster"] = pair.loc[pair["cluster"] == "", "sample_id"]
        frames[seed] = pair[["sample_id", "y", "affected", "cluster", "effect"]].copy()
        if seed == seeds[0]:
            n_affected_original = int(original_affected.sum())
            n_affected_used = int(pair["affected"].sum())
    return frames, n_affected_original, n_affected_used


def summarize_draws(point: float, draws: np.ndarray) -> dict[str, float]:
    return {
        "point": point,
        "boot_mean": float(draws.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "positive_probability": float((draws > 0).mean()),
    }


def analyze_one_cell(job: tuple[int, dict[str, object], str, int, int]) -> tuple[dict[str, object], np.ndarray, np.ndarray | None]:
    index, cell, artifacts_dir, draws, seed = job
    artifact = Path(artifacts_dir) / str(cell["artifact"])
    task = str(cell["task"])
    relation = str(cell["relation"])
    dose = int(cell["dose"])
    rng = np.random.default_rng(seed + 104729 * (index + 1))
    frames, original_n, _ = load_cell_frames(artifact, task, relation, dose, False)
    if len(frames) != int(cell["n_seeds"]):
        raise RuntimeError(
            f"Seed mismatch for {artifact.name} {task} {relation}/{dose}: "
            f"found {len(frames)}, expected {cell['n_seeds']}"
        )
    point, primary_boot = bootstrap_cell(frames, draws, rng)
    record: dict[str, object] = {
        "cell_index": index,
        "artifact": artifact.name,
        "task": task,
        "relation": relation,
        "dose": dose,
        "n_seeds": len(frames),
        "n_affected_primary": original_n,
        **summarize_draws(point, primary_boot),
    }
    changed_boot: np.ndarray | None = None
    if relation in {"parent", "stereo", "tautomer"}:
        changed_frames, _, changed_n = load_cell_frames(artifact, task, relation, dose, True)
        changed_point, changed_boot = bootstrap_cell(changed_frames, draws, rng)
        record.update(
            {
                "n_affected_changed": changed_n,
                "changed_point": changed_point,
                "changed_ci_low": float(np.quantile(changed_boot, 0.025)),
                "changed_ci_high": float(np.quantile(changed_boot, 0.975)),
                "changed_positive_probability": float((changed_boot > 0).mean()),
            }
        )
    return record, primary_boot, changed_boot


def clustered_global(
    table: pd.DataFrame,
    draw_matrix: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    selected = table.loc[mask].reset_index(drop=True)
    selected_draws = draw_matrix[mask]
    tasks = np.asarray(sorted(selected["task"].unique()), dtype=object)
    point = float(selected["point"].mean())
    output = np.empty(selected_draws.shape[1], dtype=float)
    positive_fraction = np.empty(selected_draws.shape[1], dtype=float)
    for draw in range(selected_draws.shape[1]):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(selected["task"].to_numpy(dtype=object) == task) for task in sampled_tasks]
        )
        values = selected_draws[indices, draw]
        output[draw] = float(values.mean())
        positive_fraction[draw] = float((values > 0).mean())
    return point, output, positive_fraction


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.cells).copy()
    required = {"artifact", "task", "relation", "dose", "n_seeds"}
    missing = required - set(cells.columns)
    if missing:
        raise KeyError(f"Missing cell columns: {sorted(missing)}")
    cells = cells[list(required)].copy()
    cells["dose"] = cells["dose"].astype(int)
    rows: list[dict[str, object]] = []
    primary_draws: list[np.ndarray] = []
    changed_draws: dict[int, np.ndarray] = {}
    jobs = [
        (index, cell.to_dict(), str(args.artifacts_dir), args.bootstrap, args.seed)
        for index, cell in cells.iterrows()
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            analyzed = list(executor.map(analyze_one_cell, jobs))
    else:
        analyzed = [analyze_one_cell(job) for job in jobs]
    for record, draws, changed_boot in analyzed:
        index = int(record["cell_index"])
        rows.append(record)
        primary_draws.append(draws)
        if changed_boot is not None:
            changed_draws[index] = changed_boot

    cell_results = pd.DataFrame(rows)
    draw_matrix = np.vstack(primary_draws)
    cell_results.to_csv(args.out_dir / "cell_sensitivity.csv", index=False)

    global_rng = np.random.default_rng(args.seed + 1)
    summaries: list[dict[str, object]] = []
    masks = {
        "all_51_reported_cells": np.ones(len(cell_results), dtype=bool),
        "primary_protocol_23_cells": cell_results["artifact"].eq("phase0_tf_confirm_top_len158_extra").to_numpy(),
    }
    for name, mask in masks.items():
        point, draws, positive = clustered_global(cell_results, draw_matrix, mask, global_rng)
        summaries.append(
            {
                "analysis": name,
                "n_cells": int(mask.sum()),
                "n_endpoint_clusters": int(cell_results.loc[mask, "task"].nunique()),
                **summarize_draws(point, draws),
                "positive_cell_fraction_point": float((cell_results.loc[mask, "point"] > 0).mean()),
                "positive_cell_fraction_ci_low": float(np.quantile(positive, 0.025)),
                "positive_cell_fraction_ci_high": float(np.quantile(positive, 0.975)),
            }
        )

    changed_indices = np.asarray(sorted(changed_draws), dtype=int)
    changed_table = cell_results.loc[changed_indices].reset_index(drop=True)
    changed_table["point"] = changed_table["changed_point"].astype(float)
    changed_matrix = np.vstack([changed_draws[int(index)] for index in changed_indices])
    changed_masks = {
        "primary_protocol_non_exact": changed_table["artifact"].eq("phase0_tf_confirm_top_len158_extra").to_numpy(),
        "all_non_exact": np.ones(len(changed_table), dtype=bool),
        "parent": changed_table["relation"].eq("parent").to_numpy(),
        "stereo": changed_table["relation"].eq("stereo").to_numpy(),
        "tautomer": changed_table["relation"].eq("tautomer").to_numpy(),
    }
    for relation, mask in changed_masks.items():
        point, draws, positive = clustered_global(changed_table, changed_matrix, mask, global_rng)
        summaries.append(
            {
                "analysis": f"changed_only_{relation}",
                "n_cells": int(mask.sum()),
                "n_endpoint_clusters": int(changed_table.loc[mask, "task"].nunique()),
                **summarize_draws(point, draws),
                "positive_cell_fraction_point": float((changed_table.loc[mask, "changed_point"] > 0).mean()),
                "positive_cell_fraction_ci_low": float(np.quantile(positive, 0.025)),
                "positive_cell_fraction_ci_high": float(np.quantile(positive, 0.975)),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "global_cluster_bootstrap.csv", index=False)

    point_values = cell_results["point"].to_numpy(dtype=float)
    observed_variance = float(np.var(point_values, ddof=1))
    centered = draw_matrix - draw_matrix.mean(axis=1, keepdims=True)
    null_variance = np.var(centered + point_values.mean(), axis=0, ddof=1)
    raw_variance = np.var(draw_matrix, axis=0, ddof=1)
    dispersion = pd.DataFrame(
        [
            {
                "n_cells": len(point_values),
                "observed_between_cell_variance": observed_variance,
                "raw_bootstrap_variance_ci_low": float(np.quantile(raw_variance, 0.025)),
                "raw_bootstrap_variance_ci_high": float(np.quantile(raw_variance, 0.975)),
                "centered_null_variance_mean": float(null_variance.mean()),
                "centered_bootstrap_p_value": float((1 + np.sum(null_variance >= observed_variance)) / (len(null_variance) + 1)),
            }
        ]
    )
    dispersion.to_csv(args.out_dir / "between_cell_dispersion.csv", index=False)

    metadata = {
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "estimand": "label-adjusted affected-minus-control contrast in loss_decoy - loss_injected",
        "global_resampling": "endpoint clusters with within-cell paired-seed and scaffold-cluster bootstrap",
        "changed_only_definition": "affected sample is retained when canonical(injected) differs from canonical(original)",
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(cell_results.to_string(index=False))
    print("\nGLOBAL\n", summary.to_string(index=False))
    print("\nDISPERSION\n", dispersion.to_string(index=False))


if __name__ == "__main__":
    main()
