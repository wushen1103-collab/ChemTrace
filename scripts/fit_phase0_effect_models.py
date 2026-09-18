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
    p = argparse.ArgumentParser(description="Fit CPU-only per-sample Phase-0 effect models.")
    p.add_argument("--artifact-roots", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260831)
    return p.parse_args()


def condition_parts(condition: str) -> tuple[str, int]:
    m = COND_RE.match(condition)
    if not m:
        return condition, -1
    return m.group("relation"), int(m.group("dose"))


def load_frames(artifact: Path) -> dict[tuple[str, str, int], pd.DataFrame]:
    frames = {}
    for path in sorted((artifact / "predictions").glob("*.csv")):
        task, condition, seed = _parse_name(path)
        frames[(task, condition, seed)] = pd.read_csv(path)
    return frames


def adjusted_affected_coef(pair: pd.DataFrame) -> float:
    y = pair["delta"].to_numpy(dtype=float)
    affected = pair["affected"].astype(bool).astype(float).to_numpy()
    cols = [np.ones(len(pair), dtype=float), affected]
    clean_loss = pair["loss_clean"].to_numpy(dtype=float)
    if np.isfinite(clean_loss).all() and np.nanstd(clean_loss) > 0:
        cols.append((clean_loss - clean_loss.mean()) / clean_loss.std())

    labels = pair["y"]
    if labels.nunique(dropna=True) <= 12:
        dummies = pd.get_dummies(labels.astype(str), prefix="label", drop_first=True, dtype=float)
        for col in dummies.columns:
            vals = dummies[col].to_numpy(dtype=float)
            if vals.std() > 0:
                cols.append(vals)
    else:
        vals = labels.to_numpy(dtype=float)
        if np.isfinite(vals).all() and np.nanstd(vals) > 0:
            cols.append((vals - vals.mean()) / vals.std())

    x = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(coef[1])


def paired_effect(clean: pd.DataFrame, contam: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    keep = ["sample_id", "y", "affected", "scaffold", "canonical", "sample_loss", "split", "task_type"]
    c = clean[clean["split"] == "test"][keep].rename(columns={"sample_loss": "loss_clean"})
    z = contam[contam["split"] == "test"][["sample_id", "sample_loss"]].rename(columns={"sample_loss": "loss_contam"})
    pair = c.merge(z, on="sample_id", how="inner")
    if len(pair) != len(c):
        raise RuntimeError(f"Pairing mismatch: paired={len(pair)} clean_test={len(c)}")
    pair["delta"] = pair["loss_clean"] - pair["loss_contam"]
    pair["canonical_len"] = pair["canonical"].astype(str).str.len()
    aff = pair[pair["affected"].astype(bool)]
    ctl = pair[~pair["affected"].astype(bool)]
    label_adj = (
        _label_adjusted_did(pair)
        if str(pair["task_type"].iloc[0]) == "classification"
        else {"did_label_adjusted": np.nan, "did_label_equal": np.nan, "n_labels_adjusted": 0}
    )
    stats = {
        "n_test": int(len(pair)),
        "n_affected": int(len(aff)),
        "n_control": int(len(ctl)),
        "affected_delta_mean": float(aff["delta"].mean()) if len(aff) else np.nan,
        "control_delta_mean": float(ctl["delta"].mean()) if len(ctl) else np.nan,
        "did": float(aff["delta"].mean() - ctl["delta"].mean()) if len(aff) and len(ctl) else np.nan,
        "affected_win_rate": float((aff["delta"] > 0).mean()) if len(aff) else np.nan,
        "control_win_rate": float((ctl["delta"] > 0).mean()) if len(ctl) else np.nan,
        "linear_affected_coef": adjusted_affected_coef(pair),
        **label_adj,
    }
    return pair, stats


def seed_bootstrap(vals: np.ndarray, boot: int, rng: np.random.Generator) -> dict[str, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"boot_mean": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_positive_rate": np.nan}
    draws = np.empty(boot, dtype=float)
    for i in range(boot):
        draws[i] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "boot_mean": float(draws.mean()),
        "boot_ci_low": float(np.quantile(draws, 0.025)),
        "boot_ci_high": float(np.quantile(draws, 0.975)),
        "boot_positive_rate": float((draws > 0).mean()),
    }


def support_bucket(ci_low: float, positive_rate: float) -> str:
    if np.isfinite(ci_low) and ci_low > 0:
        return "ci_positive"
    if np.isfinite(positive_rate) and positive_rate >= 0.8:
        return "directional_positive"
    if np.isfinite(positive_rate) and positive_rate <= 0.2:
        return "directional_negative"
    return "unstable_or_null"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for artifact in args.artifact_roots:
        frames = load_frames(artifact)
        for (task, condition, seed), contam in sorted(frames.items()):
            if condition == "clean":
                continue
            clean = frames.get((task, "clean", seed))
            if clean is None:
                continue
            relation, dose = condition_parts(condition)
            _, stats = paired_effect(clean, contam)
            rows.append(
                {
                    "artifact": artifact.name,
                    "task": task,
                    "condition": condition,
                    "relation": relation,
                    "dose": dose,
                    "seed": seed,
                    **stats,
                }
            )

    by_seed = pd.DataFrame(rows)
    if by_seed.empty:
        raise RuntimeError("No paired clean/contaminated prediction files found.")
    by_seed = by_seed.sort_values(["artifact", "task", "relation", "dose", "seed"])
    by_seed.to_csv(args.out_dir / "effect_model_by_seed.csv", index=False)

    decoy_cols = ["did", "did_label_adjusted", "linear_affected_coef"]
    minus_rows = []
    for _, row in by_seed[by_seed["relation"] != "decoy"].iterrows():
        decoy = by_seed[
            (by_seed["artifact"] == row["artifact"])
            & (by_seed["task"] == row["task"])
            & (by_seed["dose"] == row["dose"])
            & (by_seed["seed"] == row["seed"])
            & (by_seed["relation"] == "decoy")
        ]
        out = row.to_dict()
        if decoy.empty:
            for col in decoy_cols:
                out[f"{col}_minus_decoy"] = np.nan
        else:
            decoy_row = decoy.iloc[0]
            for col in decoy_cols:
                out[f"{col}_minus_decoy"] = float(row[col] - decoy_row[col])
        minus_rows.append(out)
    minus = pd.DataFrame(minus_rows).sort_values(["artifact", "task", "relation", "dose", "seed"])
    minus.to_csv(args.out_dir / "effect_model_minus_decoy_by_seed.csv", index=False)

    summary_rows = []
    for keys, sub in minus.groupby(["artifact", "task", "relation", "dose"], sort=True):
        artifact, task, relation, dose = keys
        out = {"artifact": artifact, "task": task, "relation": relation, "dose": int(dose), "n_seeds": int(sub["seed"].nunique())}
        for col in ["did", "did_label_adjusted", "linear_affected_coef", "did_minus_decoy", "did_label_adjusted_minus_decoy", "linear_affected_coef_minus_decoy"]:
            vals = sub[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            out[f"{col}_n_finite"] = int(len(finite))
            out[f"{col}_mean"] = float(finite.mean()) if len(finite) else np.nan
            out[f"{col}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            boot = seed_bootstrap(vals, args.boot, rng)
            for boot_key, boot_val in boot.items():
                out[f"{col}_{boot_key}"] = boot_val
            out[f"{col}_support"] = support_bucket(boot["boot_ci_low"], boot["boot_positive_rate"])
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["did_label_adjusted_minus_decoy_mean", "linear_affected_coef_minus_decoy_mean"],
        ascending=[False, False],
    )
    summary.to_csv(args.out_dir / "effect_model_summary.csv", index=False)

    top_cols = [
        "artifact",
        "task",
        "relation",
        "dose",
        "n_seeds",
        "did_label_adjusted_minus_decoy_n_finite",
        "did_label_adjusted_minus_decoy_mean",
        "did_label_adjusted_minus_decoy_boot_ci_low",
        "did_label_adjusted_minus_decoy_boot_ci_high",
        "did_label_adjusted_minus_decoy_boot_positive_rate",
        "linear_affected_coef_minus_decoy_n_finite",
        "linear_affected_coef_minus_decoy_mean",
        "linear_affected_coef_minus_decoy_boot_ci_low",
        "linear_affected_coef_minus_decoy_boot_ci_high",
        "linear_affected_coef_minus_decoy_boot_positive_rate",
        "linear_affected_coef_minus_decoy_support",
    ]
    print(f"wrote effect-model tables to {args.out_dir}")
    print(summary[top_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
