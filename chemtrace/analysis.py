from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r"(?P<task>.+)__ (?P<condition>.+)__seed(?P<seed>\d+)".replace(" ", ""))


def _parse_name(path: Path) -> tuple[str, str, int]:
    stem = path.stem
    m = RUN_RE.match(stem)
    if not m:
        raise ValueError(f"Bad prediction filename: {path.name}")
    return m.group("task"), m.group("condition"), int(m.group("seed"))


def _task_metric(df: pd.DataFrame) -> dict:
    task_type = str(df["task_type"].iloc[0])
    test = df[df["split"] == "test"].copy()
    out = {"test_loss": float(test["sample_loss"].mean()), "n_test": int(len(test))}
    if task_type == "classification":
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = (test["y"].to_numpy() > 0.5).astype(int)
        p = test["prediction"].to_numpy()
        out["auprc"] = float(average_precision_score(y, p)) if len(set(y)) > 1 else np.nan
        out["auroc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y = test["y"].to_numpy()
        p = test["prediction"].to_numpy()
        out["rmse"] = float(mean_squared_error(y, p) ** 0.5)
        out["mae"] = float(mean_absolute_error(y, p))
        out["r2"] = float(r2_score(y, p)) if len(y) > 1 else np.nan
    return out


def analyze_predictions(pred_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    rows = []
    for path in sorted(pred_dir.glob("*.csv")):
        task, condition, seed = _parse_name(path)
        df = pd.read_csv(path)
        frames[(task, condition, seed)] = df
        row = {"task": task, "condition": condition, "seed": seed}
        row.update(_task_metric(df))
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(["task", "condition", "seed"]) if rows else pd.DataFrame()
    metrics.to_csv(out_dir / "phase0_task_metrics.csv", index=False)

    did_rows = []
    for (task, condition, seed), contam in frames.items():
        if condition == "clean":
            continue
        clean = frames.get((task, "clean", seed))
        if clean is None:
            continue
        task_type = str(clean["task_type"].iloc[0])
        keep_cols = ["sample_id", "sample_loss", "affected", "y", "task_type"]
        c = clean[clean["split"] == "test"][keep_cols].rename(columns={"sample_loss": "loss_clean"})
        z = contam[contam["split"] == "test"][["sample_id", "sample_loss"]].rename(columns={"sample_loss": "loss_contam"})
        pair = c.merge(z, on="sample_id", how="inner")
        pair["delta"] = pair["loss_clean"] - pair["loss_contam"]
        aff = pair[pair["affected"].astype(bool)]
        ctl = pair[~pair["affected"].astype(bool)]
        label_adjusted = (
            _label_adjusted_did(pair)
            if task_type == "classification"
            else {"did_label_adjusted": np.nan, "did_label_equal": np.nan, "n_labels_adjusted": 0}
        )
        did_rows.append(
            {
                "task": task,
                "condition": condition,
                "seed": seed,
                "n_affected": int(len(aff)),
                "n_control": int(len(ctl)),
                "delta_affected": float(aff["delta"].mean()) if len(aff) else np.nan,
                "delta_control": float(ctl["delta"].mean()) if len(ctl) else np.nan,
                "did": float(aff["delta"].mean() - ctl["delta"].mean()) if len(aff) and len(ctl) else np.nan,
                "affected_win_rate": float((aff["delta"] > 0).mean()) if len(aff) else np.nan,
                "control_win_rate": float((ctl["delta"] > 0).mean()) if len(ctl) else np.nan,
                **label_adjusted,
            }
        )
        pair.to_csv(out_dir / f"paired_losses_{task}_{condition}_seed{seed}.csv", index=False)
    did = pd.DataFrame(did_rows).sort_values(["task", "condition", "seed"]) if did_rows else pd.DataFrame()
    did.to_csv(out_dir / "phase0_did_by_seed.csv", index=False)
    summary = pd.DataFrame()
    if not did.empty:
        summary = (
            did.groupby(["task", "condition"])
            .agg(
                seeds=("seed", "nunique"),
                did_mean=("did", "mean"),
                did_std=("did", "std"),
                delta_affected_mean=("delta_affected", "mean"),
                delta_control_mean=("delta_control", "mean"),
                affected_win_rate=("affected_win_rate", "mean"),
                did_label_adjusted=("did_label_adjusted", "mean"),
                did_label_equal=("did_label_equal", "mean"),
                n_labels_adjusted=("n_labels_adjusted", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(out_dir / "phase0_did_summary.csv", index=False)
        dose = did["condition"].str.extract(r"(?P<relation>.+)_(?P<dose>\d+)x")
        if dose.notna().all(axis=None):
            did_dose = pd.concat([did, dose], axis=1)
            did_dose["dose"] = did_dose["dose"].astype(int)
            did_dose.to_csv(out_dir / "phase0_did_by_seed_with_dose.csv", index=False)
            dose_summary = (
                did_dose.groupby(["task", "relation", "dose"])
                .agg(
                    seeds=("seed", "nunique"),
                    did_mean=("did", "mean"),
                    did_std=("did", "std"),
                    delta_affected_mean=("delta_affected", "mean"),
                    delta_control_mean=("delta_control", "mean"),
                    affected_win_rate=("affected_win_rate", "mean"),
                    did_label_adjusted=("did_label_adjusted", "mean"),
                    did_label_equal=("did_label_equal", "mean"),
                    n_labels_adjusted=("n_labels_adjusted", "mean"),
                )
                .reset_index()
                .sort_values(["task", "relation", "dose"])
            )
            dose_summary.to_csv(out_dir / "phase0_dose_summary.csv", index=False)
        with (out_dir / "phase0_summary.md").open("w", encoding="utf-8") as f:
            f.write("# Phase-0 ChemTrace tables\n\n")
            f.write("## Task metrics\n\n")
            f.write(metrics.to_markdown(index=False))
            f.write("\n\n## DiD by seed\n\n")
            f.write(did.to_markdown(index=False))
            f.write("\n\n## DiD summary\n\n")
            f.write(summary.to_markdown(index=False))
            f.write("\n")
    else:
        with (out_dir / "phase0_summary.md").open("w", encoding="utf-8") as f:
            f.write("# Phase-0 ChemTrace tables\n\n")
            f.write("## Task metrics\n\n")
            f.write(metrics.to_markdown(index=False) if not metrics.empty else "No prediction files found.")
            f.write("\n\nNo paired contaminated conditions were available for DiD tables.\n")
    print(f"wrote tables to {out_dir}")


def _label_adjusted_did(pair: pd.DataFrame) -> dict:
    rows = []
    for label, part in pair.groupby("y", dropna=True):
        aff = part[part["affected"].astype(bool)]
        ctl = part[~part["affected"].astype(bool)]
        if aff.empty or ctl.empty:
            continue
        rows.append(
            {
                "label": label,
                "weight": len(aff),
                "did": float(aff["delta"].mean() - ctl["delta"].mean()),
            }
        )
    if not rows:
        return {"did_label_adjusted": np.nan, "did_label_equal": np.nan, "n_labels_adjusted": 0}
    weights = np.asarray([r["weight"] for r in rows], dtype=float)
    dids = np.asarray([r["did"] for r in rows], dtype=float)
    return {
        "did_label_adjusted": float(np.average(dids, weights=weights)),
        "did_label_equal": float(dids.mean()),
        "n_labels_adjusted": int(len(rows)),
    }
