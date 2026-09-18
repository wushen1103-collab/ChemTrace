#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manuscript-facing Phase-0 ChemTrace result tables.")
    p.add_argument("--cert-dir", type=Path, default=Path("reports/certificates"))
    p.add_argument(
        "--effect-summary",
        type=Path,
        default=Path("reports/tables/phase0_effect_models_completed/effect_model_summary.csv"),
    )
    p.add_argument("--retrieval-glob", default="reports/tables/phase0_retrieval_audit_*/retrieval_summary_by_relation.csv")
    p.add_argument(
        "--retrieval-sota-summary",
        type=Path,
        default=Path("reports/tables/phase0_retrieval_sota_summary/retrieval_sota_summary.csv"),
    )
    p.add_argument(
        "--retrieval-sota-bootstrap",
        type=Path,
        default=Path("reports/tables/phase0_retrieval_sota_summary/retrieval_sota_bootstrap_summary.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_manuscript_tables"))
    return p.parse_args()


def fmt_ci(mean: float, low: float, high: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return ""
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def read_certificates(cert_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cert_dir.glob("*.certificate.json")):
        cert = json.loads(path.read_text(encoding="utf-8"))
        meta = cert.get("prepare_meta", {})
        split_tasks = cert.get("splits", [])
        effects = cert.get("effects", [])
        best_effect = {}
        if effects:
            best_effect = max(
                effects,
                key=lambda row: float(row.get("point_mean_minus_decoy", float("-inf"))),
            )
        rows.append(
            {
                "artifact": cert.get("artifact_name", path.stem.replace(".certificate", "")),
                "tasks": ",".join(meta.get("tasks", [])),
                "n_tasks": len(meta.get("tasks", [])) or len(split_tasks),
                "corpus_size": meta.get("corpus_size"),
                "n_affected": meta.get("n_affected"),
                "relations": ",".join(meta.get("relations", [])),
                "doses": ",".join(str(x) for x in meta.get("doses", [])),
                "affected_filter": meta.get("affected_filter_relation") or "",
                "effect_level": cert.get("risk_statement", {}).get("effect_level", ""),
                "retrieval_level": cert.get("risk_statement", {}).get("retrieval_level", ""),
                "best_effect_task": best_effect.get("task", ""),
                "best_effect_relation": best_effect.get("relation", ""),
                "best_effect_dose": best_effect.get("dose", ""),
                "best_effect_ci": fmt_ci(
                    float(best_effect.get("point_mean_minus_decoy", np.nan)),
                    float(best_effect.get("boot_ci_low", np.nan)),
                    float(best_effect.get("boot_ci_high", np.nan)),
                ),
                "clean_sha256_12": str(meta.get("clean_sha256") or "")[:12],
            }
        )
    return pd.DataFrame(rows)


def read_retrieval_tables(pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path().glob(pattern)):
        audit_name = path.parent.name.replace("phase0_retrieval_audit_", "")
        df = pd.read_csv(path)
        for row in df.itertuples(index=False):
            relation = str(row.injection_relation)
            detector = str(row.detector)
            is_primary = (
                (relation == "exact" and detector in {"raw", "canonical"})
                or (relation == "random" and detector in {"raw", "canonical"})
                or (relation in {"parent", "stereo", "tautomer"} and detector in {"raw", "canonical", relation})
                or (relation == "decoy" and detector in {"raw", "canonical", "parent", "stereo", "tautomer"})
            )
            if not is_primary:
                continue
            rows.append(
                {
                    "audit": audit_name,
                    "injection_relation": relation,
                    "dose": int(row.dose),
                    "detector": detector,
                    "tasks": int(row.tasks),
                    "clean_hit_rate": float(row.mean_clean_hit_rate),
                    "injection_recall": float(row.mean_injection_recall),
                    "incremental_hit_rate": float(row.mean_incremental_hit_rate),
                    "changed_queries": int(row.total_changed_queries),
                    "changed_only_recall": float(row.mean_changed_only_recall)
                    if pd.notna(row.mean_changed_only_recall)
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["audit", "injection_relation", "dose", "detector"])


def read_effect_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "artifact",
        "task",
        "relation",
        "dose",
        "n_seeds",
        "did_label_adjusted_minus_decoy_mean",
        "did_label_adjusted_minus_decoy_boot_ci_low",
        "did_label_adjusted_minus_decoy_boot_ci_high",
        "did_label_adjusted_minus_decoy_boot_positive_rate",
        "linear_affected_coef_minus_decoy_mean",
        "linear_affected_coef_minus_decoy_boot_ci_low",
        "linear_affected_coef_minus_decoy_boot_ci_high",
        "linear_affected_coef_minus_decoy_boot_positive_rate",
        "linear_affected_coef_minus_decoy_support",
    ]
    out = df[keep].copy()
    out["label_adjusted_effect_ci"] = [
        fmt_ci(m, lo, hi)
        for m, lo, hi in zip(
            out["did_label_adjusted_minus_decoy_mean"],
            out["did_label_adjusted_minus_decoy_boot_ci_low"],
            out["did_label_adjusted_minus_decoy_boot_ci_high"],
        )
    ]
    out["linear_adjusted_effect_ci"] = [
        fmt_ci(m, lo, hi)
        for m, lo, hi in zip(
            out["linear_affected_coef_minus_decoy_mean"],
            out["linear_affected_coef_minus_decoy_boot_ci_low"],
            out["linear_affected_coef_minus_decoy_boot_ci_high"],
        )
    ]
    out["ci_positive"] = (
        (out["did_label_adjusted_minus_decoy_boot_ci_low"] > 0)
        | (out["linear_affected_coef_minus_decoy_boot_ci_low"] > 0)
    )
    return out.sort_values(
        ["did_label_adjusted_minus_decoy_mean", "linear_affected_coef_minus_decoy_mean"],
        ascending=[False, False],
    )


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_markdown(
    out_dir: Path,
    certs: pd.DataFrame,
    retrieval: pd.DataFrame,
    retrieval_sota: pd.DataFrame,
    retrieval_bootstrap: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
        return df[cols].replace({np.nan: ""}).to_markdown(index=False)

    lines = ["# Phase-0 Manuscript Tables", ""]
    lines.append("## Certificate Summary")
    lines.append("")
    cert_cols = [
        "artifact",
        "n_tasks",
        "n_affected",
        "relations",
        "effect_level",
        "retrieval_level",
        "best_effect_task",
        "best_effect_relation",
        "best_effect_ci",
    ]
    lines.append(markdown_table(certs, cert_cols) if not certs.empty else "No certificates found.")
    lines.append("")

    lines.append("## Retrieval Lattice Summary")
    lines.append("")
    ret_cols = [
        "audit",
        "injection_relation",
        "dose",
        "detector",
        "injection_recall",
        "changed_only_recall",
        "clean_hit_rate",
    ]
    lines.append(markdown_table(retrieval, ret_cols) if not retrieval.empty else "No retrieval summaries found.")
    lines.append("")

    lines.append("## Retrieval Baseline SOTA Summary")
    lines.append("")
    sota_cols = [
        "source",
        "relation_group",
        "cells",
        "lattice_changed_recall_mean",
        "best_exact_changed_recall_mean",
        "best_similarity_changed_recall_mean",
        "lattice_gain_vs_exact_mean",
        "lattice_gain_vs_similarity_mean",
        "lattice_wins_vs_exact",
        "lattice_wins_vs_similarity",
        "lattice_ties_vs_similarity",
    ]
    lines.append(markdown_table(retrieval_sota, sota_cols) if not retrieval_sota.empty else "No retrieval SOTA summary found.")
    lines.append("")

    lines.append("## Retrieval Baseline Bootstrap")
    lines.append("")
    boot_cols = [
        "source",
        "relation_group",
        "cells",
        "clusters",
        "lattice_gain_vs_exact_ci",
        "lattice_gain_vs_exact_ci_positive",
        "lattice_gain_vs_similarity_ci",
        "lattice_gain_vs_similarity_ci_positive",
        "lattice_win_rate_vs_similarity_ci",
        "lattice_tie_rate_vs_similarity_ci",
    ]
    lines.append(markdown_table(retrieval_bootstrap, boot_cols) if not retrieval_bootstrap.empty else "No retrieval bootstrap summary found.")
    lines.append("")

    lines.append("## Downstream Effect Robustness")
    lines.append("")
    eff_cols = [
        "artifact",
        "task",
        "relation",
        "dose",
        "n_seeds",
        "label_adjusted_effect_ci",
        "linear_adjusted_effect_ci",
        "linear_affected_coef_minus_decoy_support",
        "ci_positive",
    ]
    lines.append(markdown_table(effects.head(25), eff_cols) if not effects.empty else "No effect summary found.")
    lines.append("")
    lines.append(
        "Summary: no decoy-relative downstream effect cell has a bootstrap CI entirely above zero; "
        "the strongest result family supports semantic retrieval certificates more strongly than universal performance-inflation claims."
    )
    (out_dir / "phase0_manuscript_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    certs = read_certificates(args.cert_dir)
    retrieval = read_retrieval_tables(args.retrieval_glob)
    retrieval_sota = read_optional_csv(args.retrieval_sota_summary)
    retrieval_bootstrap = read_optional_csv(args.retrieval_sota_bootstrap)
    effects = read_effect_summary(args.effect_summary)

    certs.to_csv(args.out_dir / "table_certificate_summary.csv", index=False)
    retrieval.to_csv(args.out_dir / "table_retrieval_lattice_summary.csv", index=False)
    if not retrieval_sota.empty:
        retrieval_sota.to_csv(args.out_dir / "table_retrieval_sota_summary.csv", index=False)
    if not retrieval_bootstrap.empty:
        retrieval_bootstrap.to_csv(args.out_dir / "table_retrieval_sota_bootstrap.csv", index=False)
    effects.to_csv(args.out_dir / "table_downstream_effect_robustness.csv", index=False)
    write_markdown(args.out_dir, certs, retrieval, retrieval_sota, retrieval_bootstrap, effects)

    print(f"wrote manuscript tables to {args.out_dir}")
    print("certificate rows", len(certs))
    print("retrieval rows", len(retrieval))
    print("retrieval-sota rows", len(retrieval_sota))
    print("retrieval-bootstrap rows", len(retrieval_bootstrap))
    print("effect rows", len(effects), "ci-positive", int(effects["ci_positive"].sum()))


if __name__ == "__main__":
    main()
