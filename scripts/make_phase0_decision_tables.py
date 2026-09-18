#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = [
    (
        "focused15_selected",
        Path("reports/tables/phase0_confirm_top_combined_focus15/phase0_confirm_top_focus15_cluster_bootstrap.csv"),
        "selected high-signal conditions after 15-seed confirmation",
    ),
    (
        "aff128_multitask_probe",
        Path("reports/tables/phase0_aff128_probe/phase0_aff128_cluster_bootstrap.csv"),
        "multi-task aff128 probe; MMP parent seeds extended to 12",
    ),
    (
        "mmp_aff128_isolated_parent",
        Path("reports/tables/phase0_mmp_aff128_isolated/phase0_mmp_aff128_isolated_cluster_bootstrap.csv"),
        "Tox21_MMP only, aff128, parent/exact/decoy 1x",
    ),
    (
        "mmp_stereo_changed_d20",
        Path("reports/tables/phase0_mmp_stereo_changed_aff128_d20/phase0_mmp_stereo_changed_aff128_d20_cluster_bootstrap.csv"),
        "Tox21_MMP stereo-changed affected set, dose 20",
    ),
    (
        "bbbp_stereo_changed_d20",
        Path("reports/tables/phase0_bbbp_stereo_changed_aff82_d20/phase0_bbbp_stereo_changed_aff82_d20_cluster_bootstrap.csv"),
        "BBBP stereo-changed affected set, dose 20",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compact Phase-0 decision tables from bootstrap outputs.")
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_decision"))
    p.add_argument(
        "--retrieval-bootstrap",
        type=Path,
        default=Path("reports/tables/phase0_retrieval_sota_summary/retrieval_sota_bootstrap_summary.csv"),
    )
    p.add_argument(
        "--effect-summary",
        type=Path,
        default=Path("reports/tables/phase0_effect_models_completed/effect_model_summary.csv"),
    )
    p.add_argument(
        "--literature-collision",
        type=Path,
        default=Path("reports/tables/phase0_literature_collision/semantic_provenance_collision_scan.csv"),
    )
    p.add_argument(
        "--modern-sota-readiness",
        type=Path,
        default=Path("reports/tables/phase0_modern_sota_baselines/modern_sota_readiness_after_baselines.csv"),
    )
    p.add_argument(
        "--modern-sota-claim",
        type=Path,
        default=Path("reports/tables/phase0_modern_sota_baselines/modern_sota_claim_summary.csv"),
    )
    return p.parse_args()


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {
        "mean_minus_decoy": "point_mean_minus_decoy",
        "ci_low": "boot_ci_low",
        "ci_high": "boot_ci_high",
    }
    return df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})


def _support_bucket(row: pd.Series) -> str:
    lo = float(row["boot_ci_low"])
    hi = float(row["boot_ci_high"])
    mean = float(row["point_mean_minus_decoy"])
    pos = float(row.get("boot_positive_rate", float("nan")))
    if lo > 0:
        return "positive_ci_excludes_zero"
    if hi < 0:
        return "negative_ci_excludes_zero"
    if mean > 0 and pos >= 0.85:
        return "directional_positive"
    if mean > 0:
        return "weak_positive"
    if mean < 0 and pos <= 0.15:
        return "directional_negative"
    return "near_null"


def _fmt_ci(row: pd.Series, prefix: str) -> str:
    return f"{float(row[prefix + '_mean']):.4f} [{float(row[prefix + '_ci_low']):.4f}, {float(row[prefix + '_ci_high']):.4f}]"


def build_sota_verdict(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    if args.retrieval_bootstrap.exists():
        retrieval = pd.read_csv(args.retrieval_bootstrap)
        focused = retrieval[
            (retrieval["source"] == "confirm_top_sample5k")
            & (retrieval["relation_group"] == "semantic_all")
        ]
        parent = retrieval[
            (retrieval["source"] == "confirm_top_sample5k")
            & (retrieval["relation_group"] == "parent")
        ]
        tautomer = retrieval[
            (retrieval["source"] == "confirm_top_sample5k")
            & (retrieval["relation_group"] == "tautomer")
        ]
        if not focused.empty:
            row = focused.iloc[0]
            parent_ci = _fmt_ci(parent.iloc[0], "lattice_gain_vs_similarity") if not parent.empty else ""
            tautomer_ci = _fmt_ci(tautomer.iloc[0], "lattice_gain_vs_similarity") if not tautomer.empty else ""
            similarity_positive = bool(row.get("lattice_gain_vs_similarity_ci_positive", False))
            exact_positive = bool(row.get("lattice_gain_vs_exact_ci_positive", False))
            rows.append(
                {
                    "claim_area": "semantic_provenance_retrieval",
                    "status": "sota_ready_primary_claim" if exact_positive and similarity_positive else "promising_needs_more_evidence",
                    "primary_evidence": (
                        f"focused semantic_all cells={int(row['cells'])}, task clusters={int(row['clusters'])}; "
                        f"lattice-vs-exact {_fmt_ci(row, 'lattice_gain_vs_exact')}; "
                        f"lattice-vs-best-similarity {_fmt_ci(row, 'lattice_gain_vs_similarity')}"
                    ),
                    "supporting_evidence": f"parent similarity gain {parent_ci}; tautomer similarity gain {tautomer_ci}",
                    "boundary": (
                        "Claim relation-typed semantic provenance and certificate-ready audit coverage; "
                        "do not claim universal superiority over every similarity baseline because stereo is a tie."
                    ),
                }
            )

    if args.modern_sota_readiness.exists() and args.modern_sota_claim.exists():
        readiness = pd.read_csv(args.modern_sota_readiness)
        claim = pd.read_csv(args.modern_sota_claim)
        failures = readiness[~readiness["status"].isin(["pass", "narrow_pass"])]
        semantic = claim[claim["relation_group"] == "semantic_all"]
        parent = claim[claim["relation_group"] == "parent"]
        stereo = claim[claim["relation_group"] == "stereo"]
        tautomer = claim[claim["relation_group"] == "tautomer"]
        if not semantic.empty:
            s = semantic.iloc[0]
            ptxt = parent.iloc[0]["gain_vs_best_external_adjusted_mean_std"] if not parent.empty else "NA"
            stxt = stereo.iloc[0]["gain_vs_best_external_adjusted_mean_std"] if not stereo.empty else "NA"
            ttxt = tautomer.iloc[0]["gain_vs_best_external_adjusted_mean_std"] if not tautomer.empty else "NA"
            external = readiness[readiness["criterion"] == "external_methods_8_to_12"]
            protocol = readiness[readiness["criterion"] == "same_protocol_same_split"]
            rows.append(
                {
                    "claim_area": "modern_external_baseline_suite",
                    "status": "sota_ready_narrow_provenance_claim" if failures.empty else "needs_more_evidence",
                    "primary_evidence": (
                        f"semantic_all cells/repeat={int(s['cells_per_repeat'])}, seeds={int(s['repeat_seeds'])}; "
                        f"ChemTrace adjusted={s['proposed_background_decoy_adjusted_mean_std']}; "
                        f"best external adjusted={s['best_external_adjusted_score_mean_std']}; "
                        f"gain={s['gain_vs_best_external_adjusted_mean_std']}"
                    ),
                    "supporting_evidence": (
                        f"{external.iloc[0]['evidence'] if not external.empty else 'external baseline count recorded'}; "
                        f"{protocol.iloc[0]['evidence'] if not protocol.empty else 'same protocol recorded'}; "
                        f"parent gain={ptxt}, stereo gain={stxt}, tautomer gain={ttxt}"
                    ),
                    "boundary": (
                        "This satisfies the modern same-protocol comparison for the relation-typed provenance/certificate claim. "
                        "External rows are source_type-labelled as self_rerun or adapted_rerun; downstream performance inflation remains a separate non-SOTA result."
                    ),
                }
            )

    if args.effect_summary.exists():
        effects = pd.read_csv(args.effect_summary)
        primary = effects["did_label_adjusted_minus_decoy_boot_ci_low"].astype(float) > 0.0
        diagnostic = effects["linear_affected_coef_minus_decoy_boot_ci_low"].astype(float) > 0.0
        best_idx = effects["did_label_adjusted_minus_decoy_mean"].astype(float).idxmax()
        best = effects.loc[best_idx]
        rows.append(
            {
                "claim_area": "downstream_performance_inflation",
                "status": "not_sota_not_ci_confirmed",
                "primary_evidence": (
                    f"primary CI-positive cells={int(primary.sum())}/{len(effects)}; "
                    f"best primary row={best['artifact']}:{best['task']}:{best['relation']}_{int(best['dose'])}x "
                    f"{float(best['did_label_adjusted_minus_decoy_mean']):.4f} "
                    f"[{float(best['did_label_adjusted_minus_decoy_boot_ci_low']):.4f}, "
                    f"{float(best['did_label_adjusted_minus_decoy_boot_ci_high']):.4f}]"
                ),
                "supporting_evidence": f"diagnostic adjusted-linear CI-positive cells={int(diagnostic.sum())}/{len(effects)}",
                "boundary": "Report downstream effects as calibrated risk bounds and negative/heterogeneous evidence, not as the main SOTA claim.",
            }
        )

    if args.literature_collision.exists():
        lit = pd.read_csv(args.literature_collision)
        assessment = lit["collision_assessment"].astype(str)
        direct = assessment.str.contains("direct collision", case=False, na=False) & ~assessment.str.contains(
            "not direct|no direct", case=False, na=False
        )
        rows.append(
            {
                "claim_area": "novelty_collision_scan",
                "status": "no_direct_collision_found_in_quick_scan" if not direct.any() else "needs_repositioning",
                "primary_evidence": f"screened {len(lit)} close works; direct-collision rows={int(direct.sum())}",
                "supporting_evidence": "Closest neighbors: DataSAIL, Chemical Science 2026 biomolecular benchmark audit, scContam, molecular ICL blinding.",
                "boundary": "Cite adjacent work explicitly and frame ChemTrace as pretraining-corpus semantic provenance rather than split hygiene alone.",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    missing = []
    for experiment, path, note in DEFAULT_INPUTS:
        if not path.exists():
            missing.append({"experiment": experiment, "path": str(path), "note": note})
            continue
        df = _canonicalize_columns(pd.read_csv(path))
        required = {"task", "relation", "dose", "n_seeds", "point_mean_minus_decoy", "boot_ci_low", "boot_ci_high"}
        miss = required.difference(df.columns)
        if miss:
            raise ValueError(f"{path} missing required columns {sorted(miss)}")
        df["experiment"] = experiment
        df["source_path"] = str(path)
        df["note"] = note
        frames.append(df)

    if not frames:
        raise RuntimeError("No bootstrap result files found.")

    summary = pd.concat(frames, ignore_index=True)
    summary["support_bucket"] = summary.apply(_support_bucket, axis=1)
    keep = [
        "experiment",
        "task",
        "relation",
        "dose",
        "n_seeds",
        "point_mean_minus_decoy",
        "point_std_over_seeds",
        "boot_ci_low",
        "boot_ci_high",
        "boot_positive_rate",
        "support_bucket",
        "note",
        "source_path",
    ]
    keep = [c for c in keep if c in summary.columns]
    summary = summary[keep].sort_values(["support_bucket", "point_mean_minus_decoy"], ascending=[True, False])
    summary.to_csv(args.out_dir / "phase0_decision_summary.csv", index=False)

    top = summary.sort_values("point_mean_minus_decoy", ascending=False).head(20)
    top.to_csv(args.out_dir / "phase0_decision_top20.csv", index=False)
    sota = build_sota_verdict(args)
    if not sota.empty:
        sota.to_csv(args.out_dir / "phase0_sota_verdict.csv", index=False)

    with (args.out_dir / "phase0_decision_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Phase-0 Decision Summary\n\n")
        f.write("Higher `point_mean_minus_decoy` indicates stronger affected-specific improvement after subtracting the matched decoy condition.\n\n")
        f.write(top.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n")
        if not sota.empty:
            f.write("\n## SOTA Verdict\n\n")
            f.write(sota.to_markdown(index=False))
            f.write("\n")
        if missing:
            f.write("\n## Missing Inputs\n\n")
            f.write(pd.DataFrame(missing).to_markdown(index=False))
            f.write("\n")

    print(f"wrote decision tables to {args.out_dir}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
