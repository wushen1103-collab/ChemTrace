#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.normalize import canonicalize, parent_smiles, scaffold_smiles, stereo_stripped_smiles, tautomer_smiles


COND_RE = re.compile(r"(?P<relation>.+)_(?P<dose>\d+)x$")
KEY_DETECTORS = ["raw", "canonical", "parent", "stereo", "tautomer", "scaffold"]
ECFP_THRESHOLDS = [0.7, 0.8, 0.9]
DETECTORS = KEY_DETECTORS + [f"ecfp{int(t * 100)}" for t in ECFP_THRESHOLDS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare exact, semantic-lattice, scaffold, and ECFP retrieval baselines.")
    p.add_argument("--artifact-roots", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_retrieval_baseline_comparison"))
    p.add_argument("--max-clean", type=int, default=0, help="Optional cap for clean-corpus background scan; 0 means all.")
    p.add_argument("--tasks", nargs="+", default=None, help="Optional task subset for smoke tests.")
    return p.parse_args()


@lru_cache(maxsize=1_000_000)
def _mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=1_000_000)
def key_for(smiles: str, detector: str) -> str:
    if detector == "raw":
        return smiles if isinstance(smiles, str) else ""
    if detector == "canonical":
        return canonicalize(smiles)
    if detector == "parent":
        return parent_smiles(smiles)
    if detector == "stereo":
        return stereo_stripped_smiles(smiles)
    if detector == "tautomer":
        return tautomer_smiles(smiles)
    if detector == "scaffold":
        return scaffold_smiles(smiles) or canonicalize(smiles)
    raise KeyError(detector)


@lru_cache(maxsize=1_000_000)
def fp_for(smiles: str):
    mol = _mol(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def read_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def condition_meta(path: Path) -> tuple[str, int]:
    match = COND_RE.match(path.stem)
    if not match:
        raise ValueError(f"Bad condition filename: {path}")
    return match.group("relation"), int(match.group("dose"))


def _read_smiles(path: Path, max_clean: int) -> list[str]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
                if max_clean and len(out) >= max_clean:
                    break
    return out


def build_clean_index(clean_path: Path, max_clean: int) -> tuple[dict[str, Counter[str]], list, int]:
    smiles_values = _read_smiles(clean_path, max_clean=max_clean)
    key_counts = {detector: Counter() for detector in KEY_DETECTORS}
    fps = []
    for smi in smiles_values:
        for detector in KEY_DETECTORS:
            key = key_for(smi, detector)
            if key:
                key_counts[detector][key] += 1
        fp = fp_for(smi)
        if fp is not None:
            fps.append(fp)
    return key_counts, fps, len(smiles_values)


def max_tanimoto(query_fp, fps: list) -> float:
    if query_fp is None or not fps:
        return float("nan")
    vals = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
    return float(max(vals)) if vals else float("nan")


def detector_family(detector: str) -> str:
    if detector in {"raw", "canonical"}:
        return "exact_identity"
    if detector in {"parent", "stereo", "tautomer"}:
        return "semantic_lattice"
    if detector == "scaffold":
        return "scaffold"
    if detector.startswith("ecfp"):
        return "ecfp_similarity"
    return "other"


def expected_lattice_detector(relation: str) -> str:
    if relation in {"exact", "random"}:
        return "canonical"
    if relation == "parent":
        return "parent"
    if relation == "stereo":
        return "stereo"
    if relation == "tautomer":
        return "tautomer"
    return "canonical"


def audit_artifact(artifact_root: Path, max_clean: int, tasks: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    affected = pd.read_csv(artifact_root / "phase0_affected.csv")
    if tasks:
        affected = affected[affected["task"].isin(tasks)].copy()
        if affected.empty:
            raise ValueError(f"No affected queries found for tasks={tasks}")
    key_queries = {
        detector: {str(row.sample_id): key_for(str(getattr(row, "canonical", row.smiles)), detector) for row in affected.itertuples(index=False)}
        for detector in KEY_DETECTORS
    }
    raw_queries = {str(row.sample_id): str(row.smiles) for row in affected.itertuples(index=False)}
    key_queries["raw"] = raw_queries
    query_fps = {str(row.sample_id): fp_for(str(row.canonical)) for row in affected.itertuples(index=False)}
    clean_keys, clean_fps, n_clean_scanned = build_clean_index(artifact_root / "corpora" / "clean.txt", max_clean=max_clean)
    clean_ecfp_sim = {sid: max_tanimoto(fp, clean_fps) for sid, fp in query_fps.items()}

    clean_rows = []
    for task, task_df in affected.groupby("task"):
        sample_ids = task_df["sample_id"].astype(str).tolist()
        for detector in DETECTORS:
            if detector.startswith("ecfp"):
                threshold = int(detector.replace("ecfp", "")) / 100.0
                sims = [clean_ecfp_sim[sid] for sid in sample_ids]
                hits = [s >= threshold if not math.isnan(s) else False for s in sims]
                mean_matches = float(np.mean(sims)) if sims else float("nan")
                max_matches = float(np.nanmax(sims)) if sims else float("nan")
            else:
                keys = [key_queries[detector][sid] for sid in sample_ids]
                counts = [clean_keys[detector].get(key, 0) if key else 0 for key in keys]
                hits = [c > 0 for c in counts]
                mean_matches = float(np.mean(counts)) if counts else 0.0
                max_matches = float(max(counts)) if counts else 0.0
            clean_rows.append(
                {
                    "artifact": artifact_root.name,
                    "task": task,
                    "detector": detector,
                    "detector_family": detector_family(detector),
                    "n_queries": len(sample_ids),
                    "n_clean_scanned": n_clean_scanned,
                    "clean_background_hit_rate": sum(hits) / max(1, len(hits)),
                    "mean_clean_match_score": mean_matches,
                    "max_clean_match_score": max_matches,
                }
            )

    recall_rows = []
    for manifest_path in sorted((artifact_root / "manifests").glob("*.jsonl")):
        relation, dose = condition_meta(manifest_path)
        manifest = read_manifest(manifest_path)
        if tasks:
            manifest = manifest[manifest["task"].isin(tasks)].copy()
        if manifest.empty:
            continue
        injected_by_sample: dict[str, list[str]] = defaultdict(list)
        canonical_changed: dict[str, bool] = defaultdict(bool)
        for row in manifest.itertuples(index=False):
            sid = str(row.sample_id)
            injected = str(row.injected_smiles)
            injected_by_sample[sid].append(injected)
            canonical_changed[sid] = canonical_changed[sid] or (canonicalize(injected) != key_queries["canonical"][sid])

        for task, task_df in affected.groupby("task"):
            sample_ids = task_df["sample_id"].astype(str).tolist()
            for detector in DETECTORS:
                hits = []
                clean_hits = []
                changed_hits = []
                changed_total = 0
                scores = []
                if detector.startswith("ecfp"):
                    threshold = int(detector.replace("ecfp", "")) / 100.0
                    clean_sims = [clean_ecfp_sim[sid] for sid in sample_ids]
                    clean_hits = [s >= threshold if not math.isnan(s) else False for s in clean_sims]
                    for sid in sample_ids:
                        inj_fps = [fp_for(smi) for smi in injected_by_sample.get(sid, [])]
                        inj_fps = [fp for fp in inj_fps if fp is not None]
                        score = max_tanimoto(query_fps[sid], inj_fps)
                        hit = score >= threshold if not math.isnan(score) else False
                        hits.append(hit)
                        scores.append(score)
                        if canonical_changed.get(sid, False):
                            changed_total += 1
                            changed_hits.append(hit)
                else:
                    for sid in sample_ids:
                        qkey = key_queries[detector][sid]
                        clean_hit = clean_keys[detector].get(qkey, 0) > 0 if qkey else False
                        clean_hits.append(clean_hit)
                        inj_hit = False
                        inj_count = 0
                        for smi in injected_by_sample.get(sid, []):
                            if qkey and key_for(smi, detector) == qkey:
                                inj_hit = True
                                inj_count += 1
                        hits.append(inj_hit)
                        scores.append(float(inj_count))
                        if canonical_changed.get(sid, False):
                            changed_total += 1
                            changed_hits.append(inj_hit)
                n = len(sample_ids)
                recall_rows.append(
                    {
                        "artifact": artifact_root.name,
                        "task": task,
                        "condition": manifest_path.stem,
                        "injection_relation": relation,
                        "dose": dose,
                        "detector": detector,
                        "detector_family": detector_family(detector),
                        "expected_lattice_detector": expected_lattice_detector(relation),
                        "n_queries": n,
                        "n_clean_scanned": n_clean_scanned,
                        "clean_background_hit_rate": sum(clean_hits) / max(1, n),
                        "injection_recall": sum(hits) / max(1, n),
                        "incremental_hit_rate": sum(h and not c for h, c in zip(hits, clean_hits)) / max(1, n),
                        "mean_injection_match_score": float(np.nanmean(scores)) if scores else float("nan"),
                        "changed_queries": changed_total,
                        "changed_only_recall": (sum(changed_hits) / changed_total) if changed_total else float("nan"),
                    }
                )
    return clean_rows, recall_rows


def add_decoy_proxy(recall: pd.DataFrame) -> pd.DataFrame:
    decoy = recall[recall["injection_relation"] == "decoy"][
        ["artifact", "task", "dose", "detector", "injection_recall", "changed_only_recall"]
    ].rename(
        columns={
            "injection_recall": "decoy_false_positive_proxy",
            "changed_only_recall": "decoy_changed_false_positive_proxy",
        }
    )
    out = recall.merge(decoy, on=["artifact", "task", "dose", "detector"], how="left")
    out["net_recall_over_decoy"] = out["injection_recall"] - out["decoy_false_positive_proxy"].fillna(0.0)
    out["net_changed_recall_over_decoy"] = out["changed_only_recall"] - out["decoy_false_positive_proxy"].fillna(0.0)
    return out


def make_claim_matrix(recall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    non_decoy = recall[recall["injection_relation"] != "decoy"].copy()
    for keys, group in non_decoy.groupby(["artifact", "task", "injection_relation", "dose"], dropna=False):
        artifact, task, relation, dose = keys
        by_detector = {r.detector: r for r in group.itertuples(index=False)}
        lattice_detector = expected_lattice_detector(str(relation))
        lattice = by_detector.get(lattice_detector)
        exact = max(
            [by_detector.get(d) for d in ["raw", "canonical"] if by_detector.get(d) is not None],
            key=lambda r: r.net_changed_recall_over_decoy if not pd.isna(r.net_changed_recall_over_decoy) else r.net_recall_over_decoy,
        )
        similarity = max(
            [by_detector.get(d) for d in ["scaffold", "ecfp70", "ecfp80", "ecfp90"] if by_detector.get(d) is not None],
            key=lambda r: r.net_changed_recall_over_decoy if not pd.isna(r.net_changed_recall_over_decoy) else r.net_recall_over_decoy,
        )
        rows.append(
            {
                "artifact": artifact,
                "task": task,
                "injection_relation": relation,
                "dose": dose,
                "lattice_detector": lattice_detector,
                "lattice_recall": getattr(lattice, "injection_recall", np.nan),
                "lattice_changed_only_recall": getattr(lattice, "changed_only_recall", np.nan),
                "lattice_decoy_fp_proxy": getattr(lattice, "decoy_false_positive_proxy", np.nan),
                "lattice_net_changed_recall": getattr(lattice, "net_changed_recall_over_decoy", np.nan),
                "best_exact_detector": exact.detector,
                "best_exact_changed_only_recall": exact.changed_only_recall,
                "best_exact_decoy_fp_proxy": exact.decoy_false_positive_proxy,
                "best_exact_net_changed_recall": exact.net_changed_recall_over_decoy,
                "best_similarity_detector": similarity.detector,
                "best_similarity_changed_only_recall": similarity.changed_only_recall,
                "best_similarity_decoy_fp_proxy": similarity.decoy_false_positive_proxy,
                "best_similarity_net_changed_recall": similarity.net_changed_recall_over_decoy,
                "lattice_gain_vs_exact_net_changed": getattr(lattice, "net_changed_recall_over_decoy", np.nan) - exact.net_changed_recall_over_decoy,
                "lattice_gain_vs_similarity_net_changed": getattr(lattice, "net_changed_recall_over_decoy", np.nan) - similarity.net_changed_recall_over_decoy,
            }
        )
    return pd.DataFrame(rows).sort_values(["artifact", "task", "injection_relation", "dose"])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_rows = []
    recall_rows = []
    for artifact in args.artifact_roots:
        print(f"auditing {artifact}", flush=True)
        c_rows, r_rows = audit_artifact(artifact, max_clean=args.max_clean, tasks=args.tasks)
        clean_rows.extend(c_rows)
        recall_rows.extend(r_rows)

    clean = pd.DataFrame(clean_rows).sort_values(["artifact", "task", "detector"])
    recall = pd.DataFrame(recall_rows).sort_values(["artifact", "injection_relation", "dose", "task", "detector"])
    recall = add_decoy_proxy(recall)
    claim = make_claim_matrix(recall)
    detector_summary = (
        recall.groupby(["injection_relation", "dose", "detector", "detector_family"], as_index=False)
        .agg(
            rows=("artifact", "count"),
            mean_clean_background_hit_rate=("clean_background_hit_rate", "mean"),
            mean_injection_recall=("injection_recall", "mean"),
            mean_changed_only_recall=("changed_only_recall", "mean"),
            mean_decoy_fp_proxy=("decoy_false_positive_proxy", "mean"),
            mean_net_recall_over_decoy=("net_recall_over_decoy", "mean"),
            mean_net_changed_recall_over_decoy=("net_changed_recall_over_decoy", "mean"),
        )
        .sort_values(["injection_relation", "dose", "mean_net_changed_recall_over_decoy"], ascending=[True, True, False])
    )

    clean.to_csv(args.out_dir / "retrieval_clean_background_by_detector.csv", index=False)
    recall.to_csv(args.out_dir / "retrieval_detector_recall_with_decoy_proxy.csv", index=False)
    claim.to_csv(args.out_dir / "retrieval_claim_matrix.csv", index=False)
    detector_summary.to_csv(args.out_dir / "retrieval_detector_summary.csv", index=False)
    with (args.out_dir / "retrieval_baseline_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Retrieval Baseline Comparison\n\n")
        f.write("## Claim Matrix\n\n")
        f.write(claim.to_markdown(index=False))
        f.write("\n\n## Detector Summary\n\n")
        f.write(detector_summary.to_markdown(index=False))
        f.write("\n")

    print(f"wrote retrieval baseline comparison tables to {args.out_dir}")
    print(f"clean rows={len(clean)} recall rows={len(recall)} claim rows={len(claim)} detector-summary rows={len(detector_summary)}")
    print("\nClaim matrix preview:")
    print(claim.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
