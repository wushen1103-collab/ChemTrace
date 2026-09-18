#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.normalize import canonicalize, parent_smiles, scaffold_smiles, stereo_stripped_smiles, tautomer_smiles


COND_RE = re.compile(r"(?P<relation>.+)_(?P<dose>\d+)x$")
DETECTORS = ["raw", "canonical", "parent", "stereo", "tautomer", "scaffold"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit L0-L5 retrieval recall on controlled ChemTrace injections.")
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0_tf_confirm_top_len158_extra"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_retrieval_audit"))
    p.add_argument("--num-workers", type=int, default=1, help="Parallel RDKit workers for clean corpus indexing.")
    p.add_argument("--progress-every", type=int, default=10000, help="Print clean-indexing progress every N molecules.")
    return p.parse_args()


@lru_cache(maxsize=500_000)
def detector_key(smiles: str, detector: str) -> str:
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


def affected_keys(affected: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in affected.itertuples(index=False):
        canonical = str(row.canonical)
        out[str(row.sample_id)] = {
            "raw": str(row.smiles),
            "canonical": canonical,
            "parent": str(row.parent),
            "stereo": str(row.stereo_stripped),
            "tautomer": tautomer_smiles(canonical),
            "scaffold": str(row.scaffold) if str(row.scaffold) else scaffold_smiles(canonical),
        }
    return out


def clean_keys_for_smiles(smiles: str) -> dict[str, str]:
    keys = {}
    for detector in DETECTORS:
        key = detector_key(smiles, detector)
        if key:
            keys[detector] = key
    return keys


def clean_counts(clean_path: Path, num_workers: int = 1, progress_every: int = 10000) -> dict[str, Counter[str]]:
    counts = {detector: Counter() for detector in DETECTORS}
    with clean_path.open("r", encoding="utf-8") as f:
        smiles_values = [line.strip() for line in f if line.strip()]

    total = len(smiles_values)
    print(f"indexing clean corpus: {total} molecules, workers={max(1, num_workers)}", flush=True)
    if num_workers > 1 and total > 1:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        with ctx.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(clean_keys_for_smiles, smiles_values, chunksize=256)
            for idx, keys in enumerate(iterator, start=1):
                for detector, key in keys.items():
                    counts[detector][key] += 1
                if progress_every and idx % progress_every == 0:
                    print(f"indexed clean corpus: {idx}/{total}", flush=True)
    else:
        for idx, smiles in enumerate(smiles_values, start=1):
            for detector, key in clean_keys_for_smiles(smiles).items():
                counts[detector][key] += 1
            if progress_every and idx % progress_every == 0:
                print(f"indexed clean corpus: {idx}/{total}", flush=True)
    return counts


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    affected = pd.read_csv(args.artifact_root / "phase0_affected.csv")
    query_keys = affected_keys(affected)
    clean = clean_counts(args.artifact_root / "corpora" / "clean.txt", args.num_workers, args.progress_every)

    clean_rows = []
    for task, task_df in affected.groupby("task"):
        sample_ids = task_df["sample_id"].astype(str).tolist()
        for detector in DETECTORS:
            keys = [query_keys[sid][detector] for sid in sample_ids]
            hits = [clean[detector].get(key, 0) > 0 if key else False for key in keys]
            counts = [clean[detector].get(key, 0) if key else 0 for key in keys]
            clean_rows.append(
                {
                    "task": task,
                    "detector": detector,
                    "n_queries": len(sample_ids),
                    "clean_hit_rate": sum(hits) / len(hits),
                    "mean_clean_matches": sum(counts) / len(counts),
                    "max_clean_matches": max(counts) if counts else 0,
                }
            )
    clean_df = pd.DataFrame(clean_rows).sort_values(["task", "detector"])
    clean_df.to_csv(args.out_dir / "retrieval_clean_baseline.csv", index=False)

    recall_rows = []
    for manifest_path in sorted((args.artifact_root / "manifests").glob("*.jsonl")):
        relation, dose = condition_meta(manifest_path)
        manifest = read_manifest(manifest_path)
        if manifest.empty:
            continue

        injected_by_sample: dict[str, list[str]] = defaultdict(list)
        canonical_changed: dict[str, bool] = defaultdict(bool)
        for row in manifest.itertuples(index=False):
            sample_id = str(row.sample_id)
            injected = str(row.injected_smiles)
            injected_by_sample[sample_id].append(injected)
            canonical_changed[sample_id] = canonical_changed[sample_id] or (
                canonicalize(injected) != query_keys[sample_id]["canonical"]
            )

        for task, task_df in affected.groupby("task"):
            sample_ids = task_df["sample_id"].astype(str).tolist()
            for detector in DETECTORS:
                hits = []
                clean_hits = []
                post_hits = []
                injected_counts = []
                changed_hits = []
                changed_total = 0
                for sid in sample_ids:
                    qkey = query_keys[sid][detector]
                    clean_count = clean[detector].get(qkey, 0) if qkey else 0
                    inj_count = 0
                    for injected in injected_by_sample.get(sid, []):
                        if qkey and detector_key(injected, detector) == qkey:
                            inj_count += 1
                    hit = inj_count > 0
                    clean_hit = clean_count > 0
                    hits.append(hit)
                    clean_hits.append(clean_hit)
                    post_hits.append(clean_hit or hit)
                    injected_counts.append(inj_count)
                    if canonical_changed.get(sid, False):
                        changed_total += 1
                        changed_hits.append(hit)
                n = len(sample_ids)
                recall_rows.append(
                    {
                        "task": task,
                        "condition": manifest_path.stem,
                        "injection_relation": relation,
                        "dose": dose,
                        "detector": detector,
                        "n_queries": n,
                        "clean_hit_rate": sum(clean_hits) / n,
                        "injection_recall": sum(hits) / n,
                        "post_hit_rate": sum(post_hits) / n,
                        "incremental_hit_rate": sum(h and not c for h, c in zip(hits, clean_hits)) / n,
                        "mean_injected_matches": sum(injected_counts) / n,
                        "changed_queries": changed_total,
                        "changed_only_recall": (sum(changed_hits) / changed_total) if changed_total else float("nan"),
                    }
                )

    recall = pd.DataFrame(recall_rows).sort_values(["injection_relation", "dose", "task", "detector"])
    recall.to_csv(args.out_dir / "retrieval_injection_recall.csv", index=False)
    summary = (
        recall.groupby(["injection_relation", "dose", "detector"], as_index=False)
        .agg(
            tasks=("task", "nunique"),
            mean_clean_hit_rate=("clean_hit_rate", "mean"),
            mean_injection_recall=("injection_recall", "mean"),
            mean_incremental_hit_rate=("incremental_hit_rate", "mean"),
            total_changed_queries=("changed_queries", "sum"),
            mean_changed_only_recall=("changed_only_recall", "mean"),
        )
        .sort_values(["injection_relation", "dose", "mean_injection_recall"], ascending=[True, True, False])
    )
    summary.to_csv(args.out_dir / "retrieval_summary_by_relation.csv", index=False)

    print(f"wrote retrieval audit tables to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
