#!/usr/bin/env python
"""Low-cost retrieval checks requested during submission revision.

This script verifies route-specific abstention behavior on the fixed evaluated
inputs and evaluates RDKit SuperParent as a relation-collapsing standardization
reference.  It does not modify any paper figure or train a model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from chemtrace.normalize import canonicalize, parent_smiles, stereo_stripped_smiles, tautomer_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/phase0_tf_confirm_top_len158_extra"),
    )
    parser.add_argument(
        "--external-pairs",
        type=Path,
        default=Path("reports/tables/phase0_independent_external_gold_v2/independent_relation_gold_pairs.csv"),
    )
    parser.add_argument(
        "--hard-bank",
        type=Path,
        default=Path("reports/tables/phase0_reviewer_supplements/hard_negative_pair_bank.csv"),
    )
    parser.add_argument(
        "--registry-negatives",
        type=Path,
        default=Path("reports/tables/phase0_independent_external_gold_v2/independent_registry_negative_pairs.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/tables/revision_retrieval"),
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


@lru_cache(maxsize=None)
def super_parent(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return ""
    try:
        parent = rdMolStandardize.SuperParent(mol)
        return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def equal_nonempty(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def read_pair_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"query_smiles", "target_smiles"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns {sorted(missing)}")
    return frame


def evaluated_strings(args: argparse.Namespace, tables: list[pd.DataFrame]) -> set[str]:
    strings: set[str] = set()
    for path in sorted((args.artifact_root / "corpora").glob("*.txt")):
        with path.open(encoding="utf-8") as handle:
            strings.update(line.strip() for line in handle if line.strip())
    affected = pd.read_csv(args.artifact_root / "phase0_affected.csv")
    for column in ["smiles", "canonical"]:
        if column in affected:
            strings.update(affected[column].dropna().astype(str))
    for table in tables:
        strings.update(table["query_smiles"].dropna().astype(str))
        strings.update(table["target_smiles"].dropna().astype(str))
    return strings


def route_abstention_chunk(strings: list[str]) -> dict[str, int]:
    routes = {
        "exact": canonicalize,
        "parent": parent_smiles,
        "stereo": stereo_stripped_smiles,
        "tautomer": tautomer_smiles,
    }
    counts = {name: 0 for name in routes}
    for smiles in strings:
        for name, function in routes.items():
            if not function(smiles):
                counts[name] += 1
    return counts


def route_abstention_audit(strings: set[str], workers: int) -> pd.DataFrame:
    valid = [smiles for smiles in sorted(strings) if canonicalize(smiles)]
    if workers > 1:
        chunks = [valid[offset::workers] for offset in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            partial = list(executor.map(route_abstention_chunk, chunks))
        totals = {name: sum(item[name] for item in partial) for name in ["exact", "parent", "stereo", "tautomer"]}
    else:
        totals = route_abstention_chunk(valid)
    rows = []
    for name in ["exact", "parent", "stereo", "tautomer"]:
        empty = totals[name]
        rows.append(
            {
                "route": name,
                "n_valid_input_strings": len(valid),
                "n_empty_route_keys": empty,
                "empty_route_key_rate": empty / len(valid) if valid else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def external_superparent(external: pd.DataFrame, keys: dict[str, str]) -> pd.DataFrame:
    frame = external[external["gold_relation"].isin(["parent", "stereo", "tautomer"])].copy()
    frame["superparent_call"] = [
        equal_nonempty(keys.get(left, ""), keys.get(right, ""))
        for left, right in zip(frame["query_smiles"].astype(str), frame["target_smiles"].astype(str))
    ]
    rows = []
    for relation, part in list(frame.groupby("gold_relation", sort=True)) + [("non_exact_union", frame)]:
        rows.append(
            {
                "scope": relation,
                "support": len(part),
                "superparent_detected": int(part["superparent_call"].sum()),
                "superparent_recall": float(part["superparent_call"].mean()),
            }
        )
    return pd.DataFrame(rows)


def controlled_superparent(artifact: Path, keys: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    affected = pd.read_csv(artifact / "phase0_affected.csv")
    original = dict(zip(affected["sample_id"].astype(str), affected["canonical"].astype(str)))
    task = dict(zip(affected["sample_id"].astype(str), affected["task"].astype(str)))
    rows = []
    for relation in ["parent", "stereo", "tautomer"]:
        for dose in [1, 5, 20]:
            path = artifact / "manifests" / f"{relation}_{dose}x.jsonl"
            first_by_sample: dict[str, dict[str, object]] = {}
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    first_by_sample.setdefault(str(record["sample_id"]), record)
            for sample_id, record in first_by_sample.items():
                source = original[sample_id]
                injected = canonicalize(str(record["injected_smiles"]))
                changed = bool(injected and injected != source)
                call = equal_nonempty(keys.get(source, ""), keys.get(injected, "")) if changed else False
                rows.append(
                    {
                        "task": task[sample_id],
                        "relation": relation,
                        "dose": dose,
                        "sample_id": sample_id,
                        "canonical_changed": changed,
                        "superparent_call": call,
                    }
                )
    pair_level = pd.DataFrame(rows)
    cell = (
        pair_level[pair_level["canonical_changed"]]
        .groupby(["task", "relation", "dose"], as_index=False)
        .agg(changed_queries=("sample_id", "nunique"), superparent_changed_recall=("superparent_call", "mean"))
    )
    return pair_level, cell


def bank_superparent(bank: pd.DataFrame, registry: pd.DataFrame, keys: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, frame in [("conditional_hard_negatives", bank), ("registry_negatives", registry)]:
        if "binary_label" in frame:
            frame = frame[pd.to_numeric(frame["binary_label"], errors="coerce").fillna(0).eq(0)].copy()
        calls = [
            equal_nonempty(keys.get(left, ""), keys.get(right, ""))
            for left, right in zip(frame["query_smiles"].astype(str), frame["target_smiles"].astype(str))
        ]
        rows.append(
            {
                "scope": name,
                "support": len(calls),
                "superparent_calls": int(sum(calls)),
                "superparent_call_rate": float(sum(calls) / len(calls)) if calls else float("nan"),
            }
        )
    positives = bank[pd.to_numeric(bank.get("binary_label", 0), errors="coerce").fillna(0).eq(1)].copy()
    calls = [
        equal_nonempty(keys.get(left, ""), keys.get(right, ""))
        for left, right in zip(positives["query_smiles"].astype(str), positives["target_smiles"].astype(str))
    ]
    rows.append(
        {
            "scope": "typed_positive_controls",
            "support": len(calls),
            "superparent_calls": int(sum(calls)),
            "superparent_call_rate": float(sum(calls) / len(calls)) if calls else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def superparent_input_strings(
    artifact: Path,
    tables: list[pd.DataFrame],
) -> set[str]:
    strings: set[str] = set()
    for table in tables:
        strings.update(table["query_smiles"].dropna().astype(str))
        strings.update(table["target_smiles"].dropna().astype(str))
    affected = pd.read_csv(artifact / "phase0_affected.csv")
    strings.update(affected["canonical"].dropna().astype(str))
    for relation in ["parent", "stereo", "tautomer"]:
        for dose in [1, 5, 20]:
            path = artifact / "manifests" / f"{relation}_{dose}x.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    strings.add(str(json.loads(line)["injected_smiles"]))
    return strings


def build_superparent_keys(strings: set[str], workers: int) -> dict[str, str]:
    ordered = sorted(strings)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(super_parent, ordered, chunksize=32))
    else:
        values = [super_parent(smiles) for smiles in ordered]
    return dict(zip(ordered, values))


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def manifest_pairs(artifact: Path, relation: str, dose: int) -> dict[str, list[str]]:
    path = artifact / "manifests" / f"{relation}_{dose}x.jsonl"
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            out.setdefault(sample_id, []).append(str(record["injected_smiles"]))
    return out


def superparent_common_protocol(
    artifact: Path,
    keys: dict[str, str],
    repeat_seeds: tuple[int, ...] = (101, 103, 107, 109, 113),
    max_clean: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate SuperParent under the exact Table-2 retrieval aggregation.

    The query sets, manifests, 1,000-record background subsamples, doses, and
    repeat seeds match ``audit_modern_sota_baselines.py``.  SuperParent equality
    is deliberately relation-collapsed, so the same decision rule is applied to
    every semantic injection route.
    """
    affected = pd.read_csv(artifact / "phase0_affected.csv")
    affected["sample_id"] = affected["sample_id"].astype(str)
    query = dict(zip(affected["sample_id"], affected["canonical"].astype(str)))
    task = dict(zip(affected["sample_id"], affected["task"].astype(str)))
    task_samples = {
        str(name): part["sample_id"].astype(str).tolist()
        for name, part in affected.groupby("task", sort=True)
    }

    clean_path = artifact / "corpora" / "clean.txt"
    with clean_path.open(encoding="utf-8") as handle:
        clean_all = [line.strip() for line in handle if line.strip()]

    semantic_hits: dict[tuple[str, int], dict[str, tuple[bool, bool]]] = {}
    for relation in ("parent", "stereo", "tautomer"):
        for dose in (1, 5, 20):
            injected = manifest_pairs(artifact, relation, dose)
            cell: dict[str, tuple[bool, bool]] = {}
            for sample_id, source in query.items():
                targets = list(dict.fromkeys(injected.get(sample_id, [])))
                changed = any(
                    bool(canonicalize(target) and canonicalize(target) != source)
                    for target in targets
                )
                hit = any(
                    equal_nonempty(keys.get(source, ""), keys.get(target, ""))
                    for target in targets
                )
                cell[sample_id] = (hit, changed)
            semantic_hits[(relation, dose)] = cell

    decoy_hits: dict[int, dict[str, bool]] = {}
    for dose in (1, 5, 20):
        injected = manifest_pairs(artifact, "decoy", dose)
        decoy_hits[dose] = {
            sample_id: any(
                equal_nonempty(keys.get(source, ""), keys.get(target, ""))
                for target in dict.fromkeys(injected.get(sample_id, []))
            )
            for sample_id, source in query.items()
        }

    rows: list[dict[str, object]] = []
    seed_offset = stable_int(artifact.name) % 1_000_000
    for repeat_seed in repeat_seeds:
        rng = np.random.default_rng(repeat_seed + seed_offset)
        if max_clean > 0 and max_clean < len(clean_all):
            indices = rng.choice(len(clean_all), size=max_clean, replace=False)
            clean_sample = [clean_all[int(index)] for index in indices]
        else:
            clean_sample = clean_all
        clean_keys = {keys.get(smiles, "") for smiles in clean_sample}
        clean_keys.discard("")

        for relation in ("parent", "stereo", "tautomer"):
            for dose in (1, 5, 20):
                pair_results = semantic_hits[(relation, dose)]
                for task_name, sample_ids in task_samples.items():
                    changed_calls = [
                        pair_results[sample_id][0]
                        for sample_id in sample_ids
                        if pair_results[sample_id][1]
                    ]
                    recall = float(np.mean(changed_calls)) if changed_calls else float("nan")
                    background = float(
                        np.mean(
                            [
                                bool(keys.get(query[sample_id], "") in clean_keys)
                                for sample_id in sample_ids
                            ]
                        )
                    )
                    decoy = float(np.mean([decoy_hits[dose][sample_id] for sample_id in sample_ids]))
                    rows.append(
                        {
                            "artifact": artifact.name,
                            "task": task_name,
                            "injection_relation": relation,
                            "dose": dose,
                            "repeat_seed": repeat_seed,
                            "n_queries": len(sample_ids),
                            "changed_queries": len(changed_calls),
                            "changed_only_recall": recall,
                            "clean_background_hit_rate": background,
                            "decoy_false_positive_proxy": decoy,
                            "background_and_decoy_adjusted_changed_recall": (
                                recall - background - decoy if changed_calls else float("nan")
                            ),
                        }
                    )

    cell_rows = pd.DataFrame(rows)
    by_seed = (
        cell_rows.groupby("repeat_seed", as_index=False)
        .agg(
            cells=("artifact", "size"),
            changed_cells=("changed_only_recall", "count"),
            changed_only_recall=("changed_only_recall", "mean"),
            clean_background_hit_rate=("clean_background_hit_rate", "mean"),
            decoy_false_positive_proxy=("decoy_false_positive_proxy", "mean"),
            background_and_decoy_adjusted_changed_recall=(
                "background_and_decoy_adjusted_changed_recall",
                "mean",
            ),
        )
    )
    summary = pd.DataFrame(
        [
            {
                "method": "rdkit_superparent",
                "repeat_seeds": len(by_seed),
                "cells_per_repeat": int(by_seed["cells"].iloc[0]),
                "changed_cells_per_repeat": int(by_seed["changed_cells"].iloc[0]),
                "changed_only_recall_mean": float(by_seed["changed_only_recall"].mean()),
                "changed_only_recall_sd": float(by_seed["changed_only_recall"].std(ddof=1)),
                "clean_background_hit_rate_mean": float(by_seed["clean_background_hit_rate"].mean()),
                "clean_background_hit_rate_sd": float(by_seed["clean_background_hit_rate"].std(ddof=1)),
                "decoy_false_positive_proxy_mean": float(by_seed["decoy_false_positive_proxy"].mean()),
                "decoy_false_positive_proxy_sd": float(by_seed["decoy_false_positive_proxy"].std(ddof=1)),
                "background_and_decoy_adjusted_mean": float(
                    by_seed["background_and_decoy_adjusted_changed_recall"].mean()
                ),
                "background_and_decoy_adjusted_sd": float(
                    by_seed["background_and_decoy_adjusted_changed_recall"].std(ddof=1)
                ),
            }
        ]
    )
    return cell_rows, summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    external = read_pair_table(args.external_pairs)
    bank = read_pair_table(args.hard_bank)
    registry = read_pair_table(args.registry_negatives)
    strings = evaluated_strings(args, [external, bank, registry])
    abstention = route_abstention_audit(strings, args.workers)
    sp_strings = superparent_input_strings(args.artifact_root, [external, bank, registry])
    with (args.artifact_root / "corpora" / "clean.txt").open(encoding="utf-8") as handle:
        sp_strings.update(line.strip() for line in handle if line.strip())
    sp_keys = build_superparent_keys(sp_strings, args.workers)
    external_result = external_superparent(external, sp_keys)
    controlled_pairs, controlled_cells = controlled_superparent(args.artifact_root, sp_keys)
    bank_result = bank_superparent(bank, registry, sp_keys)
    common_cells, common_summary = superparent_common_protocol(args.artifact_root, sp_keys)
    abstention.to_csv(args.out_dir / "route_abstention_counts.csv", index=False)
    external_result.to_csv(args.out_dir / "superparent_external.csv", index=False)
    controlled_pairs.to_csv(args.out_dir / "superparent_controlled_pairs.csv", index=False)
    controlled_cells.to_csv(args.out_dir / "superparent_controlled_cells.csv", index=False)
    bank_result.to_csv(args.out_dir / "superparent_negative_and_control_rates.csv", index=False)
    common_cells.to_csv(args.out_dir / "superparent_common_protocol_cells.csv", index=False)
    common_summary.to_csv(args.out_dir / "superparent_common_protocol_summary.csv", index=False)
    summary = {
        "rdkit_version": __import__("rdkit").__version__,
        "n_unique_evaluated_strings": len(strings),
        "superparent_preserves_relation_type": False,
        "controlled_changed_macro_recall": float(controlled_cells["superparent_changed_recall"].mean()),
        "controlled_changed_cells": len(controlled_cells),
        "common_protocol": common_summary.iloc[0].to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("ROUTE ABSTENTION\n", abstention.to_string(index=False))
    print("\nEXTERNAL SUPERPARENT\n", external_result.to_string(index=False))
    print("\nCONTROLLED SUPERPARENT\n", controlled_cells.groupby("relation").agg(cells=("task", "size"), recall=("superparent_changed_recall", "mean")).to_string())
    print("\nBANK SUPERPARENT\n", bank_result.to_string(index=False))
    print("\nCOMMON-PROTOCOL SUPERPARENT\n", common_summary.to_string(index=False))
    print("\nSUMMARY\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
