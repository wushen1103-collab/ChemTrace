#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_reviewer_supplement_experiments import (
    DEFAULT_CHEMBL_DB,
    ROOT,
    relation_flags,
    run_independent_relation_gold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild external relation gold without detector-defined inclusion criteria."
    )
    parser.add_argument("--chembl-db", type=Path, default=DEFAULT_CHEMBL_DB)
    parser.add_argument(
        "--hard-negative-bank",
        type=Path,
        default=ROOT / "reports/tables/phase0_reviewer_supplements/hard_negative_pair_bank.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/tables/phase0_independent_external_gold_v2",
    )
    parser.add_argument("--max-chembl-exact", type=int, default=1000)
    parser.add_argument("--max-chembl-parent", type=int, default=3000)
    parser.add_argument("--max-chembl-stereo", type=int, default=2000)
    parser.add_argument("--max-tautobase", type=int, default=1680)
    parser.add_argument("--tautobase-dir", type=Path, default=Path("external/tautobase"))
    parser.add_argument(
        "--openchemlib-node-modules",
        type=Path,
        default=Path("external/tautobase_tools/node_modules"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inchi_formula(value: str) -> str:
    parts = value.split("/")
    return parts[1] if len(parts) > 1 else ""


def _heavy_atom_count(formula: str) -> int:
    count = 0
    for symbol, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if symbol != "H":
            count += int(number or 1)
    return count


def independent_registry_negatives(db_path: Path, limit: int, seed: int = 20260910) -> pd.DataFrame:
    """Select obvious negatives using only ChEMBL registry identifiers and InChI formulae."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """
            select molregno, canonical_smiles, standard_inchi, standard_inchi_key
            from compound_structures
            where canonical_smiles is not null
              and standard_inchi is not null
              and standard_inchi_key is not null
              and length(standard_inchi_key) >= 14
              and (molregno % 997) < 3
            order by molregno
            """
        ).fetchall()
    finally:
        con.close()

    records = []
    for molregno, smiles, standard_inchi, inchikey in rows:
        formula = _inchi_formula(str(standard_inchi))
        heavy_atoms = _heavy_atom_count(formula)
        if formula and heavy_atoms >= 5:
            records.append((int(molregno), str(smiles), formula, str(inchikey), heavy_atoms))
    rng = random.Random(seed)
    rng.shuffle(records)

    selected = []
    used: set[int] = set()
    for i, left in enumerate(records):
        if left[0] in used:
            continue
        for right in records[i + 1 :]:
            if right[0] in used:
                continue
            # The negative label is fixed before ChemTrace evaluation. Different
            # registry connectivity blocks, formulae, and sizes make these clearly
            # non-identical records without invoking any proposed relation key.
            if left[3][:14] == right[3][:14] or left[2] == right[2]:
                continue
            if abs(left[4] - right[4]) < 10:
                continue
            selected.append(
                {
                    "pair_id": f"chembl_obvious_negative_{len(selected):04d}",
                    "query_id": left[0],
                    "target_id": right[0],
                    "query_smiles": left[1],
                    "target_smiles": right[1],
                    "query_inchi_formula": left[2],
                    "target_inchi_formula": right[2],
                    "query_heavy_atoms_from_formula": left[4],
                    "target_heavy_atoms_from_formula": right[4],
                    "selection_source": "ChEMBL36 Standard InChI formula/connectivity and registry IDs",
                }
            )
            used.update((left[0], right[0]))
            break
        if len(selected) >= limit:
            break

    audited = []
    for pair in selected:
        flags = relation_flags(pair["query_smiles"], pair["target_smiles"])
        audited.append({**pair, **{f"pred_{key}": value for key, value in flags.items()}})
    return pd.DataFrame(audited)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hard_negative_bank = pd.read_csv(args.hard_negative_bank)
    metrics = run_independent_relation_gold(args, args.out_dir, hard_negative_bank)
    independent_negatives = independent_registry_negatives(args.chembl_db, limit=500)
    negative_path = args.out_dir / "independent_registry_negative_pairs.csv"
    independent_negatives.to_csv(negative_path, index=False)
    typed_columns = ["pred_exact", "pred_parent", "pred_stereo", "pred_tautomer"]
    typed_calls = independent_negatives[typed_columns].astype(bool).any(axis=1)
    pd.DataFrame(
        [
            {
                "task": "independent_registry_obvious_negatives",
                "support": len(independent_negatives),
                "typed_certificate_calls": int(typed_calls.sum()),
                "typed_certificate_call_rate": float(typed_calls.mean()),
                "selection_used_chemtrace_key": False,
                "population_specificity_claim": False,
            }
        ]
    ).to_csv(args.out_dir / "independent_registry_negative_metrics.csv", index=False)
    pair_path = args.out_dir / "independent_relation_gold_pairs.csv"
    manifest = {
        "schema": "chemtrace.external_relation_gold.v2",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_boundary": {
            "stereo": (
                "same ChEMBL36 Standard InChI after removing /b,/t,/m,/s layers, "
                "different nonempty explicit stereo-layer signatures; no ChemTrace relation key used for inclusion"
            ),
            "parent": "ChEMBL36 molecule_hierarchy",
            "tautomer": "Tautobase curated pairs",
            "exact": "ChEMBL36 Standard InChIKey self-identity sanity set",
        },
        "chembl_db": str(args.chembl_db),
        "hard_negative_bank": str(args.hard_negative_bank),
        "hard_negative_bank_sha256": sha256(args.hard_negative_bank),
        "pairs_sha256": sha256(pair_path),
        "independent_registry_negatives_sha256": sha256(negative_path),
        "independent_registry_negative_support": len(independent_negatives),
        "independent_registry_negative_typed_calls": int(typed_calls.sum()),
        "supports": {
            str(row.relation): int(row.support)
            for row in metrics.itertuples(index=False)
            if int(row.support) > 0
        },
    }
    (args.out_dir / "external_relation_gold_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
