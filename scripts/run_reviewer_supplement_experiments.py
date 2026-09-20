#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, inchi, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.normalize import (
    canonicalize,
    parent_smiles,
    scaffold_smiles,
    stereo_stripped_smiles,
)

RDLogger.DisableLog("rdApp.*")

DEFAULT_CHEMBL_DB = Path("data/external/chembl_36.db")
DEFAULT_ZINC15 = Path("data/external/zinc15_250K.csv")

RELATIONS = ["exact", "parent", "stereo", "tautomer"]
PRIMARY_CLASSES = ["exact", "parent", "stereo", "tautomer", "family", "none"]
PREVALENCES = [0.001, 0.01, 0.05, 0.10]


@lru_cache(maxsize=1_000_000)
def tautomer_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return ""
    if len(smiles) > 220 or mol.GetNumHeavyAtoms() > 90:
        return ""
    try:
        enum = rdMolStandardize.TautomerEnumerator()
        enum.SetMaxTautomers(64)
        enum.SetMaxTransforms(200)
        taut = enum.Canonicalize(mol)
        return Chem.MolToSmiles(taut, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


@dataclass(frozen=True)
class MolRec:
    rec_id: str
    task: str
    smiles: str
    canonical: str
    parent: str
    stereo: str
    tautomer: str
    scaffold: str
    formula: str
    mw: float
    logp: float
    inchikey_full: str
    inchikey_connectivity: str
    fp: object


@dataclass(frozen=True)
class PairRow:
    pair_id: str
    task: str
    query_id: str
    target_id: str
    query_smiles: str
    target_smiles: str
    gold_relation: str
    binary_label: int
    gold_source: str
    source_type: str
    hard_negative_type: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run reviewer-requested supplement experiments: independent registry gold, hard negatives, "
            "external corpus exposure, equivalence/null analysis, path stability, and retrieval scale checks."
        )
    )
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0_tf_confirm_top_len158_extra"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_reviewer_supplements"))
    p.add_argument("--chembl-db", type=Path, default=DEFAULT_CHEMBL_DB)
    p.add_argument("--zinc15-csv", type=Path, default=DEFAULT_ZINC15)
    p.add_argument("--tautobase-dir", type=Path, default=Path("external/tautobase"))
    p.add_argument("--openchemlib-node-modules", type=Path, default=Path("external/tautobase_tools/node_modules"))
    p.add_argument("--max-chembl-parent", type=int, default=3000)
    p.add_argument("--max-chembl-stereo", type=int, default=2000)
    p.add_argument("--max-chembl-exact", type=int, default=1000)
    p.add_argument("--max-tautobase", type=int, default=1680)
    p.add_argument("--max-hard-clean", type=int, default=15000)
    p.add_argument("--max-hard-task-candidates", type=int, default=20000)
    p.add_argument("--max-hard-external-candidates", type=int, default=200000)
    p.add_argument("--max-hard-query-pool", type=int, default=6000)
    p.add_argument("--max-hard-per-type", type=int, default=600)
    p.add_argument("--hard-highsim-scan-top", type=int, default=3000)
    p.add_argument("--hard-highsim-per-query", type=int, default=5)
    p.add_argument("--max-concordance-pairs", type=int, default=700)
    p.add_argument("--max-exposure-corpus", type=int, default=200000)
    p.add_argument("--scale-sizes", type=int, nargs="+", default=[50000, 100000, 250000, 500000])
    p.add_argument("--scale-relation-measure-limit", type=int, default=100000)
    p.add_argument("--path-stability-n", type=int, default=50000)
    p.add_argument("--num-workers", type=int, default=0, help="0 means auto: use available CPUs while leaving 30 idle.")
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--fast", action="store_true", help="Use smaller limits for smoke/debug runs.")
    return p.parse_args()


def worker_count(requested: int) -> int:
    if requested and requested > 0:
        return requested
    total = os.cpu_count() or 1
    return max(1, min(96, total - 30))


def apply_fast_limits(args: argparse.Namespace) -> None:
    if not args.fast:
        args.num_workers = worker_count(args.num_workers)
        return
    args.max_chembl_parent = min(args.max_chembl_parent, 300)
    args.max_chembl_stereo = min(args.max_chembl_stereo, 250)
    args.max_chembl_exact = min(args.max_chembl_exact, 150)
    args.max_tautobase = min(args.max_tautobase, 200)
    args.max_hard_clean = min(args.max_hard_clean, 2500)
    args.max_hard_task_candidates = min(args.max_hard_task_candidates, 1500)
    args.max_hard_external_candidates = min(args.max_hard_external_candidates, 3000)
    args.max_hard_query_pool = min(args.max_hard_query_pool, 800)
    args.max_hard_per_type = min(args.max_hard_per_type, 120)
    args.hard_highsim_scan_top = min(args.hard_highsim_scan_top, 500)
    args.hard_highsim_per_query = min(args.hard_highsim_per_query, 3)
    args.max_concordance_pairs = min(args.max_concordance_pairs, 100)
    args.max_exposure_corpus = min(args.max_exposure_corpus, 5000)
    args.scale_sizes = [min(x, 10000) for x in args.scale_sizes[:2]]
    args.scale_relation_measure_limit = min(args.scale_relation_measure_limit, 5000)
    args.path_stability_n = min(args.path_stability_n, 5000)
    args.num_workers = min(worker_count(args.num_workers), 8)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_smiles_text(path: Path, limit: int | None = None) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            out.append(smi)
            if limit and len(out) >= limit:
                break
    return out


def read_smiles_csv(path: Path, limit: int | None = None) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    smiles_cols = [c for c in header.columns if "smiles" in c.lower()]
    if not smiles_cols:
        smiles_cols = [header.columns[0]]
    col = smiles_cols[0]
    chunks = pd.read_csv(path, usecols=[col], chunksize=100000)
    out: list[str] = []
    for chunk in chunks:
        out.extend(str(x) for x in chunk[col].dropna().tolist() if str(x).strip())
        if limit and len(out) >= limit:
            return out[:limit]
    return out


def chembl_smiles(db_path: Path, limit: int) -> list[str]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """
            select canonical_smiles
            from compound_structures
            where canonical_smiles is not null and length(canonical_smiles) > 0
            limit ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        con.close()
    return [str(r[0]) for r in rows if r and r[0]]


@lru_cache(maxsize=1_000_000)
def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=1_000_000)
def inchikey_full(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return ""


@lru_cache(maxsize=1_000_000)
def formula_for(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return ""


@lru_cache(maxsize=1_000_000)
def mw_for(smiles: str) -> float:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return float(Descriptors.MolWt(mol))
    except Exception:
        return float("nan")


@lru_cache(maxsize=1_000_000)
def logp_for(smiles: str) -> float:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return float("nan")


@lru_cache(maxsize=1_000_000)
def fp_for(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def ik_connectivity(key: str) -> str:
    return key.split("-")[0] if key else ""


def make_rec(rec_id: str, task: str, smiles: str) -> MolRec | None:
    can = canonicalize(smiles)
    if not can:
        return None
    ikey = inchikey_full(can)
    return MolRec(
        rec_id=rec_id,
        task=task,
        smiles=smiles,
        canonical=can,
        parent=parent_smiles(can),
        stereo=stereo_stripped_smiles(can),
        tautomer=tautomer_smiles(can),
        scaffold=scaffold_smiles(can),
        formula=formula_for(can),
        mw=mw_for(can),
        logp=logp_for(can),
        inchikey_full=ikey,
        inchikey_connectivity=ik_connectivity(ikey),
        fp=fp_for(can),
    )


def make_rec_from_tuple(item: tuple[str, str, str]) -> MolRec | None:
    return make_rec(item[0], item[1], item[2])


def pool_context():
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    return mp.get_context(method)


def parallel_map(func, items: list, num_workers: int, chunksize: int = 256) -> list:
    if num_workers <= 1 or len(items) < 1000:
        return [func(x) for x in items]
    ctx = pool_context()
    with ctx.Pool(processes=num_workers) as pool:
        return list(pool.imap(func, items, chunksize=chunksize))


def relation_flags(a: str, b: str) -> dict[str, bool]:
    ca = canonicalize(a)
    cb = canonicalize(b)
    if not ca or not cb:
        return {k: False for k in RELATIONS + ["family"]}
    out = {
        "exact": ca == cb,
        "parent": parent_smiles(ca) == parent_smiles(cb),
        "stereo": stereo_stripped_smiles(ca) == stereo_stripped_smiles(cb),
        "tautomer": tautomer_smiles(ca) == tautomer_smiles(cb),
        "family": bool(scaffold_smiles(ca) and scaffold_smiles(ca) == scaffold_smiles(cb)),
    }
    return out


def primary_relation(a: str, b: str) -> str:
    flags = relation_flags(a, b)
    for rel in PRIMARY_CLASSES[:-1]:
        if flags.get(rel, False):
            return rel
    return "none"


def no_typed_relation(a: MolRec, b: MolRec) -> bool:
    return not (
        a.canonical == b.canonical
        or (a.parent and a.parent == b.parent)
        or (a.stereo and a.stereo == b.stereo)
        or (a.tautomer and a.tautomer == b.tautomer)
    )


def tanimoto(a: MolRec, b: MolRec) -> float:
    if a.fp is None or b.fp is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(a.fp, b.fp))


def load_affected(artifact_root: Path) -> pd.DataFrame:
    path = artifact_root / "phase0_affected.csv"
    df = pd.read_csv(path)
    if "canonical" not in df.columns:
        df["canonical"] = df["smiles"].map(canonicalize)
    return df


def split_files(artifact_root: Path) -> list[Path]:
    return sorted((artifact_root / "splits").glob("*.csv"))


def load_task_queries(artifact_root: Path) -> pd.DataFrame:
    rows = []
    for path in split_files(artifact_root):
        df = pd.read_csv(path)
        smiles_col = "canonical" if "canonical" in df.columns else "smiles"
        for row in df.itertuples(index=False):
            sample_id = str(getattr(row, "sample_id", f"{path.stem}_{len(rows)}"))
            rows.append(
                {
                    "task": str(getattr(row, "task", path.stem)),
                    "sample_id": sample_id,
                    "split": str(getattr(row, "split", "")),
                    "y": str(getattr(row, "y", "")),
                    "smiles": str(getattr(row, smiles_col)),
                }
            )
    return pd.DataFrame(rows)


def load_manifest_positive_pairs(artifact_root: Path, affected: pd.DataFrame) -> list[PairRow]:
    by_sid = {str(r.sample_id): (str(r.task), str(r.canonical)) for r in affected.itertuples(index=False)}
    rows: list[PairRow] = []
    for relation in ["parent", "stereo", "tautomer"]:
        candidates = sorted((artifact_root / "manifests").glob(f"{relation}_*.jsonl"))
        if not candidates:
            continue
        # Use the smallest dose manifest to avoid repeating the same query many times.
        candidates.sort(key=lambda p: int(p.stem.split("_")[-1].replace("x", "")))
        manifest_path = candidates[0]
        manifest = read_jsonl(manifest_path)
        seen: set[tuple[str, str]] = set()
        for row in manifest.itertuples(index=False):
            sid = str(row.sample_id)
            target = str(row.injected_smiles)
            if sid not in by_sid:
                continue
            task, query = by_sid[sid]
            key = (sid, canonicalize(target))
            if key in seen:
                continue
            seen.add(key)
            if canonicalize(query) == canonicalize(target):
                continue
            rows.append(
                PairRow(
                    pair_id=f"controlled_{relation}_{len(rows):06d}",
                    task=task,
                    query_id=sid,
                    target_id=f"{manifest_path.stem}:{len(rows)}",
                    query_smiles=query,
                    target_smiles=target,
                    gold_relation=relation,
                    binary_label=1,
                    gold_source=str(manifest_path),
                    source_type="self_rerun_controlled_injection",
                    hard_negative_type="positive_controlled_relation",
                )
            )
    return rows


def load_clean_recs(artifact_root: Path, limit: int, num_workers: int) -> list[MolRec]:
    path = artifact_root / "corpora" / "clean.txt"
    smiles = read_smiles_text(path, limit=limit)
    items = [(f"clean_{idx:07d}", "clean_corpus", smi) for idx, smi in enumerate(smiles)]
    return [rec for rec in parallel_map(make_rec_from_tuple, items, num_workers, chunksize=256) if rec is not None]


def load_task_candidate_recs(
    artifact_root: Path, limit: int, seed: int, num_workers: int
) -> tuple[list[MolRec], dict[str, str]]:
    task_queries = load_task_queries(artifact_root)
    if limit > 0 and len(task_queries) > limit:
        rng = np.random.default_rng(seed)
        per_task = max(1, limit // max(task_queries["task"].nunique(), 1))
        pieces = []
        used = set()
        for _, group in task_queries.groupby("task"):
            take = min(per_task, len(group))
            sample = group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
            pieces.append(sample)
            used.update(sample.index.tolist())
        if sum(len(x) for x in pieces) < limit:
            remaining = task_queries.drop(index=list(used))
            take = min(limit - sum(len(x) for x in pieces), len(remaining))
            if take > 0:
                pieces.append(remaining.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        task_queries = pd.concat(pieces, ignore_index=True)
    labels = {str(r.sample_id): str(r.y) for r in task_queries.itertuples(index=False)}
    items = [(str(r.sample_id), str(r.task), str(r.smiles)) for r in task_queries.itertuples(index=False)]
    recs = [rec for rec in parallel_map(make_rec_from_tuple, items, num_workers, chunksize=256) if rec is not None]
    return recs, labels


def load_external_hard_recs(args: argparse.Namespace, num_workers: int) -> list[MolRec]:
    limit = max(0, int(args.max_hard_external_candidates))
    if limit <= 0:
        return []
    chembl_limit = int(limit * 0.7)
    zinc_limit = limit - chembl_limit
    items: list[tuple[str, str, str]] = []
    for idx, smi in enumerate(chembl_smiles(args.chembl_db, chembl_limit)):
        items.append((f"chembl36_hard_{idx:07d}", "chembl36_external", smi))
    if args.zinc15_csv.exists() and zinc_limit > 0:
        for idx, smi in enumerate(read_smiles_csv(args.zinc15_csv, limit=zinc_limit)):
            items.append((f"zinc15_hard_{idx:07d}", "zinc15_external", smi))
    return [rec for rec in parallel_map(make_rec_from_tuple, items, num_workers, chunksize=512) if rec is not None]


def affected_recs(affected: pd.DataFrame) -> list[MolRec]:
    recs = []
    for row in affected.itertuples(index=False):
        rec = make_rec(str(row.sample_id), str(row.task), str(row.canonical))
        if rec is not None:
            recs.append(rec)
    return recs


def add_pair(
    out: list[PairRow],
    seen: set[tuple[str, str, str]],
    query: MolRec,
    target: MolRec,
    neg_type: str,
    source: str,
) -> bool:
    if not no_typed_relation(query, target):
        return False
    key = (query.rec_id, target.rec_id, neg_type)
    if key in seen:
        return False
    seen.add(key)
    out.append(
        PairRow(
            pair_id=f"hardneg_{len(out):06d}",
            task=query.task,
            query_id=query.rec_id,
            target_id=target.rec_id,
            query_smiles=query.canonical,
            target_smiles=target.canonical,
            gold_relation="none",
            binary_label=0,
            gold_source=source,
            source_type="self_rerun_hard_negative",
            hard_negative_type=neg_type,
        )
    )
    return True


def build_hard_negative_pairs(
    queries: list[MolRec],
    candidates: list[MolRec],
    max_per_type: int,
    seed: int,
    labels: dict[str, str],
    highsim_scan_top: int,
    highsim_per_query: int,
) -> list[PairRow]:
    rng = np.random.default_rng(seed)
    out: list[PairRow] = []
    seen: set[tuple[str, str, str]] = set()
    clean_by_scaffold: dict[str, list[MolRec]] = defaultdict(list)
    clean_by_formula: dict[str, list[MolRec]] = defaultdict(list)
    candidates_by_task: dict[str, list[MolRec]] = defaultdict(list)
    candidate_indices_by_scaffold: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(candidates):
        if rec.scaffold:
            clean_by_scaffold[rec.scaffold].append(rec)
            candidate_indices_by_scaffold[rec.scaffold].append(idx)
        if rec.formula:
            clean_by_formula[rec.formula].append(rec)
        candidates_by_task[rec.task].append(rec)

    shuffled_queries = list(queries)
    rng.shuffle(shuffled_queries)

    def shuffled(values: list[MolRec]) -> list[MolRec]:
        if len(values) <= 1:
            return values
        order = rng.permutation(len(values))
        return [values[int(i)] for i in order]

    counters = Counter()
    for q in shuffled_queries:
        if counters["same_murcko_scaffold"] < max_per_type and q.scaffold in clean_by_scaffold:
            for c in shuffled(clean_by_scaffold[q.scaffold])[:50]:
                if add_pair(out, seen, q, c, "same_murcko_scaffold", "phase0_clean_corpus"):
                    counters["same_murcko_scaffold"] += 1
                    break

        if counters["same_formula_constitutional_isomer"] < max_per_type and q.formula in clean_by_formula:
            for c in shuffled(clean_by_formula[q.formula])[:80]:
                if q.inchikey_connectivity and q.inchikey_connectivity == c.inchikey_connectivity:
                    continue
                if add_pair(out, seen, q, c, "same_formula_constitutional_isomer", "phase0_clean_corpus"):
                    counters["same_formula_constitutional_isomer"] += 1
                    break

        if counters["same_scaffold_mw_logp_matched"] < max_per_type and q.scaffold in clean_by_scaffold:
            for c in shuffled(clean_by_scaffold[q.scaffold])[:100]:
                if math.isnan(q.mw) or math.isnan(c.mw) or math.isnan(q.logp) or math.isnan(c.logp):
                    continue
                if abs(q.mw - c.mw) > 25.0 or abs(q.logp - c.logp) > 1.0:
                    continue
                if add_pair(out, seen, q, c, "same_scaffold_mw_logp_matched", "phase0_clean_corpus"):
                    counters["same_scaffold_mw_logp_matched"] += 1
                    break

    candidate_fps = [(idx, c.fp) for idx, c in enumerate(candidates) if c.fp is not None]
    scaffold_fps = {
        scaffold: [(idx, candidates[idx].fp) for idx in indices if candidates[idx].fp is not None]
        for scaffold, indices in candidate_indices_by_scaffold.items()
    }
    for q in shuffled_queries:
        if q.fp is None:
            continue
        q_label = labels.get(q.rec_id, "")
        if counters["activity_cliff_same_task_opposite_label"] < max_per_type and q_label not in {"", "nan"}:
            same_task = [c for c in candidates_by_task.get(q.task, []) if labels.get(c.rec_id, "") not in {"", "nan", q_label}]
            same_task_fps = [(idx, c.fp) for idx, c in enumerate(same_task) if c.fp is not None]
            if same_task_fps:
                same_sims = DataStructs.BulkTanimotoSimilarity(q.fp, [fp for _, fp in same_task_fps])
                for pos in np.argsort(np.asarray(same_sims))[::-1][:100]:
                    c = same_task[same_task_fps[int(pos)][0]]
                    if same_sims[int(pos)] < 0.70 and q.scaffold != c.scaffold:
                        continue
                    if add_pair(out, seen, q, c, "activity_cliff_same_task_opposite_label", "phase0_same_benchmark_split"):
                        counters["activity_cliff_same_task_opposite_label"] += 1
                        break

        local_fps = scaffold_fps.get(q.scaffold, []) if q.scaffold else []
        # High-similarity chemistry negatives are most efficiently mined from same-scaffold buckets.
        # A small global fallback keeps acyclic/no-scaffold molecules from being skipped entirely.
        scan_fps = local_fps if len(local_fps) >= 2 else candidate_fps[: min(len(candidate_fps), max(highsim_scan_top, 20000))]
        sims = DataStructs.BulkTanimotoSimilarity(q.fp, [fp for _, fp in scan_fps])
        if not sims:
            continue
        order = np.argsort(np.asarray(sims))[::-1]
        for neg_type, threshold in [("ecfp_tanimoto_ge_0_90", 0.90), ("ecfp_tanimoto_ge_0_80", 0.80)]:
            if counters[neg_type] >= max_per_type:
                continue
            added_for_query = 0
            for pos in order[:highsim_scan_top]:
                if sims[int(pos)] < threshold:
                    break
                c = candidates[scan_fps[int(pos)][0]]
                source = (
                    "external_registry_and_corpus_pool"
                    if c.task.endswith("_external") or q.task.endswith("_external")
                    else "phase0_clean_and_same_benchmark_pool"
                )
                if add_pair(out, seen, q, c, neg_type, source):
                    counters[neg_type] += 1
                    added_for_query += 1
                    if counters[neg_type] >= max_per_type or added_for_query >= max(1, highsim_per_query):
                        break
        if counters["nearest_neighbor_same_benchmark"] >= max_per_type:
            continue
        same_task_indices = [
            pos
            for pos in order[:500]
            if candidates[scan_fps[int(pos)][0]].task in {q.task, "clean_corpus", "chembl36_external", "zinc15_external"}
        ]
        for pos in same_task_indices[:100]:
            c = candidates[scan_fps[int(pos)][0]]
            source = (
                "external_registry_and_corpus_pool"
                if c.task.endswith("_external") or q.task.endswith("_external")
                else "phase0_clean_and_same_benchmark_pool"
            )
            if add_pair(out, seen, q, c, "nearest_neighbor_same_benchmark_or_clean", source):
                counters["nearest_neighbor_same_benchmark"] += 1
                break

    return out


def score_pair(method: str, q: str, t: str) -> float:
    ca = canonicalize(q)
    cb = canonicalize(t)
    if not ca or not cb:
        return 0.0
    if method == "raw_smiles_identity":
        return 1.0 if q == t else 0.0
    if method == "canonical_smiles_identity":
        return 1.0 if ca == cb else 0.0
    if method == "inchikey_full_identity":
        return 1.0 if inchikey_full(ca) and inchikey_full(ca) == inchikey_full(cb) else 0.0
    if method == "inchikey_connectivity_identity":
        return 1.0 if ik_connectivity(inchikey_full(ca)) and ik_connectivity(inchikey_full(ca)) == ik_connectivity(inchikey_full(cb)) else 0.0
    if method == "parent_key_identity":
        return 1.0 if parent_smiles(ca) and parent_smiles(ca) == parent_smiles(cb) else 0.0
    if method == "stereo_stripped_identity":
        return 1.0 if stereo_stripped_smiles(ca) and stereo_stripped_smiles(ca) == stereo_stripped_smiles(cb) else 0.0
    if method == "tautomer_identity":
        return 1.0 if tautomer_smiles(ca) and tautomer_smiles(ca) == tautomer_smiles(cb) else 0.0
    if method == "chemtrace_typed_certificate":
        flags = relation_flags(ca, cb)
        return 1.0 if any(flags[k] for k in RELATIONS) else 0.0
    if method == "murcko_scaffold":
        return 1.0 if scaffold_smiles(ca) and scaffold_smiles(ca) == scaffold_smiles(cb) else 0.0
    if method in {"morgan_tanimoto", "morgan_ecfp80", "morgan_ecfp90", "datasail_similarity_cluster", "lohi_high_similarity_split"}:
        fa = fp_for(ca)
        fb = fp_for(cb)
        sim = float(DataStructs.TanimotoSimilarity(fa, fb)) if fa is not None and fb is not None else 0.0
        if method == "morgan_tanimoto":
            return sim
        if method == "morgan_ecfp80":
            return 1.0 if sim >= 0.80 else 0.0
        if method == "morgan_ecfp90":
            return 1.0 if sim >= 0.90 else 0.0
        if method == "datasail_similarity_cluster":
            same_scaffold = scaffold_smiles(ca) and scaffold_smiles(ca) == scaffold_smiles(cb)
            return 1.0 if same_scaffold or sim >= 0.70 else 0.0
        if method == "lohi_high_similarity_split":
            return 1.0 if sim >= 0.60 else 0.0
    if method == "chemical_science_audit_bundle":
        return 1.0 if (
            score_pair("canonical_smiles_identity", ca, cb)
            or score_pair("inchikey_connectivity_identity", ca, cb)
            or score_pair("murcko_scaffold", ca, cb)
            or score_pair("morgan_ecfp80", ca, cb)
        ) else 0.0
    if method == "tdc_polaris_curation_bundle":
        return 1.0 if (
            score_pair("canonical_smiles_identity", ca, cb)
            or score_pair("inchikey_full_identity", ca, cb)
            or score_pair("murcko_scaffold", ca, cb)
            or score_pair("morgan_ecfp80", ca, cb)
        ) else 0.0
    raise KeyError(method)


def method_registry() -> pd.DataFrame:
    rows = [
        ("raw_smiles_identity", "classic", "self_rerun", "raw string equality"),
        ("canonical_smiles_identity", "classic", "self_rerun", "RDKit canonical SMILES equality"),
        ("inchikey_full_identity", "direct_mechanism", "self_rerun", "standard InChIKey full equality"),
        ("inchikey_connectivity_identity", "direct_mechanism", "self_rerun", "standard InChIKey first-block equality"),
        ("parent_key_identity", "direct_mechanism", "self_rerun", "largest-fragment/uncharged parent equality"),
        ("stereo_stripped_identity", "direct_mechanism", "self_rerun", "stereochemistry-stripped equality"),
        ("tautomer_identity", "direct_mechanism", "self_rerun", "canonical tautomer equality"),
        ("morgan_tanimoto", "similarity", "self_rerun", "continuous Morgan ECFP Tanimoto score"),
        ("morgan_ecfp80", "similarity", "self_rerun", "Morgan ECFP Tanimoto >= 0.80"),
        ("morgan_ecfp90", "similarity", "self_rerun", "Morgan ECFP Tanimoto >= 0.90"),
        ("murcko_scaffold", "split_control", "self_rerun", "Bemis-Murcko scaffold equality"),
        ("datasail_similarity_cluster", "split_control", "adapted_rerun", "DataSAIL-style scaffold or ECFP >= 0.70"),
        ("lohi_high_similarity_split", "split_control", "adapted_rerun", "LoHi-style ECFP >= 0.60"),
        ("chemical_science_audit_bundle", "benchmark_audit_bundle", "adapted_rerun", "exact/InChI/scaffold/ECFP80 bundle"),
        ("tdc_polaris_curation_bundle", "benchmark_audit_bundle", "adapted_rerun", "canonical/full InChI/scaffold/ECFP80 bundle"),
        ("chemtrace_typed_certificate", "proposed", "self_rerun", "exact/parent/stereo/tautomer typed certificate"),
    ]
    return pd.DataFrame(rows, columns=["method", "method_group", "source_type", "operational_definition"])


def roc_auc_score_np(y: np.ndarray, scores: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def average_precision_np(y: np.ndarray, scores: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def threshold_for_recall(y: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> tuple[float, float, float, float]:
    y = y.astype(int)
    thresholds = sorted(set(float(x) for x in scores), reverse=True)
    thresholds.append(float("-inf"))
    best = None
    for thr in thresholds:
        pred = scores >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        if not math.isnan(tpr) and tpr >= target_recall:
            cand = (fpr, -precision if not math.isnan(precision) else 0.0, -thr, thr, tpr, precision)
            if best is None or cand < best:
                best = cand
    if best is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(best[3]), float(best[4]), float(best[0]), float(best[5])


def ppv_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    if math.isnan(tpr) or math.isnan(fpr):
        return float("nan")
    denom = prevalence * tpr + (1.0 - prevalence) * fpr
    if denom == 0:
        return 1.0 if tpr > 0 else float("nan")
    return prevalence * tpr / denom


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def zero_event_rule_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return float("nan")
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    return wilson_ci(k, n)[1]


def binary_count_stats(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = y.astype(int)
    pred = pred.astype(bool)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    low, high = wilson_ci(fp, fp + tn)
    return {
        "tp_default": tp,
        "fp_default": fp,
        "fn_default": fn,
        "tn_default": tn,
        "recall_default": recall,
        "precision_default": precision,
        "fpr_default": fpr,
        "fpr_default_wilson95_low": low,
        "fpr_default_wilson95_high": high,
        "fpr_default_one_sided95_upper": zero_event_rule_upper(fp, fp + tn),
    }


def run_hard_negative_retrieval(
    args: argparse.Namespace,
) -> pd.DataFrame:
    artifact_root = args.artifact_root
    out_dir = args.out_dir
    max_clean = args.max_hard_clean
    max_task_candidates = args.max_hard_task_candidates
    max_per_type = args.max_hard_per_type
    seed = args.seed
    num_workers = args.num_workers
    affected = load_affected(artifact_root)
    positives = load_manifest_positive_pairs(artifact_root, affected)
    queries = affected_recs(affected)
    clean = load_clean_recs(artifact_root, limit=max_clean, num_workers=num_workers)
    task_candidates, label_map = load_task_candidate_recs(
        artifact_root, limit=max_task_candidates, seed=seed, num_workers=num_workers
    )
    label_map.update({str(r.sample_id): str(r.y) for r in affected.itertuples(index=False)})
    external_candidates = load_external_hard_recs(args, num_workers=num_workers)
    candidates = clean + task_candidates + external_candidates
    query_pool = list(queries)
    extra_queries = task_candidates + external_candidates
    if len(query_pool) < args.max_hard_query_pool and extra_queries:
        rng = np.random.default_rng(seed)
        take = min(args.max_hard_query_pool - len(query_pool), len(extra_queries))
        idx = rng.choice(len(extra_queries), size=take, replace=False)
        query_pool.extend(extra_queries[int(i)] for i in idx)
    negatives = build_hard_negative_pairs(
        query_pool,
        candidates,
        max_per_type=max_per_type,
        seed=seed,
        labels=label_map,
        highsim_scan_top=args.hard_highsim_scan_top,
        highsim_per_query=args.hard_highsim_per_query,
    )
    pairs = positives + negatives
    pair_df = pd.DataFrame([p.__dict__ for p in pairs])
    pair_df.to_csv(out_dir / "hard_negative_pair_bank.csv", index=False)

    registry = method_registry()
    score_rows = []
    for method in registry["method"].tolist():
        scores = [score_pair(method, p.query_smiles, p.target_smiles) for p in pairs]
        for p, score in zip(pairs, scores):
            score_rows.append(
                {
                    "pair_id": p.pair_id,
                    "method": method,
                    "score": score,
                    "binary_label": p.binary_label,
                    "gold_relation": p.gold_relation,
                    "hard_negative_type": p.hard_negative_type,
                }
            )
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(out_dir / "hard_negative_pair_scores.csv", index=False)

    rows = []
    for method, sdf in scores_df.groupby("method"):
        y = sdf["binary_label"].to_numpy(dtype=int)
        s = sdf["score"].to_numpy(dtype=float)
        thr, tpr, fpr, precision = threshold_for_recall(y, s, 0.95)
        default_stats = binary_count_stats(y, s >= 1.0)
        row = {
            "method": method,
            "n_pairs": len(sdf),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "auroc": roc_auc_score_np(y, s),
            "auprc": average_precision_np(y, s),
            "threshold_at_recall_ge_0_95": thr,
            "recall_at_threshold": tpr,
            "fpr_at_95_tpr": fpr,
            "precision_at_95_recall": precision,
        }
        row.update(default_stats)
        for prev in PREVALENCES:
            row[f"ppv_at_prevalence_{prev:g}"] = ppv_at_prevalence(tpr, fpr, prev)
            row[f"ppv_empirical_default_at_prevalence_{prev:g}"] = ppv_at_prevalence(
                default_stats["recall_default"], default_stats["fpr_default"], prev
            )
            row[f"ppv_default_fpr_upper95_at_prevalence_{prev:g}"] = ppv_at_prevalence(
                default_stats["recall_default"], default_stats["fpr_default_one_sided95_upper"], prev
            )
        rows.append(row)
    summary = pd.DataFrame(rows).merge(registry, on="method", how="left")
    summary = summary.sort_values(["auprc", "auroc"], ascending=False)
    summary.to_csv(out_dir / "hard_negative_retrieval_metrics.csv", index=False)

    scenario_rows = []
    for (method, neg_type), sdf in scores_df[scores_df["binary_label"] == 0].groupby(["method", "hard_negative_type"]):
        fp_default = int((sdf["score"] >= 1.0).sum())
        n_negative = len(sdf)
        low, high = wilson_ci(fp_default, n_negative)
        scenario_rows.append(
            {
                "method": method,
                "hard_negative_type": neg_type,
                "n_negative": n_negative,
                "fp_default": fp_default,
                "mean_score": float(sdf["score"].mean()),
                "false_positive_rate_at_default_binary": fp_default / n_negative if n_negative else float("nan"),
                "fpr_default_wilson95_low": low,
                "fpr_default_wilson95_high": high,
                "fpr_default_one_sided95_upper": zero_event_rule_upper(fp_default, n_negative),
                "share_score_ge_0_80": float((sdf["score"] >= 0.80).mean()),
                "share_score_ge_0_90": float((sdf["score"] >= 0.90).mean()),
            }
        )
    scenario = pd.DataFrame(scenario_rows)
    scenario.to_csv(out_dir / "hard_negative_scenario_breakdown.csv", index=False)
    return summary


def chembl_parent_gold(db_path: Path, limit: int) -> list[PairRow]:
    if not db_path.exists() or limit <= 0:
        return []
    con = sqlite3.connect(str(db_path))
    rows: list[PairRow] = []
    try:
        cur = con.execute(
            """
            select h.molregno, h.parent_molregno, c.canonical_smiles, p.canonical_smiles,
                   c.standard_inchi_key, p.standard_inchi_key
            from molecule_hierarchy h
            join compound_structures c on c.molregno = h.molregno
            join compound_structures p on p.molregno = h.parent_molregno
            where h.parent_molregno is not null
              and h.molregno != h.parent_molregno
              and c.canonical_smiles is not null
              and p.canonical_smiles is not null
            limit ?
            """,
            (limit * 4,),
        )
        for molregno, parent_molregno, smi, psmi, ik, pik in cur:
            if len(rows) >= limit:
                break
            if canonicalize(smi) == canonicalize(psmi):
                continue
            rows.append(
                PairRow(
                    pair_id=f"chembl_parent_{len(rows):06d}",
                    task="chembl36_registry",
                    query_id=str(molregno),
                    target_id=str(parent_molregno),
                    query_smiles=str(smi),
                    target_smiles=str(psmi),
                    gold_relation="parent",
                    binary_label=1,
                    gold_source="ChEMBL36 molecule_hierarchy",
                    source_type="external_registry_gold",
                    hard_negative_type="positive_parent_registry",
                )
            )
    finally:
        con.close()
    return rows


def chembl_structure_rows(db_path: Path, scan_limit: int = 500000) -> list[tuple[int, str, str]]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(
            """
            select molregno, canonical_smiles, standard_inchi_key
            from compound_structures
            where canonical_smiles is not null
              and standard_inchi_key is not null
              and length(standard_inchi_key) >= 14
            limit ?
            """,
            (scan_limit,),
        )
        return [(int(a), str(b), str(c)) for a, b, c in cur]
    finally:
        con.close()


def inchi_stereo_signature(value: str) -> tuple[str, ...]:
    """Return explicit Standard InChI stereo layers without using ChemTrace keys."""
    if not value or not value.startswith("InChI="):
        return ()
    return tuple(layer for layer in value.split("/")[1:] if layer and layer[0] in {"b", "t", "m", "s"})


def inchi_without_stereo(value: str) -> str:
    """Remove Standard InChI stereo layers while retaining every other layer."""
    if not value or not value.startswith("InChI="):
        return ""
    parts = value.split("/")
    return "/".join([parts[0]] + [layer for layer in parts[1:] if not layer or layer[0] not in {"b", "t", "m", "s"}])


def chembl_inchi_rows(
    db_path: Path, scan_limit: int = 1000000
) -> list[tuple[int, str, str, str]]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(
            """
            select molregno, canonical_smiles, standard_inchi, standard_inchi_key
            from compound_structures
            where canonical_smiles is not null
              and standard_inchi is not null
              and standard_inchi_key is not null
              and length(standard_inchi_key) >= 14
              and (standard_inchi like '%/b%' or standard_inchi like '%/t%')
            order by molregno
            limit ?
            """,
            (scan_limit,),
        )
        return [(int(a), str(b), str(c), str(d)) for a, b, c, d in cur]
    finally:
        con.close()


def chembl_stereo_gold(db_path: Path, limit: int) -> list[PairRow]:
    # Candidate inclusion is based exclusively on registry-provided Standard InChI
    # layers. ChemTrace/RDKit relation keys are evaluated only after this set is frozen.
    rows = chembl_inchi_rows(db_path, scan_limit=max(1000000, limit * 500))
    by_nonstereo_inchi: dict[str, list[tuple[int, str, str, str, tuple[str, ...]]]] = defaultdict(list)
    for molregno, smi, standard_inchi, ikey in rows:
        signature = inchi_stereo_signature(standard_inchi)
        nonstereo = inchi_without_stereo(standard_inchi)
        if signature and nonstereo:
            by_nonstereo_inchi[nonstereo].append((molregno, smi, standard_inchi, ikey, signature))
    out: list[PairRow] = []
    for nonstereo_inchi in sorted(by_nonstereo_inchi):
        group = by_nonstereo_inchi[nonstereo_inchi]
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: (x[4], x[0]))
        for i in range(len(group) - 1):
            a = group[i]
            b = group[i + 1]
            if a[4] == b[4]:
                continue
            if canonicalize(a[1]) == canonicalize(b[1]):
                continue
            out.append(
                PairRow(
                    pair_id=f"chembl_stereo_{len(out):06d}",
                    task="chembl36_registry",
                    query_id=str(a[0]),
                    target_id=str(b[0]),
                    query_smiles=a[1],
                    target_smiles=b[1],
                    gold_relation="stereo",
                    binary_label=1,
                    gold_source="ChEMBL36 Standard InChI non-stereo identity with differing explicit stereo layers",
                    source_type="external_registry_gold_inchi_stereo_layers",
                    hard_negative_type="positive_stereo_registry",
                )
            )
            if len(out) >= limit:
                return out
    return out


def chembl_exact_gold(db_path: Path, limit: int) -> list[PairRow]:
    rows = chembl_structure_rows(db_path, scan_limit=max(200000, limit * 500))
    by_full: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for molregno, smi, ikey in rows:
        by_full[ikey].append((molregno, smi, ikey))
    out: list[PairRow] = []
    for group in by_full.values():
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            a = group[i]
            b = group[i + 1]
            if canonicalize(a[1]) != canonicalize(b[1]):
                continue
            out.append(
                PairRow(
                    pair_id=f"chembl_exact_{len(out):06d}",
                    task="chembl36_registry",
                    query_id=str(a[0]),
                    target_id=str(b[0]),
                    query_smiles=a[1],
                    target_smiles=b[1],
                    gold_relation="exact",
                    binary_label=1,
                    gold_source="ChEMBL36 duplicate standard_inchi_key",
                    source_type="external_registry_gold",
                    hard_negative_type="positive_exact_registry",
                )
            )
            if len(out) >= limit:
                return out
    if not out:
        # ChEMBL de-duplicates most exact structures. These rows still test an independent
        # standard-InChI identity source while marking the source precisely.
        for molregno, smi, ikey in rows[:limit]:
            out.append(
                PairRow(
                    pair_id=f"chembl_exact_self_{len(out):06d}",
                    task="chembl36_registry",
                    query_id=str(molregno),
                    target_id=str(molregno),
                    query_smiles=smi,
                    target_smiles=smi,
                    gold_relation="exact",
                    binary_label=1,
                    gold_source="ChEMBL36 standard_inchi_key self-identity",
                    source_type="external_registry_identity_sanity_check",
                    hard_negative_type="positive_exact_identity",
                )
            )
            if len(out) >= limit:
                break
    return out


def tautobase_pairs(root: Path, tautobase_dir: Path, node_modules: Path, limit: int) -> list[PairRow]:
    dwar = root / tautobase_dir / "Tautobase.dwar"
    module_dir = root / node_modules
    if not dwar.exists() or not module_dir.exists() or limit <= 0:
        return []
    js = r"""
const fs = require('fs');
const OCL = require('openchemlib');
const path = process.argv[1];
const limit = Number(process.argv[2]);
const lines = fs.readFileSync(path, 'utf8').split(/\r?\n/);
const headerIndex = lines.findIndex(line => line.startsWith('idcoordinates2D\tFragFp'));
if (headerIndex < 0) process.exit(2);
const header = lines[headerIndex].split('\t');
const i1 = header.indexOf('Tautomer1');
const i2 = header.indexOf('Tautomer2');
const iRef = header.indexOf('Reference');
const iRow = header.indexOf('Row Nr');
let n = 0;
console.log(['row_nr', 'reference', 'smiles1', 'smiles2'].join('\t'));
for (let i = headerIndex + 1; i < lines.length && n < limit; i++) {
  const line = lines[i];
  if (!line || line.startsWith('<')) continue;
  const cells = line.split('\t');
  try {
    const m1 = OCL.Molecule.fromIDCode(cells[i1]);
    const m2 = OCL.Molecule.fromIDCode(cells[i2]);
    const s1 = m1.toSmiles();
    const s2 = m2.toSmiles();
    if (s1 && s2) {
      console.log([cells[iRow] || String(i), cells[iRef] || '', s1, s2].join('\t'));
      n++;
    }
  } catch (err) {}
}
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(module_dir)
    proc = subprocess.run(
        ["node", "-e", js, str(dwar), str(limit)],
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    df = pd.read_csv(io.StringIO(proc.stdout), sep="\t")
    out: list[PairRow] = []
    for row in df.itertuples(index=False):
        s1 = str(row.smiles1)
        s2 = str(row.smiles2)
        if not canonicalize(s1) or not canonicalize(s2):
            continue
        if canonicalize(s1) == canonicalize(s2):
            continue
        out.append(
            PairRow(
                pair_id=f"tautobase_{len(out):06d}",
                task="tautobase",
                query_id=str(row.row_nr),
                target_id=str(row.row_nr),
                query_smiles=s1,
                target_smiles=s2,
                gold_relation="tautomer",
                binary_label=1,
                gold_source="Tautobase DataWarrior idcode converted by OpenChemLib",
                source_type="external_curated_gold",
                hard_negative_type="positive_tautomer_curated",
            )
        )
    return out


def run_independent_relation_gold(
    args: argparse.Namespace, out_dir: Path, hard_negative_bank: pd.DataFrame
) -> pd.DataFrame:
    rows: list[PairRow] = []
    rows.extend(chembl_exact_gold(args.chembl_db, args.max_chembl_exact))
    rows.extend(chembl_parent_gold(args.chembl_db, args.max_chembl_parent))
    rows.extend(chembl_stereo_gold(args.chembl_db, args.max_chembl_stereo))
    rows.extend(tautobase_pairs(ROOT, args.tautobase_dir, args.openchemlib_node_modules, args.max_tautobase))

    neg_df = hard_negative_bank[hard_negative_bank["binary_label"] == 0].copy()
    neg_df = neg_df.head(max(args.max_chembl_exact, 500))
    for row in neg_df.itertuples(index=False):
        rows.append(
            PairRow(
                pair_id=f"gold_none_{len(rows):06d}",
                task=str(row.task),
                query_id=str(row.query_id),
                target_id=str(row.target_id),
                query_smiles=str(row.query_smiles),
                target_smiles=str(row.target_smiles),
                gold_relation="none",
                binary_label=0,
                gold_source=str(row.gold_source),
                source_type="hard_negative_gold",
                hard_negative_type=str(row.hard_negative_type),
            )
        )

    gold = pd.DataFrame([r.__dict__ for r in rows])
    pred_rows = []
    for row in gold.itertuples(index=False):
        flags = relation_flags(str(row.query_smiles), str(row.target_smiles))
        pred = primary_relation(str(row.query_smiles), str(row.target_smiles))
        pred_rows.append(
            {
                "pair_id": row.pair_id,
                "gold_relation": row.gold_relation,
                "pred_relation": pred,
                "pred_exact": flags["exact"],
                "pred_parent": flags["parent"],
                "pred_stereo": flags["stereo"],
                "pred_tautomer": flags["tautomer"],
                "pred_family": flags["family"],
            }
        )
    pred = pd.DataFrame(pred_rows)
    audited = gold.merge(pred, on=["pair_id", "gold_relation"], how="left")
    audited.to_csv(out_dir / "independent_relation_gold_pairs.csv", index=False)

    confusion = pd.crosstab(audited["gold_relation"], audited["pred_relation"], dropna=False)
    confusion.to_csv(out_dir / "independent_relation_confusion_matrix.csv")

    metric_rows = []
    for rel in ["exact", "parent", "stereo", "tautomer", "family", "none"]:
        y_true = (audited["gold_relation"] == rel).to_numpy()
        y_pred = (audited["pred_relation"] == rel).to_numpy()
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
        metric_rows.append(
            {
                "relation": rel,
                "support": int(y_true.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    macro_f1 = float(metrics.loc[metrics["support"] > 0, "f1"].dropna().mean()) if not metrics.empty else float("nan")
    metrics["macro_f1_over_reported_relations"] = macro_f1
    metrics.to_csv(out_dir / "independent_relation_classification_metrics.csv", index=False)

    y_true = (audited["gold_relation"].isin(["exact", "parent", "stereo", "tautomer"])).to_numpy()
    y_pred = (audited[["pred_exact", "pred_parent", "pred_stereo", "pred_tautomer"]].astype(bool).any(axis=1)).to_numpy()
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    pd.DataFrame(
        [
            {
                "task": "independent_relation_gold",
                "positive_relations": "exact,parent,stereo,tautomer",
                "negative_relations": "family,none",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
            }
        ]
    ).to_csv(out_dir / "independent_certificate_detection_metrics.csv", index=False)
    write_tautomer_miss_taxonomy(audited, out_dir, max_examples=100)
    return metrics


def charge_count(mol) -> int:
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0))


def aromatic_atom_count(mol) -> int:
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))


def classify_tautomer_miss(query_smiles: str, target_smiles: str) -> str:
    qa = canonicalize(query_smiles)
    ta = canonicalize(target_smiles)
    qm = mol_from_smiles(query_smiles)
    tm = mol_from_smiles(target_smiles)
    if qm is None or tm is None or not qa or not ta:
        return "parsing_or_conversion"
    qf = formula_for(qa)
    tf = formula_for(ta)
    if qf != tf:
        return "formula_or_protonation_change"
    if charge_count(qm) != charge_count(tm):
        return "charge_protonation_interaction"
    qs = scaffold_smiles(qa)
    ts = scaffold_smiles(ta)
    if qs and ts and qs != ts:
        return "scaffold_or_connectivity_shift"
    if aromatic_atom_count(qm) != aromatic_atom_count(tm):
        return "aromaticity_kekulization_interaction"
    return "rdkit_tautomer_rule_coverage_or_ambiguous"


def write_tautomer_miss_taxonomy(audited: pd.DataFrame, out_dir: Path, max_examples: int = 100) -> None:
    misses = audited[(audited["gold_relation"] == "tautomer") & (~audited["pred_tautomer"].astype(bool))].copy()
    if misses.empty:
        pd.DataFrame().to_csv(out_dir / "tautomer_miss_taxonomy.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "tautomer_miss_examples.csv", index=False)
        return
    sample = misses.head(max_examples).copy()
    rows = []
    for row in sample.itertuples(index=False):
        rows.append(
            {
                "pair_id": row.pair_id,
                "query_smiles": row.query_smiles,
                "target_smiles": row.target_smiles,
                "pred_relation": row.pred_relation,
                "query_tautomer_key": tautomer_smiles(str(row.query_smiles)),
                "target_tautomer_key": tautomer_smiles(str(row.target_smiles)),
                "query_formula": formula_for(canonicalize(str(row.query_smiles))),
                "target_formula": formula_for(canonicalize(str(row.target_smiles))),
                "miss_category": classify_tautomer_miss(str(row.query_smiles), str(row.target_smiles)),
            }
        )
    examples = pd.DataFrame(rows)
    examples.to_csv(out_dir / "tautomer_miss_examples.csv", index=False)
    summary = (
        examples.groupby("miss_category", as_index=False)
        .agg(n=("pair_id", "count"))
        .sort_values("n", ascending=False)
    )
    summary["sample_fraction"] = summary["n"] / max(len(examples), 1)
    summary["sample_n"] = len(examples)
    summary["total_misses"] = len(misses)
    summary.to_csv(out_dir / "tautomer_miss_taxonomy.csv", index=False)


def shared_activity_stats(con: sqlite3.Connection, mol_a: str, mol_b: str, limit_assays: int = 80) -> list[tuple]:
    return con.execute(
        """
        select a1.assay_id, a1.pchembl_value, a2.pchembl_value
        from activities a1
        join activities a2 on a1.assay_id = a2.assay_id
        where a1.molregno = ?
          and a2.molregno = ?
          and a1.pchembl_value is not null
          and a2.pchembl_value is not null
          and a1.standard_relation in ('=', '<', '>', '<=', '>=')
          and a2.standard_relation in ('=', '<', '>', '<=', '>=')
        limit ?
        """,
        (mol_a, mol_b, limit_assays),
    ).fetchall()


def run_label_concordance(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    gold_path = out_dir / "independent_relation_gold_pairs.csv"
    if not args.chembl_db.exists() or not gold_path.exists():
        pd.DataFrame().to_csv(out_dir / "relation_label_concordance.csv", index=False)
        return pd.DataFrame()
    gold = pd.read_csv(gold_path)
    gold = gold[gold["task"] == "chembl36_registry"]
    relation_groups = []
    eligible_relations = [rel for rel in ["exact", "parent", "stereo"] if (gold["gold_relation"] == rel).any()]
    per_relation = max(1, args.max_concordance_pairs // max(len(eligible_relations), 1))
    for rel in eligible_relations:
        rel_df = gold[gold["gold_relation"] == rel]
        relation_groups.append(rel_df.sample(n=min(per_relation, len(rel_df)), random_state=args.seed))
    gold = pd.concat(relation_groups, ignore_index=True) if relation_groups else gold.head(0)
    rows = []
    con = sqlite3.connect(str(args.chembl_db))
    try:
        for row in gold.itertuples(index=False):
            shared = shared_activity_stats(con, str(row.query_id), str(row.target_id))
            if not shared:
                rows.append(
                    {
                        "pair_id": row.pair_id,
                        "gold_relation": row.gold_relation,
                        "n_shared_assays": 0,
                        "activity_label_agreement": float("nan"),
                        "median_abs_pchembl_diff": float("nan"),
                    }
                )
                continue
            diffs = [abs(float(a[1]) - float(a[2])) for a in shared]
            agreements = [(float(a[1]) >= 6.0) == (float(a[2]) >= 6.0) for a in shared]
            rows.append(
                {
                    "pair_id": row.pair_id,
                    "gold_relation": row.gold_relation,
                    "n_shared_assays": len(shared),
                    "activity_label_agreement": float(np.mean(agreements)),
                    "median_abs_pchembl_diff": float(np.median(diffs)),
                }
            )
    finally:
        con.close()
    per_pair = pd.DataFrame(rows)
    per_pair.to_csv(out_dir / "relation_label_concordance_pairs.csv", index=False)
    if per_pair.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            per_pair.groupby("gold_relation", as_index=False)
            .agg(
                n_pairs=("pair_id", "count"),
                pairs_with_shared_assay=("n_shared_assays", lambda x: int((x > 0).sum())),
                total_shared_assays=("n_shared_assays", "sum"),
                mean_label_agreement=("activity_label_agreement", "mean"),
                median_abs_pchembl_diff=("median_abs_pchembl_diff", "median"),
            )
            .sort_values("gold_relation")
        )
        observed = set(summary["gold_relation"])
        missing = [
            {
                "gold_relation": rel,
                "n_pairs": 0,
                "pairs_with_shared_assay": 0,
                "total_shared_assays": 0,
                "mean_label_agreement": float("nan"),
                "median_abs_pchembl_diff": float("nan"),
            }
            for rel in ["exact", "parent", "stereo"]
            if rel in eligible_relations and rel not in observed
        ]
        if missing:
            summary = pd.concat([summary, pd.DataFrame(missing)], ignore_index=True).sort_values("gold_relation")
    summary.to_csv(out_dir / "relation_label_concordance.csv", index=False)
    return summary


def key_tuple(smiles: str) -> tuple[str, str, str, str, str]:
    can = canonicalize(smiles)
    if not can:
        return "", "", "", "", ""
    return can, parent_smiles(can), stereo_stripped_smiles(can), tautomer_smiles(can), scaffold_smiles(can)


def build_key_sets(smiles: list[str], num_workers: int) -> dict[str, set[str]]:
    key_sets: dict[str, set[str]] = {k: set() for k in ["canonical", "parent", "stereo", "tautomer", "scaffold"]}
    for can, parent, stereo, tautomer, scaffold in parallel_map(key_tuple, smiles, num_workers, chunksize=512):
        key_sets["canonical"].add(can)
        key_sets["parent"].add(parent)
        key_sets["stereo"].add(stereo)
        key_sets["tautomer"].add(tautomer)
        key_sets["scaffold"].add(scaffold)
    for key in key_sets:
        key_sets[key].discard("")
    return key_sets


def exposure_label(smi: str, key_sets: dict[str, set[str]]) -> str:
    can = canonicalize(smi)
    if not can:
        return "invalid"
    if can in key_sets["canonical"]:
        return "exact"
    if parent_smiles(can) in key_sets["parent"]:
        return "parent"
    if stereo_stripped_smiles(can) in key_sets["stereo"]:
        return "stereo"
    if tautomer_smiles(can) in key_sets["tautomer"]:
        return "tautomer"
    return "none"


def exposure_label_from_keys(row, key_sets: dict[str, set[str]]) -> str:
    can = str(row.canonical_key)
    if not can:
        return "invalid"
    if can in key_sets["canonical"]:
        return "exact"
    parent = str(row.parent_key)
    if parent and parent in key_sets["parent"]:
        return "parent"
    stereo = str(row.stereo_key)
    if stereo and stereo in key_sets["stereo"]:
        return "stereo"
    tautomer = str(row.tautomer_key)
    if tautomer and tautomer in key_sets["tautomer"]:
        return "tautomer"
    return "none"


def run_external_corpus_exposure(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    task_queries = load_task_queries(args.artifact_root)
    print(f"[{now_iso()}] exposure precomputing task keys: {len(task_queries)} molecules", flush=True)
    task_features = parallel_map(key_tuple, task_queries["smiles"].astype(str).tolist(), args.num_workers, chunksize=512)
    task_queries["canonical_key"] = [x[0] for x in task_features]
    task_queries["parent_key"] = [x[1] for x in task_features]
    task_queries["stereo_key"] = [x[2] for x in task_features]
    task_queries["tautomer_key"] = [x[3] for x in task_features]
    task_queries["scaffold_key"] = [x[4] for x in task_features]
    corpora: list[tuple[str, str, list[str], str]] = []
    clean_path = args.artifact_root / "corpora" / "clean.txt"
    if clean_path.exists():
        corpora.append(
            (
                f"phase0_clean_first_{args.max_exposure_corpus}",
                "controlled_clean_corpus",
                read_smiles_text(clean_path, limit=args.max_exposure_corpus),
                "self_rerun",
            )
        )
    if args.chembl_db.exists():
        corpora.append(
            (
                f"chembl36_first_{args.max_exposure_corpus}",
                "ChEMBL36 compound_structures canonical_smiles",
                chembl_smiles(args.chembl_db, args.max_exposure_corpus),
                "external_registry_corpus",
            )
        )
    if args.zinc15_csv.exists():
        corpora.append(
            (
                "zinc15_250k_available_cache",
                str(args.zinc15_csv),
                read_smiles_csv(args.zinc15_csv, limit=args.max_exposure_corpus),
                "external_corpus_cache",
            )
        )

    rows = []
    for corpus_name, corpus_source, smiles, source_type in corpora:
        print(f"[{now_iso()}] exposure keying {corpus_name}: {len(smiles)} molecules", flush=True)
        key_sets = build_key_sets(smiles, args.num_workers)
        print(f"[{now_iso()}] exposure querying {corpus_name}: {len(task_queries)} task molecules", flush=True)
        for task, df in task_queries.groupby("task"):
            labels = [exposure_label_from_keys(row, key_sets) for row in df.itertuples(index=False)]
            counts = Counter(labels)
            n = len(labels)
            row = {
                "corpus_name": corpus_name,
                "corpus_source": corpus_source,
                "source_type": source_type,
                "n_corpus_molecules_loaded": len(smiles),
                "task": task,
                "n_task_molecules": n,
            }
            for label in ["exact", "parent", "stereo", "tautomer", "none", "invalid"]:
                row[f"{label}_count"] = counts.get(label, 0)
                row[f"{label}_fraction"] = counts.get(label, 0) / n if n else float("nan")
            rows.append(row)
    exposure = pd.DataFrame(rows)
    exposure.to_csv(out_dir / "external_corpus_exposure_by_task.csv", index=False)

    fm_rows = [
        {
            "target_model_or_corpus": "ChemBERTa-MLM-10M / ChemBERTa-MLM-100M",
            "requested_audit": "exact/parent/stereo/tautomer exposure plus loss/error stratified by certificate status",
            "status": "not_completed_without_official_pretraining_smiles_manifest",
            "available_proxy_completed": "phase0_clean, ChEMBL36 sample, ZINC15_250K cache exposure fractions",
            "paper_claim_allowed": "external registry/corpus proxy audit only",
        },
        {
            "target_model_or_corpus": "c3-MoLFormer-100M/550M/1.1B",
            "requested_audit": "corpus-growth exposure fractions and downstream error/loss certificate split",
            "status": "not_completed_without_nested_corpus_manifest_and_model_checkpoints",
            "available_proxy_completed": "scale benchmark plus external corpus exposure",
            "paper_claim_allowed": "do not claim causal corpus scaling from non-nested proxy corpora",
        },
    ]
    pd.DataFrame(fm_rows).to_csv(out_dir / "foundation_model_audit_status.csv", index=False)
    return exposure


def path_stability_one(item: tuple[int, str]) -> dict | None:
    idx, smi = item
    can = canonicalize(smi)
    if not can:
        return None
    p_then_t = tautomer_smiles(parent_smiles(can))
    t_then_p = parent_smiles(tautomer_smiles(can))
    strip_then_parent = parent_smiles(stereo_stripped_smiles(can))
    parent_then_strip = stereo_stripped_smiles(parent_smiles(can))
    return {
        "idx": idx,
        "canonical": can,
        "parent_tautomer_commutes": p_then_t == t_then_p,
        "stereo_parent_commutes": strip_then_parent == parent_then_strip,
        "canonical_idempotent": canonicalize(can) == can,
        "parent_idempotent": parent_smiles(parent_smiles(can)) == parent_smiles(can),
        "stereo_idempotent": stereo_stripped_smiles(stereo_stripped_smiles(can)) == stereo_stripped_smiles(can),
        "tautomer_idempotent": tautomer_smiles(tautomer_smiles(can)) == tautomer_smiles(can),
        "parent_then_tautomer": p_then_t,
        "tautomer_then_parent": t_then_p,
        "stereo_then_parent": strip_then_parent,
        "parent_then_stereo": parent_then_strip,
    }


def run_path_stability(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    smiles = []
    clean_path = args.artifact_root / "corpora" / "clean.txt"
    if clean_path.exists():
        smiles.extend(read_smiles_text(clean_path, limit=args.path_stability_n // 2))
    if args.chembl_db.exists() and len(smiles) < args.path_stability_n:
        smiles.extend(chembl_smiles(args.chembl_db, args.path_stability_n - len(smiles)))
    items = list(enumerate(smiles[: args.path_stability_n]))
    rows = [x for x in parallel_map(path_stability_one, items, args.num_workers, chunksize=512) if x is not None]
    examples = [
        row
        for row in rows
        if (not row["parent_tautomer_commutes"] or not row["stereo_parent_commutes"])
    ][:50]
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "normalization_path_stability_examples_full.csv", index=False)
    pd.DataFrame(examples).to_csv(out_dir / "normalization_path_instability_examples.csv", index=False)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = pd.DataFrame(
            [
                {
                    "n_molecules": len(detail),
                    "parent_tautomer_disagreement_rate": float((~detail["parent_tautomer_commutes"]).mean()),
                    "stereo_parent_disagreement_rate": float((~detail["stereo_parent_commutes"]).mean()),
                    "canonical_non_idempotent_rate": float((~detail["canonical_idempotent"]).mean()),
                    "parent_non_idempotent_rate": float((~detail["parent_idempotent"]).mean()),
                    "stereo_non_idempotent_rate": float((~detail["stereo_idempotent"]).mean()),
                    "tautomer_non_idempotent_rate": float((~detail["tautomer_idempotent"]).mean()),
                    "terminology_recommendation": (
                        "relation-typed provenance graph"
                        if ((~detail["parent_tautomer_commutes"]).mean() > 0 or (~detail["stereo_parent_commutes"]).mean() > 0)
                        else "graph terminology acceptable for tested normalizers"
                    ),
                }
            ]
        )
    summary.to_csv(out_dir / "normalization_path_stability_summary.csv", index=False)
    return summary


def equivalence_status(low: float, high: float, eps: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "not_estimable"
    if low >= -eps and high <= eps:
        return "practical_equivalence"
    if low > eps:
        return "positive_exceeds_epsilon"
    if high < -eps:
        return "negative_exceeds_epsilon"
    return "inconclusive"


def meta_rows_for_group(df: pd.DataFrame, group_cols: list[str], estimate_col: str, low_col: str, high_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        vals = []
        ses = []
        for row in g.itertuples(index=False):
            est = float(getattr(row, estimate_col))
            low = float(getattr(row, low_col))
            high = float(getattr(row, high_col))
            se = (high - low) / (2 * 1.96) if high > low else float("nan")
            if math.isfinite(est) and math.isfinite(se) and se > 0:
                vals.append(est)
                ses.append(se)
        if not vals:
            continue
        vals_np = np.asarray(vals)
        weights = 1.0 / np.square(np.asarray(ses))
        fixed = float(np.sum(weights * vals_np) / np.sum(weights))
        se_fixed = float(math.sqrt(1.0 / np.sum(weights)))
        q = float(np.sum(weights * np.square(vals_np - fixed)))
        df_q = max(len(vals) - 1, 1)
        i2 = max(0.0, (q - df_q) / q) if q > 0 else 0.0
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                "cells": len(vals),
                "fixed_effect_mean": fixed,
                "fixed_effect_ci_low": fixed - 1.96 * se_fixed,
                "fixed_effect_ci_high": fixed + 1.96 * se_fixed,
                "cochran_q": q,
                "i2": i2,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_equivalence_analysis(out_dir: Path) -> pd.DataFrame:
    candidates = [
        Path("reports/tables/phase0_effect_models_completed/effect_model_summary.csv"),
        Path("reports/tables/phase0_lowtrain25_effect_models_completed/effect_model_summary.csv"),
        Path("reports/tables/phase0_lowtrain05_effect_models_completed/effect_model_summary.csv"),
        Path("reports/tables/phase0_lowtrain02_effect_models_completed/effect_model_summary.csv"),
        Path("reports/tables/phase0_gru_lowtrain05_effect_models_completed/effect_model_summary.csv"),
        Path("reports/tables/phase0_gin_masked_atom_lowtrain05_effect_models_completed/effect_model_summary.csv"),
    ]
    frames = [pd.read_csv(p) for p in candidates if p.exists()]
    if not frames:
        pd.DataFrame().to_csv(out_dir / "downstream_equivalence_by_cell.csv", index=False)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    metric = "did_label_adjusted_minus_decoy"
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_boot_ci_low"
    high_col = f"{metric}_boot_ci_high"
    keep = df[[c for c in ["artifact", "task", "relation", "dose", "n_seeds", mean_col, low_col, high_col] if c in df.columns]].copy()
    for eps in [0.01, 0.02]:
        keep[f"equivalence_status_eps_{eps:g}"] = [
            equivalence_status(float(l), float(h), eps) for l, h in zip(keep[low_col], keep[high_col])
        ]
    keep.to_csv(out_dir / "downstream_equivalence_by_cell.csv", index=False)

    summary_rows = []
    for eps in [0.01, 0.02]:
        col = f"equivalence_status_eps_{eps:g}"
        for relation, g in keep.groupby("relation"):
            counts = Counter(g[col])
            summary_rows.append(
                {
                    "epsilon": eps,
                    "relation": relation,
                    "cells": len(g),
                    "practical_equivalence": counts.get("practical_equivalence", 0),
                    "positive_exceeds_epsilon": counts.get("positive_exceeds_epsilon", 0),
                    "negative_exceeds_epsilon": counts.get("negative_exceeds_epsilon", 0),
                    "inconclusive": counts.get("inconclusive", 0),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "downstream_equivalence_summary.csv", index=False)

    meta_relation = meta_rows_for_group(keep, ["relation"], mean_col, low_col, high_col)
    meta_relation.to_csv(out_dir / "downstream_heterogeneity_meta_by_relation.csv", index=False)
    meta_task = meta_rows_for_group(keep, ["task"], mean_col, low_col, high_col)
    meta_task.to_csv(out_dir / "downstream_heterogeneity_meta_by_task.csv", index=False)
    return summary


def estimate_key_storage_mb(keys: set[str]) -> float:
    return float(sum(len(k.encode("utf-8")) + 16 for k in keys) / 1_000_000.0)


def peak_rss_mb() -> float:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return value / 1_000_000.0
        return value / 1024.0
    except Exception:
        return float("nan")


def key_for_family(item: tuple[str, str]) -> str:
    key_name, smi = item
    if key_name == "canonical":
        return canonicalize(smi)
    if key_name == "parent":
        return parent_smiles(smi)
    if key_name == "stereo":
        return stereo_stripped_smiles(smi)
    if key_name == "tautomer":
        return tautomer_smiles(smi)
    if key_name == "scaffold":
        return scaffold_smiles(smi)
    raise KeyError(key_name)


def build_keys_for_family(key_name: str, smiles: list[str], num_workers: int) -> set[str]:
    items = [(key_name, smi) for smi in smiles]
    keys = set(parallel_map(key_for_family, items, num_workers, chunksize=512))
    keys.discard("")
    return keys


def run_scale_benchmark(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    smiles: list[str] = []
    clean_path = args.artifact_root / "corpora" / "clean.txt"
    if clean_path.exists():
        smiles.extend(read_smiles_text(clean_path, limit=None))
    if args.zinc15_csv.exists():
        smiles.extend(read_smiles_csv(args.zinc15_csv, limit=max(args.scale_sizes)))
    if args.chembl_db.exists() and len(smiles) < max(args.scale_sizes):
        smiles.extend(chembl_smiles(args.chembl_db, max(args.scale_sizes) - len(smiles)))
    smiles = list(dict.fromkeys(smiles))

    rows = []
    key_names = ["canonical", "parent", "stereo", "tautomer", "scaffold"]
    query_df = load_affected(args.artifact_root)
    query_smiles = query_df["canonical"].astype(str).head(512).tolist()
    for requested in args.scale_sizes:
        corpus = smiles[: min(requested, len(smiles))]
        for key_name in key_names:
            measure_n = len(corpus)
            projected = False
            if key_name in {"parent", "tautomer"} and measure_n > args.scale_relation_measure_limit:
                measure_n = args.scale_relation_measure_limit
                projected = True
            print(f"[{now_iso()}] scale keying {key_name} requested={requested} measured={measure_n}", flush=True)
            subset = corpus[:measure_n]
            start = time.perf_counter()
            keys = build_keys_for_family(key_name, subset, args.num_workers)
            wall = time.perf_counter() - start
            mps = measure_n / wall if wall > 0 else float("inf")
            est_build_s = requested / mps if mps > 0 else float("nan")
            q_start = time.perf_counter()
            hits = 0
            rounds = 30
            for _ in range(rounds):
                for smi in query_smiles:
                    qkey = key_for_family((key_name, smi))
                    hits += int(bool(qkey and qkey in keys))
            q_wall = time.perf_counter() - q_start
            qps = (len(query_smiles) * rounds) / q_wall if q_wall > 0 else float("inf")
            rows.append(
                {
                    "corpus_source": "phase0_clean_plus_zinc15_plus_chembl36",
                    "requested_corpus_size": requested,
                    "available_unique_smiles_used": len(corpus),
                    "key_family": key_name,
                    "measured_rows": measure_n,
                    "projection_used_for_requested_size": projected,
                    "measured_build_wall_s": wall,
                    "measured_molecules_per_s": mps,
                    "estimated_build_wall_s_for_requested_size": est_build_s,
                    "unique_keys_measured": len(keys),
                    "estimated_index_storage_mb": estimate_key_storage_mb(keys) * requested / max(measure_n, 1),
                    "peak_rss_mb": peak_rss_mb(),
                    "query_rounds": rounds,
                    "query_hits": hits,
                    "queries_per_s": qps,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "retrieval_scale_benchmark.csv", index=False)

    projection_rows = []
    for _, row in df.iterrows():
        for target in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
            projection_rows.append(
                {
                    "key_family": row["key_family"],
                    "projection_target_molecules": target,
                    "basis_requested_corpus_size": row["requested_corpus_size"],
                    "basis_measured_rows": row["measured_rows"],
                    "estimated_build_wall_s": row["measured_build_wall_s"] * target / max(row["measured_rows"], 1),
                    "estimated_index_storage_mb": row["estimated_index_storage_mb"] * target / max(row["requested_corpus_size"], 1),
                    "projection_note": "linear projection from measured streaming key construction; not a full materialized 1B run",
                }
            )
    pd.DataFrame(projection_rows).to_csv(out_dir / "retrieval_scale_projection_to_foundation_corpora.csv", index=False)
    return df


def write_readiness(args: argparse.Namespace, out_dir: Path) -> None:
    files = {
        "independent_relation_gold": out_dir / "independent_relation_classification_metrics.csv",
        "independent_certificate_detection": out_dir / "independent_certificate_detection_metrics.csv",
        "tautomer_miss_taxonomy": out_dir / "tautomer_miss_taxonomy.csv",
        "hard_negative_retrieval": out_dir / "hard_negative_retrieval_metrics.csv",
        "external_corpus_exposure": out_dir / "external_corpus_exposure_by_task.csv",
        "relation_label_concordance": out_dir / "relation_label_concordance.csv",
        "normalization_path_stability": out_dir / "normalization_path_stability_summary.csv",
        "downstream_equivalence": out_dir / "downstream_equivalence_summary.csv",
        "retrieval_scale": out_dir / "retrieval_scale_benchmark.csv",
        "foundation_model_status": out_dir / "foundation_model_audit_status.csv",
    }
    rows = []
    for item, path in files.items():
        exists = path.exists()
        n_rows = 0
        if exists:
            try:
                n_rows = len(pd.read_csv(path))
            except Exception:
                n_rows = 0
        status = "pass" if exists and n_rows > 0 else "not_completed"
        rows.append({"check": item, "status": status, "n_rows": n_rows, "path": str(path)})
    if args.chembl_db.exists():
        rows.append({"check": "chembl36_external_registry_available", "status": "pass", "n_rows": 1, "path": str(args.chembl_db)})
    else:
        rows.append({"check": "chembl36_external_registry_available", "status": "not_completed", "n_rows": 0, "path": str(args.chembl_db)})
    taut_path = ROOT / args.tautobase_dir / "Tautobase.dwar"
    rows.append(
        {
            "check": "tautobase_external_curated_available",
            "status": "pass" if taut_path.exists() else "not_completed",
            "n_rows": 1 if taut_path.exists() else 0,
            "path": str(taut_path),
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "reviewer_supplement_readiness.csv", index=False)


def write_markdown_summary(out_dir: Path) -> None:
    def load(path: str) -> pd.DataFrame:
        p = out_dir / path
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    hard = load("hard_negative_retrieval_metrics.csv")
    scenario = load("hard_negative_scenario_breakdown.csv")
    gold = load("independent_relation_classification_metrics.csv")
    cert = load("independent_certificate_detection_metrics.csv")
    tautomer_tax = load("tautomer_miss_taxonomy.csv")
    path_stability = load("normalization_path_stability_summary.csv")
    eq = load("downstream_equivalence_summary.csv")
    scale = load("retrieval_scale_benchmark.csv")
    fm = load("foundation_model_audit_status.csv")
    readiness = load("reviewer_supplement_readiness.csv")
    exposure = load("external_corpus_exposure_by_task.csv")
    concordance = load("relation_label_concordance.csv")

    lines = [
        "# Reviewer Supplement Experiments",
        "",
        f"Generated: {now_iso()}",
        "",
        "This supplement targets the reviewer risk that relation ground truth is self-generated, that decoys are too easy, and that downstream null results need equivalence-style interpretation.",
        "",
        "## Completed Tables",
        "",
    ]
    if not readiness.empty:
        lines.append(readiness.to_markdown(index=False))
        lines.append("")
    if not gold.empty:
        lines.extend(["## Independent Relation Gold", "", gold.to_markdown(index=False), ""])
    if not cert.empty:
        lines.extend(["## Independent Certificate Detection", "", cert.to_markdown(index=False), ""])
    if not hard.empty:
        cols = [
            "method",
            "method_group",
            "auroc",
            "auprc",
            "fpr_default",
            "fpr_default_one_sided95_upper",
            "precision_default",
            "ppv_empirical_default_at_prevalence_0.01",
            "ppv_default_fpr_upper95_at_prevalence_0.01",
        ]
        lines.extend(["## Hard-Negative Retrieval", "", hard[cols].head(12).to_markdown(index=False), ""])
    if not scenario.empty:
        chemtrace_scen = scenario[scenario["method"] == "chemtrace_typed_certificate"].copy()
        scen_cols = [
            "hard_negative_type",
            "n_negative",
            "fp_default",
            "false_positive_rate_at_default_binary",
            "fpr_default_one_sided95_upper",
        ]
        if not chemtrace_scen.empty:
            lines.extend(["## ChemTrace Hard-Negative FPR Bounds", "", chemtrace_scen[scen_cols].to_markdown(index=False), ""])
    if not tautomer_tax.empty:
        lines.extend(["## Tautomer Miss Taxonomy", "", tautomer_tax.to_markdown(index=False), ""])
    if not exposure.empty:
        corpus_col = "corpus_name" if "corpus_name" in exposure.columns else "corpus"
        total_col = "n_task_molecules" if "n_task_molecules" in exposure.columns else "total_queries"
        relation_cols = [
            f"{c}_count" if f"{c}_count" in exposure.columns else c
            for c in ["exact", "parent", "stereo", "tautomer", "none", "invalid"]
            if f"{c}_count" in exposure.columns or c in exposure.columns
        ]
        exposure_summary = (
            exposure.groupby(corpus_col, as_index=False)[relation_cols + [total_col]]
            .sum(numeric_only=True)
            .sort_values(corpus_col)
        )
        for col in relation_cols:
            base = col.removesuffix("_count")
            exposure_summary[f"{base}_fraction"] = exposure_summary[col] / exposure_summary[total_col].replace(0, np.nan)
        show_cols = [corpus_col, total_col] + relation_cols + [
            f"{col}_fraction"
            for col in ["exact", "parent", "stereo", "tautomer"]
            if f"{col}_fraction" in exposure_summary.columns
        ]
        lines.extend(["## External Corpus Exposure", "", exposure_summary[show_cols].to_markdown(index=False), ""])
    if not concordance.empty:
        lines.extend(
            [
                "## ChEMBL Relation-Label Concordance",
                "",
                concordance.to_markdown(index=False),
                "",
                "This table is a registry-side sanity check only: exact self-identity is expected to agree almost perfectly, while parent/stereo rows may be sparse because related ChEMBL forms are not always assayed in the same assay.",
                "",
            ]
        )
    if not path_stability.empty:
        lines.extend(["## Normalization Path Stability", "", path_stability.to_markdown(index=False), ""])
    if not eq.empty:
        lines.extend(["## Downstream Equivalence Summary", "", eq.to_markdown(index=False), ""])
    if not scale.empty:
        show = scale.sort_values(["requested_corpus_size", "key_family"]).head(20)
        lines.extend(["## Retrieval Scale Benchmark", "", show.to_markdown(index=False), ""])
    if not fm.empty:
        lines.extend(
            [
                "## Foundation Model Audit Boundary",
                "",
                fm.to_markdown(index=False),
                "",
                "The allowed manuscript claim is therefore an external registry/corpus proxy audit plus a clearly scoped future/secondary official-corpus audit, unless official ChemBERTa/MoLFormer corpus manifests and checkpoints are added.",
            ]
        )
    (out_dir / "reviewer_supplement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_fast_limits(args)
    ensure_out(args.out_dir)
    (args.out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"[{now_iso()}] running hard-negative retrieval", flush=True)
    hard_summary = run_hard_negative_retrieval(args)
    hard_bank = pd.read_csv(args.out_dir / "hard_negative_pair_bank.csv")

    print(f"[{now_iso()}] running independent relation gold", flush=True)
    run_independent_relation_gold(args, args.out_dir, hard_bank)

    print(f"[{now_iso()}] running ChEMBL label concordance", flush=True)
    run_label_concordance(args, args.out_dir)

    print(f"[{now_iso()}] running external corpus exposure", flush=True)
    run_external_corpus_exposure(args, args.out_dir)

    print(f"[{now_iso()}] running normalization path stability", flush=True)
    run_path_stability(args, args.out_dir)

    print(f"[{now_iso()}] running downstream equivalence/null analysis", flush=True)
    run_equivalence_analysis(args.out_dir)

    print(f"[{now_iso()}] running retrieval scale benchmark", flush=True)
    run_scale_benchmark(args, args.out_dir)

    write_readiness(args, args.out_dir)
    write_markdown_summary(args.out_dir)
    print(f"[{now_iso()}] wrote reviewer supplement tables to {args.out_dir}", flush=True)
    print(hard_summary.head(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
