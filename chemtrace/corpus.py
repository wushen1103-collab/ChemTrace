from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .datasets import load_task, scaffold_split
from .normalize import canonicalize, parent_smiles, random_smiles, stereo_stripped_smiles, tautomer_smiles
from .util import ensure_dir, sha256_text, stable_float_key, write_lines


def _scan_smiles_files(external_cache: Path) -> list[Path]:
    patterns = ["*.csv", "*.csv.gz", "*.smi", "*.smiles", "*.txt"]
    files: list[Path] = []
    if not external_cache.exists():
        return files
    for root in [external_cache]:
        for pat in patterns:
            files.extend(root.rglob(pat))
    return sorted(set(files))[:2000]


def _extract_smiles_from_file(path: Path, limit: int = 100000) -> list[str]:
    vals: list[str] = []
    try:
        if path.suffix in {".csv", ".gz"}:
            df = pd.read_csv(path, nrows=limit)
            candidates = [c for c in df.columns if c.lower() in {"smiles", "smile", "mol", "molecule", "compound_iso_smiles"}]
            if not candidates:
                return []
            vals = df[candidates[0]].dropna().astype(str).tolist()
        else:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    token = line.strip().split()[0] if line.strip() else ""
                    if token and any(ch in token for ch in "CON[]=#()+-123456789"):
                        vals.append(token)
                    if len(vals) >= limit:
                        break
    except Exception:
        return []
    out = []
    for s in vals:
        c = canonicalize(s)
        if c:
            out.append(c)
    return out


def collect_base_smiles(external_cache: Path, min_size: int, exclude: set[str], seed: int) -> list[str]:
    seen: set[str] = set()
    base: list[str] = []
    for path in _scan_smiles_files(external_cache):
        for smi in _extract_smiles_from_file(path):
            if smi not in exclude and smi not in seen:
                seen.add(smi)
                base.append(smi)
        if len(base) >= min_size:
            break
    if not base:
        raise RuntimeError("No valid base SMILES found in cache; add a corpus file or enable download first.")
    base = sorted(base, key=lambda s: stable_float_key(s, seed))
    rng = random.Random(seed)
    while len(base) < min_size:
        smi = rng.choice(base)
        aug = random_smiles(smi, rng.randrange(1_000_000))
        can = canonicalize(aug)
        if can and can not in exclude:
            base.append(aug)
    return base[:min_size]


def _select_affected_ids(test: pd.DataFrame, affected_per_task: int, seed: int, task_type: str) -> set[str]:
    test = test.copy()
    test["_affected_key"] = test["canonical"].map(lambda x: stable_float_key(str(x), seed))
    n_total = min(affected_per_task, len(test))
    if task_type != "classification" or n_total <= 1 or test["y"].nunique(dropna=True) < 2:
        return set(test.sort_values("_affected_key").head(n_total)["sample_id"])

    labels = sorted(test["y"].dropna().unique().tolist())
    max_by_label = {}
    for label in labels:
        n_label = int((test["y"] == label).sum())
        max_by_label[label] = min(n_label, n_total)

    per_label = max(1, n_total // len(labels))
    selected: list[pd.DataFrame] = []
    selected_counts = {label: 0 for label in labels}
    for label in labels:
        group = test[test["y"] == label].sort_values("_affected_key")
        n_take = min(per_label, max_by_label[label], len(group))
        chosen = group.head(n_take)
        selected.append(chosen)
        selected_counts[label] += len(chosen)

    chosen = pd.concat(selected, ignore_index=True) if selected else test.head(0)
    if len(chosen) < n_total:
        remaining = test[~test["sample_id"].isin(chosen["sample_id"])].sort_values("_affected_key")
        fill_rows = []
        for _, row in remaining.iterrows():
            label = row["y"]
            if selected_counts.get(label, 0) >= max_by_label.get(label, n_total):
                continue
            fill_rows.append(row.to_dict())
            selected_counts[label] = selected_counts.get(label, 0) + 1
            if len(chosen) + len(fill_rows) >= n_total:
                break
        if fill_rows:
            chosen = pd.concat([chosen, pd.DataFrame(fill_rows)], ignore_index=True)
    return set(chosen.head(n_total)["sample_id"])


def _relation_variant(smiles: str, relation: str, seed: int) -> str:
    if relation == "exact":
        return canonicalize(smiles)
    if relation == "random":
        return random_smiles(smiles, seed)
    if relation == "parent":
        return parent_smiles(smiles)
    if relation == "stereo":
        return stereo_stripped_smiles(smiles)
    if relation == "tautomer":
        return tautomer_smiles(smiles)
    raise KeyError(f"Unknown relation {relation}")


def _write_condition_corpus(
    clean: list[str],
    affected: pd.DataFrame,
    relation: str,
    dose: int,
    out_path: Path,
    manifest_path: Path,
    seed: int,
) -> None:
    rng = random.Random(seed + dose + len(relation))
    corpus = list(clean)
    manifest = []
    buckets: dict[int, list[int]] = {}
    for idx, smi in enumerate(corpus):
        buckets.setdefault(len(smi), []).append(idx)
    for length, values in buckets.items():
        values.sort(key=lambda i: stable_float_key(f"{length}:{corpus[i]}:{i}", seed))
    decoy_by_len: dict[int, list[str]] = {}
    for smi in clean:
        decoy_by_len.setdefault(len(smi), []).append(smi)
    for length, values in decoy_by_len.items():
        values.sort(key=lambda s: stable_float_key(f"decoy:{length}:{s}", seed))

    def take_background(target_len: int) -> int:
        max_len = max(buckets) if buckets else 0
        for offset in range(max_len + target_len + 2):
            for length in (target_len - offset, target_len + offset):
                if length in buckets and buckets[length]:
                    pos = rng.randrange(len(buckets[length]))
                    return buckets[length].pop(pos)
        raise RuntimeError("No background records left for contamination replacement")

    def pick_decoy(target_len: int, sample_id: str, rep: int) -> str:
        max_len = max(decoy_by_len) if decoy_by_len else 0
        for offset in range(max_len + target_len + 2):
            for length in (target_len - offset, target_len + offset):
                values = decoy_by_len.get(length)
                if values:
                    pos = int(stable_float_key(f"{sample_id}:{rep}:{length}", seed) * len(values)) % len(values)
                    return values[pos]
        raise RuntimeError("No clean decoy records available")

    for row in affected.itertuples(index=False):
        for rep in range(dose):
            if relation == "decoy":
                injected = pick_decoy(len(row.canonical), row.sample_id, rep)
            else:
                injected = _relation_variant(row.canonical, relation, seed + rep)
            if not injected:
                injected = row.canonical
            bg_idx = take_background(len(injected))
            before = corpus[bg_idx]
            corpus[bg_idx] = injected
            manifest.append(
                {
                    "sample_id": row.sample_id,
                    "task": row.task,
                    "relation": relation,
                    "dose": dose,
                    "rep": rep,
                    "background_index": bg_idx,
                    "background_sha256": sha256_text(before),
                    "background_len": len(before),
                    "injected_sha256": sha256_text(injected),
                    "injected_smiles": injected,
                    "injected_len": len(injected),
                    "length_delta": len(injected) - len(before),
                }
            )
    write_lines(out_path, corpus)
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def build_phase0_artifacts(
    tasks: list[str],
    data_root: Path,
    artifact_root: Path,
    external_cache: Path,
    corpus_size: int,
    affected_per_task: int,
    doses: list[int],
    relations: list[str],
    seed: int,
    max_affected_len: int = 158,
    affected_filter_relation: str | None = None,
) -> None:
    split_dir = ensure_dir(artifact_root / "splits")
    corpus_dir = ensure_dir(artifact_root / "corpora")
    manifest_dir = ensure_dir(artifact_root / "manifests")

    split_frames = []
    affected_frames = []
    exclude: set[str] = set()
    for task in tasks:
        df = scaffold_split(load_task(task, data_root, external_cache), seed=seed)
        test = df[df["split"] == "test"].copy()
        task_type = str(df["task_type"].iloc[0])
        test_for_affected = test[test["canonical"].str.len() <= max_affected_len].copy()
        if affected_filter_relation:
            test_for_affected = test_for_affected[
                test_for_affected["canonical"].map(
                    lambda s: _relation_variant(str(s), affected_filter_relation, seed) != canonicalize(str(s))
                )
            ].copy()
        if len(test_for_affected) < min(affected_per_task, len(test)):
            if affected_filter_relation:
                raise RuntimeError(
                    f"Only {len(test_for_affected)} test molecules satisfy affected_filter_relation="
                    f"{affected_filter_relation!r} for task={task}, fewer than requested {affected_per_task}."
                )
            test_for_affected = test
        affected_ids = _select_affected_ids(test_for_affected, affected_per_task, seed, task_type)
        df["affected"] = df["sample_id"].isin(affected_ids)
        df.to_csv(split_dir / f"{task}.csv", index=False)
        split_frames.append(df)
        affected_frames.append(df[df["affected"]].copy())
        exclude.update(df[df["split"] == "test"]["canonical"].tolist())
        print(f"task={task} n={len(df)} test={len(test)} affected={len(affected_ids)} type={df['task_type'].iloc[0]}")

    affected = pd.concat(affected_frames, ignore_index=True)
    clean = collect_base_smiles(external_cache, corpus_size, exclude, seed)
    write_lines(corpus_dir / "clean.txt", clean)
    pd.concat(split_frames, ignore_index=True).to_csv(artifact_root / "phase0_all_splits.csv", index=False)
    affected.to_csv(artifact_root / "phase0_affected.csv", index=False)

    meta = {
        "tasks": tasks,
        "seed": seed,
        "corpus_size": corpus_size,
        "clean_sha256": sha256_text("\n".join(clean)),
        "n_affected": len(affected),
        "relations": relations,
        "doses": doses,
        "max_affected_len": max_affected_len,
        "affected_filter_relation": affected_filter_relation,
    }
    (artifact_root / "prepare_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    for relation in relations:
        for dose in doses:
            name = f"{relation}_{dose}x"
            _write_condition_corpus(
                clean=clean,
                affected=affected,
                relation=relation,
                dose=dose,
                out_path=corpus_dir / f"{name}.txt",
                manifest_path=manifest_dir / f"{name}.jsonl",
                seed=seed,
            )
            print(f"condition={name} corpus={corpus_dir / (name + '.txt')}")
