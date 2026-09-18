from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .normalize import normalize_record
from .util import ensure_dir, stable_float_key


TASKS = {
    "esol": {
        "type": "regression",
        "aliases": ["esol.csv", "delaney-processed.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "label_candidates": ["measured log solubility in mols per litre", "logSolubility", "y"],
    },
    "bace": {
        "type": "classification",
        "aliases": ["bace.csv", "BACE.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "label_candidates": ["Class", "class", "label", "y"],
    },
    "bbbp": {
        "type": "classification",
        "aliases": ["BBBP.csv", "bbbp.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "label_candidates": ["p_np", "Class", "label", "y"],
    },
    "hiv": {
        "type": "classification",
        "aliases": ["HIV.csv", "hiv.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "label_candidates": ["HIV_active", "label", "y"],
    },
    "clintox_tox": {
        "type": "classification",
        "aliases": ["clintox.csv", "ClinTox.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "label_candidates": ["CT_TOX", "ct_tox", "label", "y"],
    },
    "clintox_fda": {
        "type": "classification",
        "aliases": ["clintox.csv", "ClinTox.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "label_candidates": ["FDA_APPROVED", "fda_approved", "label", "y"],
    },
    "tox21_ahr": {
        "type": "classification",
        "aliases": ["tox21.csv", "tox21.csv.gz", "Tox21.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "label_candidates": ["NR-AhR", "nr_ahr", "label", "y"],
    },
    "tox21_are": {
        "type": "classification",
        "aliases": ["tox21.csv", "tox21.csv.gz", "Tox21.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "label_candidates": ["SR-ARE", "sr_are", "label", "y"],
    },
    "tox21_mmp": {
        "type": "classification",
        "aliases": ["tox21.csv", "tox21.csv.gz", "Tox21.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "label_candidates": ["SR-MMP", "sr_mmp", "label", "y"],
    },
    "sider_gastro": {
        "type": "classification",
        "aliases": ["sider.csv", "SIDER.csv"],
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "label_candidates": ["Gastrointestinal disorders", "gastrointestinal_disorders", "label", "y"],
    },
}


SMILES_COLUMNS = ["smiles", "SMILES", "mol", "Molecule", "compound_iso_smiles", "smiles_raw"]


def _find_cached_file(task: str, external_cache: Path) -> Path | None:
    aliases = TASKS[task]["aliases"]
    candidates: list[Path] = []
    if not external_cache.exists():
        return None
    for root in [external_cache]:
        for alias in aliases:
            candidates.extend(root.rglob(alias))
    if not candidates:
        return None
    return max(candidates, key=_row_count_fast)


def _row_count_fast(path: Path) -> int:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return max(0, sum(1 for _ in f) - 1)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return -1


def _download(url: str, out: Path) -> Path:
    ensure_dir(out.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as src:
            tmp = path.with_suffix("")
            with tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return pd.read_csv(tmp)
    if path.suffix == ".tab":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _choose_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise ValueError(f"Could not find {kind} column in {list(df.columns)}")


def load_task(task: str, data_root: Path, external_cache: Path) -> pd.DataFrame:
    task = task.lower()
    if task not in TASKS:
        raise KeyError(f"Unsupported task {task}; known={sorted(TASKS)}")
    ensure_dir(data_root / "raw")
    cached = _find_cached_file(task, external_cache)
    if cached is None:
        cached = data_root / "raw" / TASKS[task]["aliases"][0]
        if not cached.exists():
            _download(TASKS[task]["url"], cached)
    df = _read_table(cached)
    smiles_col = _choose_column(df, SMILES_COLUMNS, "SMILES")
    y_col = _choose_column(df, TASKS[task]["label_candidates"], "label")
    out = df[[smiles_col, y_col]].rename(columns={smiles_col: "smiles", y_col: "y"}).copy()
    out["task"] = task
    out["task_type"] = TASKS[task]["type"]
    out["sample_id"] = [f"{task}_{i:06d}" for i in range(len(out))]
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["smiles", "y"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    norms = [normalize_record(s) for s in out["smiles"]]
    out["canonical"] = [n.canonical for n in norms]
    out["parent"] = [n.parent for n in norms]
    out["stereo_stripped"] = [n.stereo_stripped for n in norms]
    out["scaffold"] = [n.scaffold or n.canonical for n in norms]
    out = out[out["canonical"] != ""].reset_index(drop=True)
    return out


def scaffold_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    groups = (
        df.groupby("scaffold", dropna=False)
        .size()
        .reset_index(name="n")
        .assign(key=lambda x: x["scaffold"].map(lambda s: stable_float_key(str(s), seed)))
        .sort_values(["key", "scaffold"])
    )
    total = len(df)
    train_cut = int(total * 0.8)
    val_cut = int(total * 0.9)
    split_by_scaffold: dict[str, str] = {}
    seen = 0
    for row in groups.itertuples(index=False):
        mid = seen + int(row.n) / 2.0
        split = "train" if mid < train_cut else "val" if mid < val_cut else "test"
        split_by_scaffold[str(row.scaffold)] = split
        seen += int(row.n)
    out = df.copy()
    out["split"] = out["scaffold"].map(lambda s: split_by_scaffold[str(s)])
    out = _ensure_nonempty_splits(out, groups)
    return out


def _ensure_nonempty_splits(out: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    total = len(out)
    min_val = max(1, int(total * 0.05))
    min_test = max(1, int(total * 0.05))
    ordered = [str(s) for s in groups["scaffold"].tolist()]

    def count(split: str) -> int:
        return int((out["split"] == split).sum())

    def move_from_to(src: str, dst: str, target: int) -> None:
        moved = 0
        for scaf in reversed(ordered):
            mask = (out["scaffold"].astype(str) == scaf) & (out["split"] == src)
            n = int(mask.sum())
            if n == 0:
                continue
            out.loc[mask, "split"] = dst
            moved += n
            if moved >= target:
                break

    if count("val") == 0:
        move_from_to("train", "val", min_val)
    if count("test") == 0:
        move_from_to("val" if count("val") > min_val else "train", "test", min_test)
    return out
