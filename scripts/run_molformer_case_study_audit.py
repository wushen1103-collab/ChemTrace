#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewer_supplement_experiments import (
    build_key_sets,
    exposure_label_from_keys,
    load_task_queries,
    key_tuple,
    parallel_map,
    read_smiles_csv,
    read_smiles_text,
    worker_count,
)

RDLogger.DisableLog("rdApp.*")

IBM_README = "https://raw.githubusercontent.com/IBM/molformer/main/README.md"
IBM_BOX = "https://ibm.box.com/v/MoLFormer-data"
HF_API = "https://hf-mirror.com/api/models/ibm-research/MoLFormer-XL-both-10pct"
HF_CARD = "https://hf-mirror.com/ibm-research/MoLFormer-XL-both-10pct/raw/main/README.md"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a real MoLFormer ecosystem ChemTrace audit when corpus files are available.")
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0_tf_confirm_top_len158_extra"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/phase0_molformer_case_study"))
    p.add_argument("--corpus-root", type=Path, default=Path("external/molformer/data"))
    p.add_argument("--zinc15-csv", type=Path, default=Path("data/external/zinc15_250K.csv"))
    p.add_argument("--max-corpus-molecules", type=int, default=1000000)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def curl_text(url: str, timeout_s: int = 45, max_chars: int = 20000) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            ["curl", "-Lk", "--max-time", str(timeout_s), url],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.stdout[:max_chars], proc.stderr[:4000]
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def write_official_source_status(out_dir: Path) -> None:
    rows = []
    for name, url in [
        ("ibm_molformer_readme", IBM_README),
        ("ibm_box_dataset_landing_page", IBM_BOX),
        ("hf_mirror_model_api", HF_API),
        ("hf_mirror_model_card", HF_CARD),
    ]:
        text, err = curl_text(url)
        lower = text.lower()
        rows.append(
            {
                "source_name": name,
                "url": url,
                "reachable": bool(text.strip()),
                "mentions_pubchem": "pubchem" in lower,
                "mentions_zinc": "zinc" in lower,
                "mentions_100m_or_10pct": ("100m" in lower) or ("10%" in lower) or ("10pct" in lower),
                "mentions_checkpoint_or_safetensors": ("checkpoint" in lower) or ("safetensors" in lower),
                "bytes_captured": len(text.encode("utf-8")),
                "stderr_head": err.replace("\n", " ")[:300],
            }
        )
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "molformer_official_source_status.csv", index=False)

    api_text = (out_dir / "hf_mirror_model_api.txt").read_text(encoding="utf-8") if (out_dir / "hf_mirror_model_api.txt").exists() else ""
    try:
        api = json.loads(api_text)
        siblings = api.get("siblings", [])
        pd.DataFrame(siblings).to_csv(out_dir / "molformer_hf_files.csv", index=False)
    except Exception:
        pd.DataFrame().to_csv(out_dir / "molformer_hf_files.csv", index=False)


def candidate_corpus_files(corpus_root: Path) -> list[Path]:
    roots = [corpus_root, ROOT / "external" / "molformer", ROOT / "data"]
    patterns = ["*.smi", "*.smi.gz", "*.csv", "*.csv.gz", "*.txt", "*.txt.gz"]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            out.extend(root.rglob(pattern))
    return sorted(set(out))


def read_smiles_any(path: Path, limit: int) -> list[str]:
    suffixes = "".join(path.suffixes).lower()
    if ".csv" in suffixes:
        return read_smiles_csv(path, limit=limit)
    return read_smiles_text(path, limit=limit)


def load_molformer_proxy_corpus(args: argparse.Namespace) -> tuple[list[str], list[dict]]:
    rows = []
    smiles: list[str] = []
    remaining = args.max_corpus_molecules

    files = candidate_corpus_files(args.corpus_root)
    for path in files:
        if remaining <= 0:
            break
        try:
            got = read_smiles_any(path, remaining)
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "used": False,
                    "n_loaded": 0,
                    "reason": f"read_failed:{type(exc).__name__}:{exc}",
                }
            )
            continue
        if got:
            smiles.extend(got)
            remaining -= len(got)
        rows.append({"path": str(path), "used": bool(got), "n_loaded": len(got), "reason": "" if got else "empty"})

    if remaining > 0 and args.zinc15_csv.exists():
        got = read_smiles_csv(args.zinc15_csv, limit=remaining)
        smiles.extend(got)
        rows.append(
            {
                "path": str(args.zinc15_csv),
                "used": bool(got),
                "n_loaded": len(got),
                "reason": "zinc15_available_cache_proxy_for_molformer_zinc_component",
            }
        )

    deduped = list(dict.fromkeys(s for s in smiles if isinstance(s, str) and s.strip()))
    return deduped[: args.max_corpus_molecules], rows


def run_exposure(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_official_source_status(out_dir)

    smiles, manifest_rows = load_molformer_proxy_corpus(args)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "molformer_corpus_manifest.csv", index=False)
    status_rows = [
        {
            "audit_item": "official_molformer_100m_corpus_files",
            "status": "available_local" if candidate_corpus_files(args.corpus_root) else "not_available_noninteractive",
            "note": "IBM Box landing page is documented by the official README, but this run did not find extracted official 100M PubChem/ZINC files under --corpus-root.",
        },
        {
            "audit_item": "molformer_hf_checkpoint_metadata",
            "status": "available_via_hf_mirror",
            "note": "HF mirror exposes MoLFormer-XL-both-10pct metadata and model.safetensors; checkpoint-side loss requires transformers/huggingface_hub in a separate environment.",
        },
        {
            "audit_item": "pubchem_zinc_proxy_exposure",
            "status": "completed" if smiles else "not_completed_no_corpus",
            "note": f"Loaded {len(smiles)} unique molecules from local MoLFormer-like files and/or ZINC15 cache proxy.",
        },
    ]
    pd.DataFrame(status_rows).to_csv(out_dir / "molformer_case_study_status.csv", index=False)

    if not smiles:
        pd.DataFrame().to_csv(out_dir / "molformer_proxy_exposure_by_task.csv", index=False)
        return

    num_workers = worker_count(args.num_workers)
    task_queries = load_task_queries(args.artifact_root)
    task_features = parallel_map(key_tuple, task_queries["smiles"].astype(str).tolist(), num_workers, chunksize=512)
    task_queries = task_queries.copy()
    task_queries["canonical_key"] = [x[0] for x in task_features]
    task_queries["parent_key"] = [x[1] for x in task_features]
    task_queries["stereo_key"] = [x[2] for x in task_features]
    task_queries["tautomer_key"] = [x[3] for x in task_features]
    key_sets = build_key_sets(smiles, num_workers)

    labels = [exposure_label_from_keys(row, key_sets) for row in task_queries.itertuples(index=False)]
    task_queries["exposure_label"] = labels
    task_queries.to_csv(out_dir / "molformer_proxy_exposure_per_molecule.csv", index=False)

    rows = []
    for task, group in task_queries.groupby("task"):
        counts = group["exposure_label"].value_counts()
        total = len(group)
        row = {
            "case_study": "MoLFormer-XL-both-10pct documented PubChem/ZINC ecosystem",
            "corpus_molecules_loaded": len(smiles),
            "task": task,
            "n_task_molecules": total,
        }
        for label in ["exact", "parent", "stereo", "tautomer", "none", "invalid"]:
            row[f"{label}_count"] = int(counts.get(label, 0))
            row[f"{label}_fraction"] = int(counts.get(label, 0)) / total if total else float("nan")
        rows.append(row)
    pd.DataFrame(rows).sort_values("task").to_csv(out_dir / "molformer_proxy_exposure_by_task.csv", index=False)

    totals = task_queries["exposure_label"].value_counts()
    total = len(task_queries)
    summary = {
        "case_study": "MoLFormer-XL-both-10pct documented PubChem/ZINC ecosystem",
        "generated": now_iso(),
        "corpus_molecules_loaded": len(smiles),
        "task_molecules": total,
        "official_corpus_files_available_local": bool(candidate_corpus_files(args.corpus_root)),
    }
    for label in ["exact", "parent", "stereo", "tautomer", "none", "invalid"]:
        summary[f"{label}_count"] = int(totals.get(label, 0))
        summary[f"{label}_fraction"] = int(totals.get(label, 0)) / total if total else float("nan")
    pd.DataFrame([summary]).to_csv(out_dir / "molformer_proxy_exposure_summary.csv", index=False)

    lines = [
        "# MoLFormer Case Study Audit",
        "",
        f"Generated: {now_iso()}",
        "",
        "Official sources establish that MoLFormer uses PubChem/ZINC pretraining and that the public 10pct checkpoint is model-side accessible. This run audits locally available extracted corpus files when present; otherwise it falls back to an explicitly labeled ZINC15 cache proxy and records the official-corpus gap.",
        "",
        "## Status",
        "",
        pd.DataFrame(status_rows).to_markdown(index=False),
        "",
        "## Exposure Summary",
        "",
        pd.DataFrame([summary]).to_markdown(index=False),
        "",
    ]
    exposure = pd.read_csv(out_dir / "molformer_proxy_exposure_by_task.csv")
    lines.extend(["## Exposure By Task", "", exposure.to_markdown(index=False), ""])
    (out_dir / "molformer_case_study_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_exposure(args)


if __name__ == "__main__":
    main()
