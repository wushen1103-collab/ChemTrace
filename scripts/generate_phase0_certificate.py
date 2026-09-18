#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.util import sha256_file


COND_RE = re.compile(r"(?P<relation>.+)_(?P<dose>\d+)x$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a machine-readable ChemTrace Phase-0 provenance certificate.")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--effect-table", type=Path, default=None)
    p.add_argument("--retrieval-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--schema-file", type=Path, default=ROOT / "schemas/chemtrace-certificate.schema.json")
    p.add_argument("--repository-commit", type=str, default=None)
    p.add_argument("--analysis-version", type=str, default="final-manuscript-2026-09-10")
    p.add_argument(
        "--generated-at-utc",
        type=str,
        default=None,
        help="Optional fixed ISO-8601 UTC timestamp for byte-reproducible certificates.",
    )
    return p.parse_args()


def generated_at_utc(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("--generated-at-utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("--generated-at-utc must include a UTC offset")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("--generated-at-utc must denote UTC")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _condition_parts(name: str) -> tuple[str, int | None]:
    match = COND_RE.match(name)
    if not match:
        return name, None
    return match.group("relation"), int(match.group("dose"))


def corpus_records(artifact_root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((artifact_root / "corpora").glob("*.txt")):
        relation, dose = _condition_parts(path.stem)
        out.append(
            {
                "condition": path.stem,
                "relation": relation,
                "dose": dose,
                "records": _count_lines(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return out


def manifest_records(artifact_root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((artifact_root / "manifests").glob("*.jsonl")):
        relation, dose = _condition_parts(path.stem)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        rec: dict[str, Any] = {
            "condition": path.stem,
            "relation": relation,
            "dose": dose,
            "records": len(df),
            "sha256": sha256_file(path),
        }
        if not df.empty:
            rec.update(
                {
                    "tasks": sorted(df["task"].astype(str).unique().tolist()),
                    "unique_samples": int(df["sample_id"].nunique()),
                    "unique_injected": int(df["injected_sha256"].nunique()),
                    "mean_length_delta": float(df["length_delta"].mean()),
                    "max_abs_length_delta": int(df["length_delta"].abs().max()),
                }
            )
        out.append(rec)
    return out


def split_records(artifact_root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((artifact_root / "splits").glob("*.csv")):
        df = pd.read_csv(path)
        task = path.stem
        rec: dict[str, Any] = {
            "task": task,
            "records": len(df),
            "sha256": sha256_file(path),
            "task_type": str(df["task_type"].iloc[0]) if "task_type" in df and len(df) else None,
            "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
            "affected": int(df["affected"].astype(bool).sum()) if "affected" in df else 0,
        }
        if "y" in df:
            affected = df[df["affected"].astype(bool)] if "affected" in df else df.head(0)
            rec["label_counts_all"] = {str(k): int(v) for k, v in df["y"].value_counts(dropna=False).items()}
            rec["label_counts_affected"] = {str(k): int(v) for k, v in affected["y"].value_counts(dropna=False).items()}
        out.append(rec)
    return out


def effect_records(effect_table: Path | None, artifact_name: str) -> list[dict[str, Any]]:
    if not effect_table or not effect_table.exists():
        return []
    df = pd.read_csv(effect_table)
    if "artifact" in df.columns:
        df = df[df["artifact"].astype(str) == artifact_name].copy()
    renames = {
        "artifact": "experiment",
        "mean_minus_decoy": "point_mean_minus_decoy",
        "ci_low": "boot_ci_low",
        "ci_high": "boot_ci_high",
        "did_label_adjusted_minus_decoy_mean": "point_mean_minus_decoy",
        "did_label_adjusted_minus_decoy_boot_ci_low": "boot_ci_low",
        "did_label_adjusted_minus_decoy_boot_ci_high": "boot_ci_high",
        "did_label_adjusted_minus_decoy_boot_positive_rate": "boot_positive_rate",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    if "support_bucket" not in df.columns:
        df["support_bucket"] = df.apply(_support_bucket_from_effect, axis=1)
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
    ]
    keep = [c for c in keep if c in df.columns]
    return [{k: _jsonable(v) for k, v in row.items()} for row in df[keep].to_dict(orient="records")]


def repository_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    commit = proc.stdout.strip()
    if proc.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Unable to determine a 40-character repository commit; pass --repository-commit.")
    return commit


def _support_bucket_from_effect(row: pd.Series) -> str:
    lo = float(row.get("boot_ci_low", 0.0))
    hi = float(row.get("boot_ci_high", 0.0))
    mean = float(row.get("point_mean_minus_decoy", 0.0))
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


def retrieval_records(retrieval_dir: Path | None) -> dict[str, Any]:
    if not retrieval_dir or not retrieval_dir.exists():
        return {}
    out: dict[str, Any] = {}
    for name in ["retrieval_summary_by_relation.csv", "retrieval_clean_baseline.csv"]:
        path = retrieval_dir / name
        if path.exists():
            df = pd.read_csv(path)
            out[path.stem] = [{k: _jsonable(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    return out


def risk_statement(effects: list[dict[str, Any]], retrieval: dict[str, Any]) -> dict[str, Any]:
    buckets = [str(r.get("support_bucket")) for r in effects if r.get("support_bucket") is not None]
    if any(b == "positive_ci_excludes_zero" for b in buckets):
        effect_level = "confirmed_positive_inflation"
    elif any(b == "directional_positive" for b in buckets):
        effect_level = "directional_positive_not_ci_confirmed"
    elif effects:
        effect_level = "no_stable_positive_inflation_detected"
    else:
        effect_level = "effect_not_evaluated"

    semantic_retrieval = "not_evaluated"
    summary = retrieval.get("retrieval_summary_by_relation", [])
    if summary:
        changed = [
            r
            for r in summary
            if r.get("detector") in {"parent", "stereo", "tautomer"}
            and r.get("mean_changed_only_recall") is not None
            and float(r.get("mean_changed_only_recall") or 0) >= 0.99
        ]
        semantic_retrieval = "semantic_lattice_recovers_changed_variants" if changed else "semantic_lattice_partial"

    return {
        "effect_level": effect_level,
        "retrieval_level": semantic_retrieval,
        "recommended_claim": (
            "Use as an audit/provenance certificate. Treat downstream inflation as conditional unless a "
            "relation/task has a cluster bootstrap interval above zero."
        ),
    }


def main() -> None:
    args = parse_args()
    out = args.out or Path("reports/certificates") / f"{args.artifact_root.name}.certificate.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.schema_file.exists():
        raise FileNotFoundError(f"Certificate JSON Schema not found: {args.schema_file}")
    schema_document = json.loads(args.schema_file.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema_document)

    effects = effect_records(args.effect_table, args.artifact_root.name)
    retrieval = retrieval_records(args.retrieval_dir)
    certificate = {
        "schema": "chemtrace.phase0.certificate.v2",
        "schema_version": "2.0.0",
        "schema_sha256": sha256_file(args.schema_file),
        "repository_commit": repository_commit(args.repository_commit),
        "analysis_version": args.analysis_version,
        "generated_at_utc": generated_at_utc(args.generated_at_utc),
        "artifact_root": str(args.artifact_root),
        "artifact_name": args.artifact_root.name,
        "prepare_meta": _read_json(args.artifact_root / "prepare_meta.json"),
        "splits": split_records(args.artifact_root),
        "corpora": corpus_records(args.artifact_root),
        "manifests": manifest_records(args.artifact_root),
        "effects": effects,
        "retrieval_audit": retrieval,
        "risk_statement": risk_statement(effects, retrieval),
    }
    Draft202012Validator(schema_document).validate(certificate)
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(certificate["risk_statement"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
