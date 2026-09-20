#!/usr/bin/env python
"""Check archive integrity, numerical contracts, schema validity, and privacy."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import tarfile
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "data/paper_inputs.tgz": "868d13368c936973bd5925fed7a5fbbca90c6d54288d14d8207c03237d985ebe",
    "reports/tables/phase0_independent_external_gold_v2/independent_relation_gold_pairs.csv": "97e14faff27e4a3476d331052b81fb82947c09e130c02644d6a667b5dfac4721",
    "reports/tables/phase0_independent_external_gold_v2/independent_registry_negative_pairs.csv": "a3bc508c2fb080db7df39c34e2f1ac8065911b1df2737bb16e86d62a59dff1a3",
    "reports/tables/phase0_reviewer_supplements/hard_negative_pair_bank.csv": "df6313d1759a5007334ec1f24eef92dd89c79f9c9b3a91f0f4cd0948a8aec2bf",
    "reports/tables/phase0_reviewer_supplements/hard_negative_retrieval_metrics.csv": "0d1e9ee452fb05b07ca41124a46e5feb6b77e8bd2776f766bd9300b09f16bb16",
    "reports/tables/phase0_reviewer_supplements/hard_negative_scenario_breakdown.csv": "b831b2291d85babf27f4979abb0fd7bc4cb0e82d4450fae09ccce08787431c70",
    "reports/tables/phase0_reviewer_supplements/normalization_path_stability_summary.csv": "351dc6e76ca99c5f67af18bb9736e81b2abfa55f3e1fea25d24cbce76141562c",
    "reports/tables/phase0_reviewer_supplements/retrieval_scale_benchmark.csv": "33d642c4bc379a80842c30ca0cbc9d6bcc1e42f346d6f8b5a56a06b35d2497a5",
    "reports/tables/phase0_reviewer_supplements/tautomer_miss_taxonomy.csv": "1fe650f81b0d580d4b69428fd6a4472245c8c46511ead3ff79e720dc51618f56",
    "reports/tables/revision_retrieval/superparent_common_protocol_cells.csv": "a1adb6ae0d1106c1a07a5969223eddd443d811b4997d188212938395e4d1a5d8",
    "reports/tables/revision_retrieval/superparent_common_protocol_summary.csv": "eeaf47599179ea7ed6bee653c56c707477681a35ea89a3e0cc4f95369336fa1a",
}
TEXT_EXTENSIONS = {".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".toml", ".txt", ".yml", ".yaml"}
PRIVATE_PATTERNS = {
    "Unix home path": re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "local drive path": re.compile(r"\b[G-Z]:\\", re.IGNORECASE),
    "private-network address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "known local account": re.compile(r"(?:Administrator|test@10\.|/kkkk/|wushen1103)", re.IGNORECASE),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "Windows machine name": re.compile(r"\bDESKTOP-[A-Z0-9]+\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_hashes(errors: list[str]) -> None:
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing release file: {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            errors.append(f"SHA-256 mismatch for {relative}: {observed}")


def check_numerical_contracts(errors: list[str]) -> None:
    rows = read_csv("reports/tables/phase0_retrieval_sota_summary/retrieval_sota_bootstrap_summary.csv")
    row = next(
        (r for r in rows if r["source"] == "confirm_top_sample5k" and r["relation_group"] == "semantic_all"),
        None,
    )
    if row is None:
        errors.append("Missing Figure 2 semantic_all summary row")
    else:
        lattice = float(row["lattice_changed_recall_mean"])
        similarity_raw = float(row["best_similarity_changed_recall_mean"])
        similarity_net = float(row["best_similarity_net_changed_recall_mean"])
        gain = float(row["lattice_gain_vs_similarity_mean"])
        if not math.isclose(gain, lattice - similarity_net, abs_tol=1e-12):
            errors.append("Figure 2 gain does not equal lattice recall minus the net comparator recall")
        expected = (1.0, 0.8410118889098348, 0.8408115683970143, 0.15918843160298554)
        observed = (lattice, similarity_raw, similarity_net, gain)
        if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(observed, expected)):
            errors.append(f"Figure 2 values changed: {observed}")

    cells = read_csv("reports/tables/phase0_modern_sota_baselines/modern_sota_method_repeat_results.csv")
    for index, cell in enumerate(cells, start=2):
        if not cell["background_and_decoy_adjusted_changed_recall"]:
            continue
        rec = float(cell["changed_only_recall"])
        background = float(cell["clean_background_hit_rate"])
        decoy = float(cell["decoy_false_positive_proxy"])
        adjusted = float(cell["background_and_decoy_adjusted_changed_recall"])
        if not math.isclose(adjusted, rec - background - decoy, abs_tol=1e-12):
            errors.append(f"Table 2 cellwise score identity fails at CSV row {index}")
            break

    summary = read_csv("reports/tables/phase0_modern_sota_baselines/modern_sota_method_summary.csv")
    for row in summary:
        if row["cells_per_repeat"] != "81" or row["changed_cells_per_repeat"] != "78":
            errors.append(f"Unexpected aggregation counts for {row['method']}")


def check_certificate(errors: list[str]) -> None:
    schema_path = ROOT / "schemas/chemtrace-certificate.schema.json"
    certificate_path = ROOT / "reports/certificates/phase0_tf_bbbp_stereo_changed_aff82_d20.final-manuscript.certificate.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("schema_sha256") != sha256(schema_path):
        errors.append("Certificate schema_sha256 does not match the released schema")
    validation_errors = sorted(Draft202012Validator(schema).iter_errors(certificate), key=lambda err: list(err.path))
    for error in validation_errors:
        errors.append(f"Certificate schema error at {list(error.path)}: {error.message}")


def scan_text(name: str, text: str, errors: list[str]) -> None:
    for label, pattern in PRIVATE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            errors.append(f"{label} found in {name}: {match.group(0)!r}")


def check_privacy(errors: list[str]) -> None:
    excluded_parts = {".git", ".venv", "artifacts", "reproduced", "__pycache__", ".pytest_cache"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded_parts for part in path.relative_to(ROOT).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            scan_text(path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8", errors="replace"), errors)

    archive = ROOT / "data/paper_inputs.tgz"
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            scan_text(f"archive member name {member.name}", member.name, errors)
            if not member.isfile() or Path(member.name).suffix.lower() not in TEXT_EXTENSIONS:
                continue
            handle = bundle.extractfile(member)
            if handle is not None:
                scan_text(f"archive member {member.name}", handle.read().decode("utf-8", errors="replace"), errors)


def main() -> None:
    errors: list[str] = []
    check_hashes(errors)
    check_numerical_contracts(errors)
    check_certificate(errors)
    check_privacy(errors)
    if errors:
        print("Release check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Release check passed: {len(EXPECTED_HASHES)} hashes, numerical contracts, schema, and privacy scan.")


if __name__ == "__main__":
    main()
