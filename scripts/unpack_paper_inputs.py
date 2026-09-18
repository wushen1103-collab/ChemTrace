#!/usr/bin/env python
"""Verify and safely unpack the fixed paper-input archive."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


EXPECTED_SHA256 = "868d13368c936973bd5925fed7a5fbbca90c6d54288d14d8207c03237d985ebe"
DEFAULT_ARCHIVE = Path("data/paper_inputs.tgz")
DEFAULT_OUTPUT = Path("artifacts/phase0_tf_confirm_top_len158_extra")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(bundle: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = bundle.getmembers()
    for member in members:
        parts = Path(member.name.replace("\\", "/")).parts
        if member.name.startswith(("/", "\\")) or ".." in parts:
            raise ValueError(f"Unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not permitted in the input archive: {member.name}")
    return members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory after a verified extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)

    observed = sha256(archive)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"Archive SHA-256 mismatch: expected {EXPECTED_SHA256}, observed {observed}")

    marker = output / "prepare_meta.json"
    if marker.is_file() and not args.force:
        print(f"Inputs already unpacked at {output}")
        return
    if output.exists() and not args.force:
        raise FileExistsError(f"Output exists but is incomplete: {output}. Use --force to replace it.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chemtrace-inputs-", dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = safe_members(bundle)
            if sys.version_info >= (3, 12):
                bundle.extractall(staging, members=members, filter="data")
            else:
                bundle.extractall(staging, members=members)
        if not (staging / "prepare_meta.json").is_file():
            raise RuntimeError("Archive is missing prepare_meta.json")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)

    print(f"Verified {archive.name}: {observed}")
    print(f"Unpacked fixed inputs to {output}")


if __name__ == "__main__":
    main()
