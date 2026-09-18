from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_float_key(text: str, seed: int = 0) -> float:
    raw = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) / float(16**16)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def set_cpu_budget(reserve: int = 30) -> int:
    total = os.cpu_count() or 32
    allowed = max(1, total - reserve)
    current = int(os.environ.get("CHEMTRACE_CPU_WORKERS", min(allowed, 32)))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"]:
        os.environ.setdefault(name, str(max(1, min(current, allowed))))
    return allowed


def write_lines(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")
