#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch selected Phase-0 jobs across free GPUs.")
    p.add_argument("--jobs-csv", type=Path, required=True, help="CSV with task,condition,seed columns")
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0"))
    p.add_argument("--gpus", nargs="+", default=["auto"], help="'auto' or explicit ids")
    p.add_argument("--max-parallel-per-gpu", type=int, default=1)
    p.add_argument("--pretrain-epochs", type=int, default=4)
    p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dim", type=int, default=160)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--encoder-type", choices=["gru", "transformer"], default="gru")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def auto_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return ["cpu"]
    ids = []
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        if int(mem) < 1024 and int(util) < 20:
            ids.append(idx)
    return ids or ["0"]


def read_jobs(path: Path) -> list[tuple[str, str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"task", "condition", "seed"}
        missing = needed.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"jobs CSV missing columns: {sorted(missing)}")
        jobs = [(str(r["task"]), str(r["condition"]), int(r["seed"])) for r in reader]
    if not jobs:
        raise ValueError(f"No jobs found in {path}")
    return jobs


def main() -> None:
    args = parse_args()
    gpus = auto_gpus() if args.gpus == ["auto"] else args.gpus
    slots = [gpu for gpu in gpus for _ in range(args.max_parallel_per_gpu)]
    if not slots:
        slots = ["cpu"]

    jobs = queue.Queue()
    for task, condition, seed in read_jobs(args.jobs_csv):
        pred = args.artifact_root / "predictions" / f"{task}__{condition}__seed{seed}.csv"
        if args.skip_existing and pred.exists():
            continue
        jobs.put((task, condition, seed))

    Path("logs").mkdir(exist_ok=True)
    active: list[tuple[subprocess.Popen, str, tuple[str, str, int]]] = []
    print(f"Using slots: {slots}", flush=True)
    print(f"Queued jobs: {jobs.qsize()}", flush=True)

    while not jobs.empty() or active:
        active = [(p, gpu, meta) for p, gpu, meta in active if p.poll() is None]
        while not jobs.empty() and len(active) < len(slots):
            used = {gpu for _, gpu, _ in active}
            gpu = next((g for g in slots if g not in used), slots[len(active) % len(slots)])
            task, condition, seed = jobs.get()
            log = Path("logs") / f"{args.artifact_root.name}_{task}_{condition}_seed{seed}.log"
            cmd = [
                sys.executable,
                "scripts/phase0_run_one.py",
                "--task",
                task,
                "--condition",
                condition,
                "--seed",
                str(seed),
                "--artifact-root",
                str(args.artifact_root),
                "--pretrain-epochs",
                str(args.pretrain_epochs),
                "--finetune-epochs",
                str(args.finetune_epochs),
                "--batch-size",
                str(args.batch_size),
                "--dim",
                str(args.dim),
                "--layers",
                str(args.layers),
                "--encoder-type",
                args.encoder_type,
                "--heads",
                str(args.heads),
                "--num-workers",
                str(args.num_workers),
                "--train-frac",
                str(args.train_frac),
            ]
            env = os.environ.copy()
            root = str(Path.cwd())
            env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            if gpu != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd += ["--device", "cuda"]
            else:
                cmd += ["--device", "cpu"]
            print("START", gpu, task, condition, seed, "log", log, flush=True)
            if args.dry_run:
                continue
            f = log.open("w", encoding="utf-8")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            active.append((p, gpu, (task, condition, seed)))
        if active:
            time.sleep(15)

    failed = []
    for _, _, _ in read_jobs(args.jobs_csv):
        pass
    for log in Path("logs").glob(f"{args.artifact_root.name}_*.log"):
        txt = log.read_text(encoding="utf-8", errors="ignore")
        if "Traceback" in txt or "RuntimeError" in txt or "CUDA out of memory" in txt:
            failed.append(str(log))
    if failed:
        raise SystemExit("Failed logs: " + ", ".join(failed))
    print("All selected jobs completed.")


if __name__ == "__main__":
    main()
