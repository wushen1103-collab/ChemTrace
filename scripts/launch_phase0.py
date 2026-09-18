#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch phase-0 jobs across free GPUs.")
    p.add_argument("--tasks", nargs="+", default=["esol", "bace"])
    p.add_argument("--conditions", nargs="+", default=["clean", "exact_20x", "random_20x"])
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    p.add_argument("--gpus", nargs="+", default=["auto"], help="'auto' or explicit ids")
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0"))
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


def main() -> None:
    args = parse_args()
    gpus = auto_gpus() if args.gpus == ["auto"] else args.gpus
    jobs = queue.Queue()
    for seed in args.seeds:
        for task in args.tasks:
            for condition in args.conditions:
                jobs.put((task, condition, seed))

    Path("logs").mkdir(exist_ok=True)
    active: list[tuple[subprocess.Popen, str, tuple[str, str, int]]] = []
    gpu_slots = [gpu for gpu in gpus for _ in range(args.max_parallel_per_gpu)]
    print(f"Using slots: {gpu_slots}", flush=True)

    while not jobs.empty() or active:
        active = [(p, gpu, meta) for p, gpu, meta in active if p.poll() is None]
        while not jobs.empty() and len(active) < len(gpu_slots):
            used = {gpu for _, gpu, _ in active}
            gpu = next((g for g in gpu_slots if g not in used), gpu_slots[len(active) % len(gpu_slots)])
            task, condition, seed = jobs.get()
            log_prefix = args.artifact_root.name
            log = Path("logs") / f"{log_prefix}_{task}_{condition}_seed{seed}.log"
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
    for log in Path("logs").glob("phase0_*.log"):
        txt = log.read_text(encoding="utf-8", errors="ignore")
        if "Traceback" in txt or "RuntimeError" in txt:
            failed.append(str(log))
    if failed:
        raise SystemExit("Failed logs: " + ", ".join(failed))
    print("All launched jobs completed.")


if __name__ == "__main__":
    main()
