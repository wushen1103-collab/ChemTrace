#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safely resume the paused BBBP low-resource Phase-0 job when GPUs are free.")
    p.add_argument("--jobs-csv", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--gpu-max", type=int, default=2)
    p.add_argument("--memory-used-max-mb", type=int, default=3500)
    p.add_argument("--util-max", type=int, default=20)
    p.add_argument("--stable-checks", type=int, default=2)
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--external-memory-stop-mb", type=int, default=7000)
    p.add_argument("--launcher-log", type=Path, default=Path("logs/phase0_lowtrain25_auto_launcher.log"))
    p.add_argument("--status-json", type=Path, default=Path("logs/phase0_lowtrain25_watcher_status.json"))
    p.add_argument("--pretrain-epochs", type=int, default=4)
    p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dim", type=int, default=160)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--encoder-type", default="transformer")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.25)
    return p.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"command failed: {' '.join(cmd)}")
    return proc.stdout


def gpu_snapshot() -> list[dict[str, object]]:
    text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        idx, uuid, mem, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "uuid": uuid, "memory_used_mb": int(mem), "util_gpu": int(util)})
    return rows


def compute_apps() -> list[dict[str, object]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        uuid, pid, name, mem = parts
        rows.append({"uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mb": int(mem)})
    return rows


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
    except OSError:
        return ""


def predictions_done(artifact_root: Path) -> int:
    pred_dir = artifact_root / "predictions"
    return len(list(pred_dir.glob("*.csv"))) if pred_dir.exists() else 0


def jobs_total(jobs_csv: Path) -> int:
    with jobs_csv.open("r", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def write_status(path: Path, status: str, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": now(), "status": status, **extra}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


def free_gpu_ids(args: argparse.Namespace) -> list[str]:
    free = []
    for gpu in gpu_snapshot():
        if gpu["memory_used_mb"] <= args.memory_used_max_mb and gpu["util_gpu"] <= args.util_max:
            free.append(str(gpu["index"]))
    return free[: args.gpu_max]


def heavy_external_on_selected(selected: set[str], artifact_name: str, external_memory_stop_mb: int) -> list[dict[str, object]]:
    snapshot = gpu_snapshot()
    uuid_to_index = {str(row["uuid"]): str(row["index"]) for row in snapshot}
    offenders = []
    for app in compute_apps():
        gpu_idx = uuid_to_index.get(str(app["uuid"]))
        if gpu_idx not in selected or int(app["used_memory_mb"]) < external_memory_stop_mb:
            continue
        proc_cmd = cmdline(int(app["pid"]))
        if artifact_name not in proc_cmd and "launch_phase0_job_list.py" not in proc_cmd:
            offenders.append({**app, "gpu_index": gpu_idx, "cmdline": proc_cmd[:240]})
    return offenders


def launch(args: argparse.Namespace, gpus: list[str]) -> int:
    args.launcher_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/launch_phase0_job_list.py",
        "--jobs-csv",
        str(args.jobs_csv),
        "--artifact-root",
        str(args.artifact_root),
        "--gpus",
        *gpus,
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
        "--skip-existing",
    ]
    with args.launcher_log.open("ab", buffering=0) as log:
        log.write(f"\n[{now()}] watcher launching on GPUs {','.join(gpus)}\n".encode("utf-8"))
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        selected = set(gpus)
        while proc.poll() is None:
            time.sleep(args.poll_seconds)
            offenders = heavy_external_on_selected(selected, args.artifact_root.name, args.external_memory_stop_mb)
            if offenders:
                write_status(args.status_json, "paused_due_gpu_reoccupation", gpus=gpus, offenders=offenders)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=20)
                return 130
        return int(proc.returncode)


def main() -> None:
    args = parse_args()
    total = jobs_total(args.jobs_csv)
    stable = 0
    last_free: list[str] = []
    write_status(args.status_json, "watching", predictions_done=predictions_done(args.artifact_root), jobs_total=total)
    while predictions_done(args.artifact_root) < total:
        free = free_gpu_ids(args)
        if free:
            stable += 1
            last_free = free
        else:
            stable = 0
            last_free = []
        write_status(
            args.status_json,
            "waiting_for_free_gpu",
            free_gpus=last_free,
            stable_checks=stable,
            predictions_done=predictions_done(args.artifact_root),
            jobs_total=total,
        )
        if stable >= args.stable_checks and last_free:
            rc = launch(args, last_free)
            write_status(
                args.status_json,
                "launcher_finished" if rc == 0 else "launcher_stopped",
                return_code=rc,
                gpus=last_free,
                predictions_done=predictions_done(args.artifact_root),
                jobs_total=total,
            )
            if rc != 0:
                stable = 0
                time.sleep(args.poll_seconds)
                continue
        time.sleep(args.poll_seconds)
    write_status(args.status_json, "complete", predictions_done=predictions_done(args.artifact_root), jobs_total=total)


if __name__ == "__main__":
    main()
