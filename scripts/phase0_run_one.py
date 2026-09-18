#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemtrace.finetune import finetune_task
from chemtrace.pretrain import pretrain_lm
from chemtrace.util import ensure_dir, set_cpu_budget


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one paired ChemTrace phase-0 condition.")
    p.add_argument("--task", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase0"))
    p.add_argument("--device", default=None)
    p.add_argument("--pretrain-epochs", type=int, default=4)
    p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--dim", type=int, default=160)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--encoder-type", choices=["gru", "transformer"], default="gru")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_cpu_budget()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    corpus_path = args.artifact_root / "corpora" / f"{args.condition}.txt"
    split_path = args.artifact_root / "splits" / f"{args.task}.csv"
    run_id = f"{args.task}__{args.condition}__seed{args.seed}"
    ckpt_dir = ensure_dir(args.artifact_root / "checkpoints" / run_id)
    pred_dir = ensure_dir(args.artifact_root / "predictions")

    pretrain_info = pretrain_lm(
        corpus_path=corpus_path,
        out_dir=ckpt_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        dim=args.dim,
        layers=args.layers,
        encoder_type=args.encoder_type,
        heads=args.heads,
        lr=args.lr,
        num_workers=args.num_workers,
    )

    pred_path = pred_dir / f"{run_id}.csv"
    metrics = finetune_task(
        split_path=split_path,
        checkpoint_dir=ckpt_dir,
        pred_path=pred_path,
        seed=args.seed,
        device=args.device,
        epochs=args.finetune_epochs,
        batch_size=max(64, args.batch_size // 2),
        max_len=args.max_len,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
    )

    summary = {
        "run_id": run_id,
        "task": args.task,
        "condition": args.condition,
        "seed": args.seed,
        "pretrain": pretrain_info,
        "finetune": metrics,
        "prediction_path": str(pred_path),
    }
    with (ckpt_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
