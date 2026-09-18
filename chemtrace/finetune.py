from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .torch_models import SmilesVocab, TaskHead, build_encoder
from .util import seed_all, stable_float_key


class TaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: SmilesVocab, max_len: int):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = torch.tensor(self.vocab.encode(row["canonical"], self.max_len), dtype=torch.long)
        y = torch.tensor(float(row["y"]), dtype=torch.float32)
        return x, y, idx


def _eval(model: TaskHead, loader: DataLoader, task_type: str, device: torch.device):
    model.eval()
    ys, preds, losses, idxs = [], [], [], []
    if task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            pred = torch.sigmoid(logits) if task_type == "classification" else logits
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            losses.extend(loss.cpu().numpy().tolist())
            idxs.extend(idx.numpy().tolist())
    return np.asarray(ys), np.asarray(preds), np.asarray(losses), np.asarray(idxs)


def _metrics(y: np.ndarray, pred: np.ndarray, loss: np.ndarray, task_type: str) -> dict:
    out = {"loss": float(np.mean(loss))}
    if task_type == "classification":
        labels = (y > 0.5).astype(int)
        out["auprc"] = float(average_precision_score(labels, pred)) if len(set(labels)) > 1 else float("nan")
        out["auroc"] = float(roc_auc_score(labels, pred)) if len(set(labels)) > 1 else float("nan")
    else:
        mse = mean_squared_error(y, pred)
        out["rmse"] = float(math.sqrt(mse))
        out["mae"] = float(mean_absolute_error(y, pred))
        out["r2"] = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    return out


def finetune_task(
    split_path: Path,
    checkpoint_dir: Path,
    pred_path: Path,
    seed: int,
    device: str | None,
    epochs: int,
    batch_size: int,
    max_len: int,
    lr: float,
    patience: int,
    num_workers: int,
    train_frac: float = 1.0,
) -> dict:
    seed_all(seed)
    df = pd.read_csv(split_path)
    task_type = str(df["task_type"].iloc[0])
    vocab = SmilesVocab.load(checkpoint_dir / "vocab.json")
    ckpt = torch.load(checkpoint_dir / "pretrained_encoder.pt", map_location="cpu")
    dim = int(ckpt["config"]["dim"])
    layers = int(ckpt["config"]["layers"])
    encoder_type = str(ckpt["config"].get("encoder_type", "gru"))
    heads = int(ckpt["config"].get("heads", 4))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = build_encoder(encoder_type, len(vocab.itos), vocab.pad_id, dim=dim, layers=layers, max_len=max_len, heads=heads)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    model = TaskHead(encoder, task_type).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if task_type == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    loaders = {}
    n_train_used = 0
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].reset_index(drop=True)
        if split == "train" and train_frac < 1.0:
            part = _subsample_train(part, train_frac, seed, task_type)
        if split == "train":
            n_train_used = len(part)
        loaders[split] = DataLoader(TaskDataset(part, vocab, max_len), batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, _ in loaders["train"]:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        yv, pv, lv, _ = _eval(model, loaders["val"], task_type, dev)
        val = float(np.mean(lv))
        history.append({"epoch": epoch, "val_loss": val})
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)

    all_preds = []
    metrics = {"best_val_loss": best_val, "epochs_ran": len(history), "history": history, "train_frac": train_frac, "n_train_used": n_train_used}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].reset_index(drop=True)
        y, pred, loss, idxs = _eval(model, loaders[split], task_type, dev)
        m = _metrics(y, pred, loss, task_type)
        metrics[f"{split}_metrics"] = m
        tmp = part.iloc[idxs].copy()
        tmp["prediction"] = pred
        tmp["sample_loss"] = loss
        all_preds.append(tmp)
    out = pd.concat(all_preds, ignore_index=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    torch.save(best_state, checkpoint_dir / "finetuned_task_head.pt")
    (checkpoint_dir / "finetune_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _subsample_train(part: pd.DataFrame, train_frac: float, seed: int, task_type: str) -> pd.DataFrame:
    train_frac = max(0.0, min(1.0, train_frac))
    if train_frac >= 1.0 or part.empty:
        return part
    keyed = part.assign(_key=part["sample_id"].map(lambda x: stable_float_key(str(x), seed)))
    if task_type == "classification":
        frames = []
        for _, group in keyed.groupby("y", dropna=False):
            n = max(1, int(round(len(group) * train_frac)))
            frames.append(group.sort_values("_key").head(n))
        out = pd.concat(frames, ignore_index=True)
    else:
        n = max(1, int(round(len(keyed) * train_frac)))
        out = keyed.sort_values("_key").head(n)
    return out.drop(columns=["_key"]).reset_index(drop=True)
