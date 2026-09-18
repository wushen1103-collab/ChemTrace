from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .finetune import _subsample_train
from .graph_models import AtomVocab, GinEncoder, GraphTaskHead, MaskedAtomModel, smiles_to_graph
from .util import ensure_dir, seed_all, sha256_file


def _read_smiles(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class MaskedAtomDataset(Dataset):
    def __init__(self, smiles: list[str], vocab: AtomVocab, seed: int, mask_prob: float = 0.15):
        self.smiles = smiles
        self.vocab = vocab
        self.seed = seed
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        x, edge_index = smiles_to_graph(self.smiles[idx], self.vocab)
        gen = torch.Generator().manual_seed(self.seed + idx)
        labels = x.clone()
        mask = torch.rand(x.shape, generator=gen) < self.mask_prob
        if not bool(mask.any()) and len(mask) > 0:
            chosen = int(torch.randint(0, len(mask), (1,), generator=gen).item())
            mask[chosen] = True
        x_masked = x.clone()
        x_masked[mask] = self.vocab.mask_id
        labels[~mask] = -100
        return x_masked, edge_index, labels


class GraphTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: AtomVocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x, edge_index = smiles_to_graph(row["canonical"], self.vocab)
        y = torch.tensor(float(row["y"]), dtype=torch.float32)
        return x, edge_index, y, idx


def _collate_masked(batch):
    xs, edges, labels = zip(*batch)
    node_counts = [len(x) for x in xs]
    offsets = np.cumsum([0] + node_counts[:-1]).tolist()
    x = torch.cat(xs, dim=0)
    y = torch.cat(labels, dim=0)
    shifted = []
    batch_idx = []
    for i, (edge_index, offset, n_nodes) in enumerate(zip(edges, offsets, node_counts)):
        if edge_index.numel():
            shifted.append(edge_index + int(offset))
        batch_idx.extend([i] * n_nodes)
    edge_index = torch.cat(shifted, dim=1) if shifted else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index, torch.tensor(batch_idx, dtype=torch.long), y, len(batch)


def _collate_task(batch):
    xs, edges, ys, idxs = zip(*batch)
    node_counts = [len(x) for x in xs]
    offsets = np.cumsum([0] + node_counts[:-1]).tolist()
    x = torch.cat(xs, dim=0)
    y = torch.stack(ys)
    shifted = []
    batch_idx = []
    for i, (edge_index, offset, n_nodes) in enumerate(zip(edges, offsets, node_counts)):
        if edge_index.numel():
            shifted.append(edge_index + int(offset))
        batch_idx.extend([i] * n_nodes)
    edge_index = torch.cat(shifted, dim=1) if shifted else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index, torch.tensor(batch_idx, dtype=torch.long), y, torch.tensor(idxs, dtype=torch.long), len(batch)


def pretrain_gin(
    corpus_path: Path,
    out_dir: Path,
    seed: int,
    device: str | None,
    epochs: int,
    batch_size: int,
    dim: int,
    layers: int,
    lr: float,
    num_workers: int,
) -> dict:
    ensure_dir(out_dir)
    seed_all(seed)
    smiles = _read_smiles(corpus_path)
    vocab = AtomVocab.build_default()
    vocab.save(out_dir / "atom_vocab.json")
    ds = MaskedAtomDataset(smiles, vocab, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=_collate_masked)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = GinEncoder(len(vocab.itos), dim=dim, layers=layers)
    model = MaskedAtomModel(encoder, len(vocab.itos)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        denom = 0
        for x, edge_index, _, labels, _ in tqdm(loader, desc=f"graph pretrain epoch {epoch}", leave=False):
            x = x.to(dev, non_blocking=True)
            edge_index = edge_index.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x, edge_index)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int((labels != -100).sum().detach().cpu())
            total += float(loss.detach().cpu()) * max(1, n)
            denom += max(1, n)
        history.append({"epoch": epoch, "loss": total / max(1, denom)})
    ckpt = {
        "encoder": model.encoder.state_dict(),
        "config": {
            "encoder_type": "gin",
            "dim": dim,
            "layers": layers,
            "atom_vocab_size": len(vocab.itos),
        },
        "seed": seed,
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
    }
    torch.save(ckpt, out_dir / "pretrained_graph_encoder.pt")
    info = {"epochs": epochs, "history": history, "seconds": time.time() - start, "n_smiles": len(smiles)}
    (out_dir / "graph_pretrain_metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _eval_graph(model: GraphTaskHead, loader: DataLoader, task_type: str, device: torch.device):
    model.eval()
    ys, preds, losses, idxs = [], [], [], []
    loss_fn = nn.BCEWithLogitsLoss(reduction="none") if task_type == "classification" else nn.MSELoss(reduction="none")
    with torch.no_grad():
        for x, edge_index, batch_idx, y, idx, num_graphs in loader:
            x = x.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x, edge_index, batch_idx, int(num_graphs))
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


def finetune_graph_task(
    split_path: Path,
    checkpoint_dir: Path,
    pred_path: Path,
    seed: int,
    device: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    num_workers: int,
    train_frac: float = 1.0,
) -> dict:
    seed_all(seed)
    df = pd.read_csv(split_path)
    task_type = str(df["task_type"].iloc[0])
    vocab = AtomVocab.load(checkpoint_dir / "atom_vocab.json")
    ckpt = torch.load(checkpoint_dir / "pretrained_graph_encoder.pt", map_location="cpu")
    dim = int(ckpt["config"]["dim"])
    layers = int(ckpt["config"]["layers"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = GinEncoder(len(vocab.itos), dim=dim, layers=layers)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    model = GraphTaskHead(encoder, task_type).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()

    loaders = {}
    parts = {}
    n_train_used = 0
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].reset_index(drop=True)
        if split == "train" and train_frac < 1.0:
            part = _subsample_train(part, train_frac, seed, task_type)
        if split == "train":
            n_train_used = len(part)
        parts[split] = part
        loaders[split] = DataLoader(
            GraphTaskDataset(part, vocab),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_task,
        )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, edge_index, batch_idx, y, _, num_graphs in loaders["train"]:
            x = x.to(dev, non_blocking=True)
            edge_index = edge_index.to(dev, non_blocking=True)
            batch_idx = batch_idx.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x, edge_index, batch_idx, int(num_graphs))
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        _, _, lv, _ = _eval_graph(model, loaders["val"], task_type, dev)
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
        y, pred, loss, idxs = _eval_graph(model, loaders[split], task_type, dev)
        m = _metrics(y, pred, loss, task_type)
        metrics[f"{split}_metrics"] = m
        tmp = parts[split].iloc[idxs].copy()
        tmp["prediction"] = pred
        tmp["sample_loss"] = loss
        all_preds.append(tmp)
    out = pd.concat(all_preds, ignore_index=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    torch.save(best_state, checkpoint_dir / "finetuned_graph_task_head.pt")
    (checkpoint_dir / "graph_finetune_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
