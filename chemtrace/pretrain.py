from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .torch_models import MaskedLM, SmilesVocab, build_encoder
from .util import ensure_dir, seed_all, sha256_file


class MaskedDataset(Dataset):
    def __init__(self, smiles: list[str], vocab: SmilesVocab, max_len: int, seed: int, mask_prob: float = 0.15):
        self.smiles = smiles
        self.vocab = vocab
        self.max_len = max_len
        self.seed = seed
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        gen = torch.Generator().manual_seed(self.seed + idx)
        ids = torch.tensor(self.vocab.encode(self.smiles[idx], self.max_len), dtype=torch.long)
        labels = ids.clone()
        rand = torch.rand(ids.shape, generator=gen)
        special = (ids == self.vocab.pad_id) | (ids == self.vocab.stoi["<bos>"]) | (ids == self.vocab.stoi["<eos>"])
        mask = (rand < self.mask_prob) & (~special)
        ids[mask] = self.vocab.mask_id
        labels[~mask] = -100
        return ids, labels


def _read_smiles(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pretrain_lm(
    corpus_path: Path,
    out_dir: Path,
    seed: int,
    device: str | None,
    epochs: int,
    batch_size: int,
    max_len: int,
    dim: int,
    layers: int,
    encoder_type: str,
    heads: int,
    lr: float,
    num_workers: int,
) -> dict:
    ensure_dir(out_dir)
    seed_all(seed)
    smiles = _read_smiles(corpus_path)
    vocab = SmilesVocab.build(smiles)
    vocab.save(out_dir / "vocab.json")
    ds = MaskedDataset(smiles, vocab, max_len=max_len, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = build_encoder(encoder_type, len(vocab.itos), vocab.pad_id, dim=dim, layers=layers, max_len=max_len, heads=heads)
    model = MaskedLM(encoder, len(vocab.itos)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        denom = 0
        for x, y in tqdm(loader, desc=f"pretrain epoch {epoch}", leave=False):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * x.shape[0]
            denom += x.shape[0]
        history.append({"epoch": epoch, "loss": total / max(1, denom)})
    ckpt = {
        "encoder": model.encoder.state_dict(),
        "config": {
            "encoder_type": encoder_type,
            "dim": dim,
            "layers": layers,
            "heads": heads,
            "max_len": max_len,
            "pad_id": vocab.pad_id,
            "vocab_size": len(vocab.itos),
        },
        "seed": seed,
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
    }
    torch.save(ckpt, out_dir / "pretrained_encoder.pt")
    info = {"epochs": epochs, "history": history, "seconds": time.time() - start, "n_smiles": len(smiles)}
    (out_dir / "pretrain_metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
