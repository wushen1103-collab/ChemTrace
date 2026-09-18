from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


SPECIAL = ["<pad>", "<unk>", "<mask>", "<bos>", "<eos>"]


@dataclass
class SmilesVocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, smiles: list[str], min_freq: int = 1) -> "SmilesVocab":
        counts: dict[str, int] = {}
        for smi in smiles:
            for ch in smi:
                counts[ch] = counts.get(ch, 0) + 1
        chars = sorted([c for c, n in counts.items() if n >= min_freq])
        itos = SPECIAL + chars
        return cls({c: i for i, c in enumerate(itos)}, itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def mask_id(self) -> int:
        return self.stoi["<mask>"]

    def encode(self, smiles: str, max_len: int) -> list[int]:
        ids = [self.stoi["<bos>"]]
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in smiles[: max_len - 2])
        ids.append(self.stoi["<eos>"])
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids[:max_len]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SmilesVocab":
        itos = json.loads(path.read_text(encoding="utf-8"))["itos"]
        return cls({c: i for i, c in enumerate(itos)}, itos)


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, dim: int = 160, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.output_dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.gru(emb)
        return self.norm(self.proj(out))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.forward_tokens(x)
        mask = (x != self.pad_id).float().unsqueeze(-1)
        return (tok * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class TransformerSmilesEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        dim: int = 192,
        layers: int = 4,
        heads: int = 6,
        dropout: float = 0.1,
        max_len: int = 160,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.output_dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        emb = self.embed(x) * math.sqrt(self.output_dim)
        emb = emb + self.pos_embed(pos)
        pad_mask = x == self.pad_id
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        return self.norm(out)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.forward_tokens(x)
        mask = (x != self.pad_id).float().unsqueeze(-1)
        return (tok * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def build_encoder(
    encoder_type: str,
    vocab_size: int,
    pad_id: int,
    dim: int,
    layers: int,
    max_len: int,
    heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    if encoder_type == "gru":
        return SmilesEncoder(vocab_size, pad_id, dim=dim, layers=layers, dropout=dropout)
    if encoder_type == "transformer":
        return TransformerSmilesEncoder(
            vocab_size,
            pad_id,
            dim=dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
            max_len=max_len,
        )
    raise ValueError(f"unknown encoder_type: {encoder_type}")


class MaskedLM(nn.Module):
    def __init__(self, encoder: nn.Module, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.output_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.forward_tokens(x))


class TaskHead(nn.Module):
    def __init__(self, encoder: nn.Module, task_type: str):
        super().__init__()
        self.encoder = encoder
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(encoder.output_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.pooled(x)).squeeze(-1)
