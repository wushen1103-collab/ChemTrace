from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem
from rdkit import RDLogger
from torch import nn


RDLogger.DisableLog("rdApp.warning")

ATOM_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "B",
    "Si",
    "Se",
    "Na",
    "K",
    "Li",
    "Ca",
    "Mg",
    "Zn",
    "Fe",
    "Cu",
    "Al",
]


@dataclass
class AtomVocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build_default(cls) -> "AtomVocab":
        itos = ["<unk>", "<mask>"] + ATOM_SYMBOLS
        return cls({s: i for i, s in enumerate(itos)}, itos)

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def mask_id(self) -> int:
        return self.stoi["<mask>"]

    def encode_symbol(self, symbol: str) -> int:
        return self.stoi.get(symbol, self.unk_id)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AtomVocab":
        itos = json.loads(path.read_text(encoding="utf-8"))["itos"]
        return cls({s: i for i, s in enumerate(itos)}, itos)


def smiles_to_graph(smiles: str, vocab: AtomVocab) -> tuple[torch.Tensor, torch.Tensor]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.tensor([vocab.unk_id], dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return x, edge_index

    x = torch.tensor([vocab.encode_symbol(atom.GetSymbol()) for atom in mol.GetAtoms()], dtype=torch.long)
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        edges.append((a, b))
        edges.append((b, a))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return x, edge_index


class GinEncoder(nn.Module):
    def __init__(self, atom_vocab_size: int, dim: int = 256, layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.output_dim = dim
        self.atom_embed = nn.Embedding(atom_vocab_size, dim)
        self.eps = nn.Parameter(torch.zeros(layers))
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(layers):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                )
            )
            self.norms.append(nn.LayerNorm(dim))

    def forward_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.atom_embed(x)
        for i, (mlp, norm) in enumerate(zip(self.mlps, self.norms)):
            if edge_index.numel():
                src, dst = edge_index
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src])
            else:
                agg = torch.zeros_like(h)
            h = mlp((1.0 + self.eps[i]) * h + agg)
            h = self.dropout(torch.relu(norm(h)))
        return h

    def pooled(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, num_graphs: int | None = None) -> torch.Tensor:
        h = self.forward_nodes(x, edge_index)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        out = torch.zeros((num_graphs, h.shape[-1]), dtype=h.dtype, device=h.device)
        out.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).to(h.device, dtype=h.dtype).clamp_min(1.0).unsqueeze(-1)
        return out / counts


class MaskedAtomModel(nn.Module):
    def __init__(self, encoder: GinEncoder, atom_vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.output_dim, atom_vocab_size)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.forward_nodes(x, edge_index))


class GraphTaskHead(nn.Module):
    def __init__(self, encoder: GinEncoder, task_type: str):
        super().__init__()
        self.encoder = encoder
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(encoder.output_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        pooled = self.encoder.pooled(x, edge_index, batch, num_graphs=num_graphs)
        return self.head(pooled).squeeze(-1)
