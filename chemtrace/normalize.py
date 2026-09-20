from __future__ import annotations

from dataclasses import dataclass
import logging

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    MurckoScaffold = None
    rdMolStandardize = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedMol:
    raw: str
    canonical: str
    parent: str
    stereo_stripped: str
    scaffold: str
    valid: bool
    note: str = ""


def _mol(smiles: str):
    if Chem is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception as exc:
        LOGGER.warning("molecular parsing abstained after %s", type(exc).__name__)
        return None


def canonicalize(smiles: str, isomeric: bool = True) -> str:
    mol = _mol(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception as exc:
        LOGGER.warning("canonical normalization abstained after %s", type(exc).__name__)
        return ""


def parent_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None or rdMolStandardize is None:
        return ""
    try:
        chooser = rdMolStandardize.LargestFragmentChooser()
        uncharger = rdMolStandardize.Uncharger()
        parent = uncharger.uncharge(chooser.choose(mol))
        return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception as exc:
        LOGGER.warning("parent normalization abstained after %s", type(exc).__name__)
        return ""


def stereo_stripped_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None:
        return ""
    try:
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception as exc:
        LOGGER.warning("stereo normalization abstained after %s", type(exc).__name__)
        return ""


def tautomer_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None or rdMolStandardize is None:
        return ""
    try:
        enum = rdMolStandardize.TautomerEnumerator()
        taut = enum.Canonicalize(mol)
        return Chem.MolToSmiles(taut, canonical=True, isomericSmiles=True)
    except Exception as exc:
        LOGGER.warning("tautomer normalization abstained after %s", type(exc).__name__)
        return ""


def random_smiles(smiles: str, seed: int = 0) -> str:
    mol = _mol(smiles)
    if mol is None:
        return smiles
    try:
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
    except Exception as exc:
        LOGGER.warning("randomized-SMILES control abstained after %s", type(exc).__name__)
        return ""


def scaffold_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None or MurckoScaffold is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=False)
    except Exception:
        return ""


def normalize_record(smiles: str) -> NormalizedMol:
    can = canonicalize(smiles)
    if not can:
        return NormalizedMol(smiles, "", "", "", "", False, "rdkit_parse_failed")
    return NormalizedMol(
        raw=smiles,
        canonical=can,
        parent=parent_smiles(can),
        stereo_stripped=stereo_stripped_smiles(can),
        scaffold=scaffold_smiles(can),
        valid=True,
    )


def ecfp_bits(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = _mol(smiles)
    if mol is None or AllChem is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
