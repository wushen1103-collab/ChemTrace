import chemtrace.normalize as normalize
from chemtrace.normalize import canonicalize, parent_smiles, stereo_stripped_smiles, tautomer_smiles


def test_raw_serializations_share_canonical_identity() -> None:
    assert canonicalize("OCC") == canonicalize("CCO")


def test_largest_fragment_parent_removes_counterion() -> None:
    assert canonicalize("CCO.Cl") != canonicalize("CCO")
    assert parent_smiles("CCO.Cl") == parent_smiles("CCO")


def test_enantiomers_share_stereo_stripped_key() -> None:
    left = "F[C@H](Cl)Br"
    right = "F[C@@H](Cl)Br"
    assert canonicalize(left) != canonicalize(right)
    assert stereo_stripped_smiles(left) == stereo_stripped_smiles(right)


def test_tautomer_pair_shares_canonical_tautomer() -> None:
    keto = "O=C1NC=CC=C1"
    enol = "OC1=NC=CC=C1"
    assert canonicalize(keto) != canonicalize(enol)
    assert tautomer_smiles(keto) == tautomer_smiles(enol)


def test_invalid_smiles_returns_empty_key() -> None:
    assert canonicalize("not-a-smiles") == ""


def test_route_unavailability_abstains_without_canonical_fallback(monkeypatch) -> None:
    monkeypatch.setattr(normalize, "rdMolStandardize", None)
    assert normalize.parent_smiles("CCO") == ""
    assert normalize.tautomer_smiles("CCO") == ""


def test_route_exception_abstains_without_cross_relation_substitution(monkeypatch) -> None:
    def fail(_mol) -> None:
        raise RuntimeError("synthetic route failure")

    monkeypatch.setattr(normalize.Chem, "RemoveStereochemistry", fail)
    assert normalize.stereo_stripped_smiles("F[C@H](Cl)Br") == ""
