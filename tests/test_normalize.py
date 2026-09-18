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
