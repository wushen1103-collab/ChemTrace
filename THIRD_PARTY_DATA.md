# Third-party data

The MIT License in this repository applies to ChemTrace source code, not automatically to third-party datasets.

## MoleculeNet task data

Task URLs are declared in `chemtrace/datasets.py`. The loader downloads the public ESOL, BACE, BBBP, HIV, ClinTox, Tox21, and SIDER files when absent. Cite MoleculeNet and the original task datasets when using them.

## ChEMBL36

ChEMBL36 supplies registry self-identity, parent-hierarchy, and Standard-InChI-derived stereo evidence. Obtain the official release separately and place or link the SQLite database at `data/external/chembl_36.db`, or pass `--chembl-db` explicitly. ChEMBL identifiers and derived structures remain subject to ChEMBL terms and attribution requirements.

## Tautobase

The helper script clones the public Tautobase repository into `external/tautobase`. Tautobase's repository does not currently expose a standalone license file; consult its repository and cited source publications before redistributing source records. Bundled ChemTrace tables identify Tautobase as the upstream curated source.

## ZINC15 proxy and MoLFormer sources

These corpora are not vendored. Supply local files with `--zinc15-csv` or `--corpus-root`. The defaults point to paths inside `data/external/` or `external/`, both ignored by Git.

## Fixed ChemTrace inputs

`data/paper_inputs.tgz` contains the controlled corpus snapshot and intervention records used by the paper. It contains no usernames, hostnames, absolute paths, credentials, checkpoints, or training logs.
