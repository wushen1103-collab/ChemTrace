# ChemTrace

ChemTrace audits molecular pretraining corpora with explicit, record-level relation types. It distinguishes canonical identity, standardized-parent identity, stereochemistry-stripped connectivity, and canonical-tautomer identity from broader chemical-neighborhood similarity.

This repository contains the implementation and compact reproducibility package for the accompanying paper. It includes:

- deterministic molecular normalization and relation keys;
- controlled corpus-intervention and model-training code;
- exact fixed inputs for the paper's retrieval experiments;
- scripts for the common-protocol audit, bootstrap analyses, and provenance certificates;
- compact result tables used to check the reported claims.

Neural checkpoints, per-epoch logs, and redundant prediction caches are intentionally excluded. They are large and can be regenerated with the included code. The fixed retrieval inputs are distributed as a 22 MB compressed archive.

## Installation

Python 3.10 was used for the reported experiments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the training dependency separately when rerunning neural experiments:

```bash
python -m pip install -r requirements-train.txt
```

For GPU training, install the PyTorch build appropriate for the local CUDA runtime instead of the default wheel. The recorded environment used PyTorch 2.13.0 with CUDA 13.0.

## Verify the release

```bash
python scripts/check_release.py
python -m pytest -q
```

The first command checks the fixed-input archive, principal result files, numerical identities, and the absence of machine-specific paths in public text files.

## Reproduce the retrieval results

Unpack the fixed paper inputs:

```bash
python scripts/unpack_paper_inputs.py
```

Then rerun Figure 2 and Table 2 from the same corpora, intervention manifests, affected-query table, and seeds:

```bash
python scripts/reproduce_retrieval.py
```

Outputs are written under `reproduced/`. The full common-protocol audit is CPU-intensive; no model training is required for this step.
For a short installation and interface smoke test, run `python scripts/reproduce_retrieval.py --quick`.
Quick-mode outputs use one task and reduced sampling parameters and are not the paper estimates.

The collapsed-standardization reference and route-abstention audit use the same unpacked inputs and released pair tables. The command also evaluates SuperParent on the Table-2 clean-background/decoy grid:

```bash
python scripts/audit_revision_retrieval.py --workers 8
```

## Rerun controlled pretraining

The fixed input archive expands to `artifacts/phase0_tf_confirm_top_len158_extra`, which is the default artifact root used by the paper scripts. A single Transformer condition can be rerun with:

```bash
python scripts/phase0_run_one.py \
  --artifact-root artifacts/phase0_tf_confirm_top_len158_extra \
  --task bbbp \
  --condition clean \
  --seed 13
```

Use `python scripts/phase0_run_one.py --help` and `python scripts/phase0_run_graph_one.py --help` for the complete Transformer/GRU and GIN options. Launchers for seed grids are available under `scripts/`.

After regenerating the prediction files, the endpoint-clustered global summary, canonical-changing affected-set sensitivity, and centered-bootstrap dispersion diagnostic are produced with:

```bash
python scripts/analyze_revision_sensitivities.py --workers 8
```

The compact outputs used by the paper are released under `reports/tables/revision_sensitivities/`; per-sample prediction caches remain excluded because they can be regenerated and are substantially larger.

The MoleculeNet task files are downloaded from the URLs declared in `chemtrace/datasets.py` when they are not already present. External ChEMBL36, Tautobase, ZINC15, and MoLFormer resources are not vendored; see [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Repository map

| Path | Purpose |
|---|---|
| `chemtrace/` | normalization, corpus construction, training, and effect analysis |
| `scripts/` | command-line experiment and audit entry points |
| `data/paper_inputs.tgz` | fixed retrieval inputs, splits, corpora, and intervention manifests |
| `reports/tables/` | compact paper-facing and validation outputs |
| `reports/certificates/` | schema-validated provenance certificate example |
| `schemas/` | JSON Schema for ChemTrace certificates |
| `tests/` | normalization and numerical-contract tests |

The detailed claim-to-artifact map and commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Scope

A ChemTrace certificate establishes a relation-typed match to an inspectable record in the audited corpus snapshot. It does not establish historical lineage, prove that a model used the record, or by itself imply downstream performance inflation. The released relation family contains four atomic routes; arbitrary compositions require explicit path-typed extensions.

## License

The source code is released under the MIT License. Bundled derived tables retain their upstream data-source conditions; see [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
