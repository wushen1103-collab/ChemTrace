# Reproducibility guide

## Released evidence levels

The release separates three evidence levels:

1. **Relation validity:** independent ChEMBL36 and Tautobase relations test the operational keys.
2. **Retrieval behavior:** fixed controlled corpora compare typed relations with exact, scaffold, fingerprint, and adapted audit routes.
3. **Downstream effect:** paired per-sample losses estimate label-adjusted difference-in-differences effects separately from retrieval.

## Paper claim map

| Paper component | Primary released input/output | Regeneration entry point |
|---|---|---|
| Table 1, external relation validation | `reports/tables/phase0_independent_external_gold_v2/` | `scripts/rebuild_independent_relation_gold.py` |
| Figure 2, changed-only retrieval | `data/paper_inputs.tgz`; `reports/tables/phase0_retrieval_baseline_confirm_top_sample5k/` | `scripts/audit_retrieval_baselines.py`; `scripts/bootstrap_retrieval_sota.py` |
| Figure 3, typed-negative stress test | `reports/tables/phase0_reviewer_supplements/hard_negative_pair_bank.csv` and associated score/metric tables | `scripts/run_reviewer_supplement_experiments.py` |
| Table 2, 13-route common protocol | fixed paper inputs; `reports/tables/phase0_modern_sota_baselines/` | `scripts/audit_modern_sota_baselines.py` |
| Figure 4, downstream effects | `reports/tables/phase0_effect_models_completed/`; `reports/tables/revision_sensitivities/primary_interval_bootstrap.csv` | `scripts/fit_phase0_effect_models.py`; `scripts/bootstrap_phase0_effects.py` |
| Global and changed-only downstream sensitivity | `reports/tables/revision_sensitivities/` | `scripts/analyze_revision_sensitivities.py` after prediction regeneration |
| Appendix robustness | low-resource, GRU, and GIN effect summaries under `reports/tables/` | model launchers plus the effect scripts above |
| SuperParent common-protocol and route-abstention checks | `reports/tables/revision_retrieval/` | `scripts/audit_revision_retrieval.py` |
| Scaling and proxy audit | selected tables under `reports/tables/phase0_reviewer_supplements/` | `scripts/run_reviewer_supplement_experiments.py` |
| Provenance certificate | `schemas/chemtrace-certificate.schema.json`; `reports/certificates/` | `scripts/generate_phase0_certificate.py` |

The released certificate excerpt preserves the corpus, manifest, split, and result digests but replaces the original internal repository commit with forty zeroes. New certificates generated from a public checkout record that checkout's commit.

`RESULT_CHECKSUMS.txt` records the SHA-256 digest of every released derived table. Run `python scripts/check_release.py` to verify that manifest together with the pinned input files, numerical contracts, certificate schema, and privacy checks.

## Fixed input archive

`data/paper_inputs.tgz` contains only fixed analysis inputs:

- task splits and affected-query annotations;
- the 50,000-record clean corpus;
- exact, randomized-SMILES, parent, stereo, tautomer, and decoy corpora at the evaluated doses;
- record-level intervention manifests;
- the preparation metadata.

It excludes checkpoints, predictions, logs, caches, and machine-specific paths.

Archive SHA-256:

```text
868d13368c936973bd5925fed7a5fbbca90c6d54288d14d8207c03237d985ebe
```

## Retrieval regeneration

```bash
python scripts/unpack_paper_inputs.py
python scripts/reproduce_retrieval.py
```

The second command runs:

1. exact/scaffold/ECFP retrieval comparison;
2. the task-cluster bootstrap used for Figure 2;
3. all 13 common-protocol audit routes used for Table 2.

The displayed Table 2 means obey the paper's aggregation contract: changed-only recall and the adjusted score average the 78 cells with defined changed queries, while background and decoy rates average all 81 semantic cells. The adjusted score is computed within each cell before aggregation.

## Downstream regeneration

Full downstream retraining is intentionally not packaged as checkpoints. It is regenerated from the fixed corpus/split inputs using the model entry points and then summarized with:

```bash
python scripts/fit_phase0_effect_models.py --help
python scripts/bootstrap_phase0_effects.py --help
```

The released per-seed and summary tables allow direct verification of the reported 51 primary cells without downloading model weights.

## External relation gold

Rebuilding the independent relation gold requires a local ChEMBL36 SQLite database and a Tautobase checkout:

```bash
bash scripts/setup_reviewer_external_sources.sh
python scripts/rebuild_independent_relation_gold.py \
  --chembl-db data/external/chembl_36.db \
  --tautobase-dir external/tautobase
```

Run `--help` first if source locations differ. Source data are not silently substituted: ChEMBL release, Tautobase records, RDKit version, and selection boundaries determine the result.

## Recorded software environment

The principal environment used Python 3.10.12, RDKit 2026.03.5, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, PyArrow 25.0.1, and PyTorch 2.13.0+cu130. Exact analysis dependencies are pinned in `requirements.txt`; training uses `requirements-train.txt`.
