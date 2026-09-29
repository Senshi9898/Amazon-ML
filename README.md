# Business Entity Resolution — Amazon ML Challenge 2026

**Team Grokking:** Divyanshu (team leader), Aaditya Rawat, Shristi Chandra

For each clean reference business (S1), find every record in two noisy sources (S2, S3) that refers to the same business, across the US, India and France. Metric: macro F0.5 per S1 entity, singletons included.

## Status

This is the frozen, evidence-bounded architecture described in `FINAL_ARCHITECTURE_V4.md`. It supersedes earlier, more elaborate drafts of this pipeline. Please read the two caveats below before trusting any number in this repo.

- **No full-scale run has been reproduced from current artifacts.** `models/config.json` and the saved model pickles predate the current feature/assignment schema and must not be treated as the trained model behind any reported score. A full train → test-inference → validate pass needs to be re-run before a number here can be called final.
- **The previously reported 0.951 (current) / 0.952 (previous) scores are unattributed.** No reproducible run artifact in this workspace ties either number to a specific model, config or candidate set. Treat them as historical references, not confirmed results.

| | Macro F0.5 | Basis |
|---|---|---|
| Locked audit sample, with corrected `tfidf_score` feature (seed A) | 0.9947 | ~600-S1 sampled audit — screening only |
| Locked audit sample, with corrected `tfidf_score` feature (seed B) | 0.9924 | ~600-S1 sampled audit — screening only |
| Evidence-bounded estimate for full evaluation | **≈0.95–0.96** | Extrapolated; not a measured full-scale score |
| Rules baseline | low; no country-specific or ML matching | For orientation only |

The sampled-audit numbers use a 6,000-S1 sampled index and sampled target corpus, so they are **not** competition estimates: they have limited coverage of country mix (the France cohort cannot be sampled at all, since it has no labels) and of rare-error strata. Do not project them directly to the full test set.

## How it works

1. **Normalisation.** Deterministic, reversible views: transliteration, legal-form canonicalisation, and an address-number view. No forensic characterisation of the data-generating process has been carried out or is claimed here.
2. **Country partition.** Existing partition into US / India / France is kept as-is. `country` is treated as an open label set — no France-specific model, threshold, or normalisation branch, since there are no France labels to validate one against.
3. **Candidate generation.** Typed-token TF-IDF retrieval unioned with exact structured-key retrieval, deduplicated with deterministic tie-breaks. Each candidate carries its real typed-token retrieval score (`tfidf_score`); exact-only candidates get zero. An earlier version had exposed candidate-union order as if it were a retrieval rank — that pseudo-rank has been removed and replaced with the genuine score, which is the one change validated to help on two independent seeds (+0.0034, +0.0020 on the sampled audits).
4. **Matching model — Stage 1 only.** A single LightGBM classifier scores each (S1, candidate) pair over the existing pairwise feature set (name similarity, address/number agreement, legal-form relation, retrieval score). A second, entity-level rescoring stage was built and tested and is **excluded from the core**: after moving to strict three-fold cross-fitted training to remove an earlier leakage issue, one seed gained +0.000479 and the other lost −0.001189 — a negative mean effect across seeds — so Stage 2 was rejected rather than shipped.
5. **Assignment.** Each S2/S3 record → its best-scoring S1 owner if the score is at or above a single global threshold τ, else NULL. At most one owner per record; an S1 may own any number of records. τ is selected on a calibration partition and frozen before scoring on a separate, locked audit partition.
6. **Output and validation.** Candidate-constrained output (every match is a member of the submitted candidate set), checked with the official submission validator.

**Also excluded from the core**, tested or considered and left out: character n-gram retrieval (marginal oracle gain for a large increase in candidate edges), Stage 2 hard-negative reweighting (reduced macro F0.5), a France-specific model/threshold, an additional address parser, embeddings/cross-encoders, graph propagation, external data, and geocoding (the last two are also prohibited by the competition rules).

Every retained or rejected component is recorded, with its measured effect, in `FINAL_ARCHITECTURE_V4.md`.

## Setup
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export ER_DATA=/path/to/student_resource/dataset   # folder containing train/ and test/
```

## Run (end to end)

Run from the `student_resource/` directory:
```
python3 run_final.py --mode full
```
or as separate steps:
```
python3 run_final.py --mode train
python3 run_final.py --mode test
```
This stores the Stage 1 model/config under `models/`, records a data/settings signature under `artifacts/`, and writes submission files to `output/`. It then invokes both the official validator and a disk-backed candidate-ID/one-owner check.

Validate independently at any time:
```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Before relying on any output here:** confirm `models/config.json` and the model artifacts under `models/` match the current schema (the training path records `use_stage2=false`; inference rejects incompatible schema/config artifacts), and confirm `output/` is non-empty and validator-clean. Full-scale runtime and memory behaviour have not been verified on the development host — the implementation materialises comparatively large in-memory representations over the ~12.5M training source rows, and the host has 16 GiB RAM, so this is a real risk to check on the target machine before a final submission run.

## Layout
```
FINAL_ARCHITECTURE_V4.md       the frozen architecture, its decision table, and known open risks
models/                         Stage 1 model + config (verify schema/freshness before use — see above)
artifacts/                      data/settings signature for the current run
output/                         matching_results.tsv and candidate_pairs.tsv (generated, not tracked until run)
```

## Requirements
Only the provided challenge data is used — no geocoding, registries, embeddings, or external/web lookups. No newly trained large model or reproduced competition score is claimed until a full run is confirmed against the current schema.
