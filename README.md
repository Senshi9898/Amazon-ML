# Business Entity Resolution — Amazon ML Challenge 2026

**Team Grokking:** Divyanshu (team leader), Aaditya Rawat, Shristi Chandra

For each clean reference business (S1), find every record in two noisy sources (S2, S3) that refers to the same business, across the US, India and France. Metric: macro F0.5 per S1 entity, singletons included.

| | Macro F0.5 |
|---|---|
| Held-out states (256k S1), final configuration E-11v | **0.983** (CI 0.9827–0.9833) |
| Held-out states, E-11t (token rates from DEV-TRAIN only) | 0.982 |
| Held-out states, E-11s (without token-rate features) | 0.977 |
| Held-out states, previous submission (E-11q) | 0.971 |
| Full scale, 1.94M S1 never used for training (E-11, unpruned) | 0.961 |
| Public leaderboard (E-11r, previous upload) | 0.955 |
| Rules baseline | 0.640 |

**Candidate set (test):** 38.5M pairs, **22.2 per S1** (median 19, p99 50) — 3× fewer than unpruned, reduction ratio 0.999994.

Test outputs pass the official validator. Full test inference: ≈25 min, ≤ 7 GB RAM, 12-core laptop, no GPU, no pretrained model, no external data.

## How it works

1. **Forensics first.** The data is synthetic; we measured the corruption process (near-miss decoys with shifted house numbers, per-source address corruption, Indic transliteration, rebrands) before modelling. `python -m er.forensics` reproduces every fact in 12 s.
2. **Views** undo the reversible corruptions: DBA/formerly unwrapping, ID tags, domain segmentation, a learned Indic→Latin dictionary, the set of all address numbers.
3. **Retrieval** (record → S1, top-8): typed-token TF-IDF ∪ exact structured keys, inside state blocks of ≈150k S1 so features match the training scale; then rule-based pruning (keep candidates within 0.5 of the record's best score or with its most shared exact keys) and a cap of 50 candidates per S1. Query = each record's 6 rarest tokens plus its 3 rarest name tokens (without the name quota a corrupted house number empties the query). 98.3% of true pairs retrieved at 13.7 candidates per S1; a perfect scorer on these candidates would reach 0.994.
4. **Stage 1** LightGBM over 35 pair features (house-number relation is the decisive one).
5. **Stage 2** LightGBM over entity-level evidence computed from out-of-fold stage-1 scores: rival S1s, same-source siblings, the other source, and the training-label positive rate of the name tokens the record adds or drops relative to S1 (the generator swaps in decoy words from a fixed vocabulary — "holdings", "industries", "ventures" are 0% positive over 40k+ pairs each; corruption words like "center", "services" run 35–45%). That one feature group moved held-out F0.5 0.977 → 0.982 and pair precision 0.990 → 0.995; building the rate table from all 1.94M non-held-out training S1s (`python -m er.tokrates`, 10.5M labelled pairs, 2.8M tokens) adds a further +0.0006 (0.983).
6. **Assignment:** each record → its best S1 if p ≥ τ (0.70 on out-of-fold macro F0.5, +0.10 on the test set whose unmatched-record density is 1.9× the training set's); at most one owner per record; records without a state abstain unless one block wins by 0.1.

Every component was admitted or rejected by a pre-registered ablation (≥ +0.002 on held-out states). See `FINAL_ARCHITECTURE_V4.md` for the full decision table and `Entity-Resolution-Whitepaper.pdf` for the write-up with figures, equations and error anatomy.

## Setup
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export ER_DATA=/path/to/student_resource/dataset   # folder containing train/ and test/
```

## Run (end to end)
```
.venv/bin/python -m er.data                 # TSV -> Parquet (cache/raw)
.venv/bin/python -m er.split                # DEV-TRAIN / DEV-VAL state subsets (cache/dev)
.venv/bin/python -m er.exp E-07b devval     # DEV pair features + stage-1 out-of-fold scores
.venv/bin/python -m er.pipeline train       # stage-1 / stage-2 LightGBM + threshold (cache/models)
.venv/bin/python -m er.pipeline run test    # scores the test set in state blocks
.venv/bin/python -m er.pipeline finish test # writes submission/output/*.tsv
.venv/bin/python -m er.forensics            # reproduces the data facts
```
Validate: `python3 utils/validate_submission.py -m submission/output/matching_results.tsv -c submission/output/candidate_pairs.tsv -t $ER_DATA/test`

Any experiment in the registry: `.venv/bin/python -m er.exp E-11 devval` (ids in `er/exp.py`; results append to `experiments/experiments.csv`).

## Layout
```
er/                 source: data, split, views, retrieve, rules, features, learn, stage2, decide,
                    exp (experiment registry), pipeline (full-scale blocks + outputs), metric, forensics
docs/RESEARCH_SPEC.md          research contract: components, admission rule, validation protocol
experiments/experiments.csv    every run: hypothesis, metrics, CI, runtime, memory, verdict
paper/                         whitepaper source (LaTeX), figure scripts, figures
FINAL_ARCHITECTURE_V4.md       the frozen architecture and its decision table
Entity-Resolution-Whitepaper.pdf
submission/                    2-page methodology document; outputs and zip are generated (not tracked)
```

## Requirements
Python 3.12; polars, numpy, scipy, scikit-learn, rapidfuzz, LightGBM (pinned in `requirements.txt`; all MIT/BSD). Only the provided challenge data is used — no geocoding, registries or web lookups.
