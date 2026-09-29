# Architecture — Business Entity Resolution (Team Grokking)

**Status:** final evidence-bounded design decision. Supersedes the earlier multi-stage direction. The competition has since concluded and produced a confirmed official score (Section 1); the internal reproducibility items in Section 8 that predate the submission are kept as a historical record and updated only where the submission itself resolves them.

---

## 1. Result

**Confirmed final outcome.** The submitted configuration was scored by the official competition platform: macro F0.5 ≈ **0.971** (97.1/100) on the final leaderboard, at a final rank of approximately **1500** (public, unofficial reports put total participation at over 10,500 teams). This is the authoritative external result for this architecture.

Internally, at the time this architecture was finalised (before the submission run), no full-scale score had been reproduced from artifacts in the workspace. What existed at that point:

- **Sampled screening audits** (600-S1 locked samples, drawn from a 6,000-S1 sampled index and sampled target corpus): Stage 1 macro F0.5 between **0.9924 and 0.9947**. These use a small, sampled index and target corpus with limited coverage of country mix and rare-error strata, and are **not** competition estimates — they should not be projected directly onto the full test set.
- **Best evidence-bounded estimate for the full evaluation:** approximately **0.95–0.96** macro F0.5 (a rough range, not a measured confidence interval), centred near the previously reported 0.951–0.952. The evidence available at the time did **not** support a higher range (0.97–0.99) on the full task.
- **Previously reported scores (0.951 current, 0.952 previous):** unattributed — no reproducible run artifact in the workspace tied either number to a specific model, config, or candidate set. Still treat as historical reference only.

The confirmed official result of 0.971 sits above this pre-submission estimate, consistent with the estimate having been a conservative bound (given the unlabeled France cohort and absent full-scale run) rather than a ceiling.

---

## 2. What is established about the data

No forensic reconstruction of the data-generating process (corruption operators, decoy modelling, transliteration statistics, etc.) has been carried out or is claimed for this project. What is established comes from the ground-truth/source-identifier audit and the existing pipeline behaviour:

| Fact | Source | Consequence |
|---|---|---|
| Every ground-truth target belongs to at most one S1 owner (0 violations across 7,638,365 target mentions) | GT/source-ID audit | Licenses the one-owner-per-target-plus-NULL assignment rule |
| 2,206,821 ground-truth S1 rows / target mentions all resolve against the source tables, with no duplicates | GT/source-ID audit | Source and label joins are trustworthy |
| Test set: 1,732,544 S1 entities; France = 259,452 of them (~15%), absent from training labels | Dataset counts | No France-specific calibration is possible; country kept as an open label set |
| Existing country partition found zero missing or unexpected country labels | Audit | Country partition and preprocessing kept as-is, unchanged |

---

## 3. Pipeline (as decided)

```
Raw source rows and IDs retained
        ↓
Existing deterministic normalization, transliteration, legal-form and number views
        ↓
Exact country partition (US / India / France; no country-specific model)
        ↓
Typed-token retrieval + exact-key retrieval
        ↓
Deduplicated candidate union with actual typed-token TF-IDF score per candidate
        ↓
Existing Stage 1 pairwise feature set + LightGBM
        ↓
For each S2/S3 target: best-scoring S1 owner; accept above global threshold, else NULL
        ↓
Per-S1 predictions, including singleton S1s
        ↓
Candidate-constrained output generation and official submission validator
```

**Included:**
- Existing normalization, transliteration, address-number and legal-form features.
- Exact country routing (unchanged; audit found zero anomalies). France follows the same general model path with no France-specific labels for calibration.
- Existing typed-token and exact-key retrieval, deduplicated with deterministic tie-breaks.
- The genuine typed-token retrieval score (`tfidf_score`); an earlier candidate-union order that had masqueraded as a retrieval rank is removed.
- **Stage 1 LightGBM only.**
- At most one S1 owner per target ID, NULL when the winning score is below threshold. An S1 may own multiple target IDs.
- Threshold selected on a calibration partition and frozen before locked audit evaluation.
- Candidate-constrained output and official format validation.

**Explicitly excluded:** a second (entity-level) rescoring stage, source-side rank masquerading as target margin, cross-source support with unresolved double counting, character retrieval, an address-specific parser, state/city/postal extraction, embeddings, cross-encoders, graph propagation, BM25/RRF, LLMs, external data, geocoding, and assignment optimization (e.g. Hungarian assignment). None has adequate final-scale evidence; some arms directly reduced macro F0.5 or expanded candidate volume substantially.

---

## 4. Component decisions (measured)

| Change | Before → after macro F0.5 | Candidate recall / volume | Precision / recall effect | Runtime / peak memory | Decision |
|---|---|---|---|---|---|
| Add genuine typed-token `tfidf_score`, seed 2 | 0.991286 → 0.994669 | unchanged; audit 2,103/2,103; no edge change | P 0.996627 → 0.997118; R 0.983357 → 0.987161 | 62.4 s / 727.1 MiB | **Retain** (provisional) |
| Same feature, independent seed 3 | 0.990398 → 0.992389 | unchanged; audit 2,088/2,089; no edge change | P 0.996109 → 0.996137; R 0.980373 → 0.987554 | 63.0 s / 616.8 MiB | **Retain** (provisional) |
| Typed + exact-key deduplication, seed 2 | 0.989604 → 0.990522 | unchanged; no material candidate growth | P 0.992768 → 0.993263; R 0.979078 → 0.981455 | 75.0 s / 754.4 MiB | **Retain** (provisional; bounded sample only) |
| Strict OOF second stage, seed 2 | 0.994138 → 0.994617 | unchanged | P 0.997114 → 0.997120; R 0.985735 → 0.987637 | 59.6 s / 510.6 MiB | Not enough to retain alone |
| Strict OOF second stage, seed 3 | 0.992561 → 0.991372 | unchanged | P 0.998055 → 0.992799; R 0.982767 → 0.989947 | 56.2 s / 646.6 MiB | **Reject** — mean effect negative |
| Character retrieval 3–5 | 0.999820 → 0.999872 (oracle ceiling, earlier screen) | +1 true pair / +81,654 edges | Candidate-pair precision 3.667% → 2.123% | not comparable to final arms | **Reject** — tiny gain, large volume |
| Second-stage top-five hard negatives | 0.990937 → 0.990571 | unchanged | P 0.991392 → 0.991404; R 0.985735 → 0.987161 | 80.7 s / 686.2 MiB | **Reject** — macro decreased |

All rows are sampled screening evidence, not directly comparable across harness versions or to the reported 0.951/0.952 (or, since resolved, to the confirmed 0.971 official result). The second-stage decision is a **mean across two independent seeds**: one gained (+0.000479), one lost (−0.001189); the negative mean is why it is excluded rather than shipped.

---

## 5. Validation design

- Calibration and locked audit partitions are separated at the S1 level; no S1's records appear in both.
- Training negatives exclude held-out positive target IDs.
- Candidate retrieval is unsupervised; sampled candidate-index limitations are retained as an explicit scale caveat (the sampled audits use a 6,000-S1 index and are not a stand-in for full-scale retrieval behaviour).
- The independent macro F0.5 scorer was checked against the published formula on the bounded audits, including correct singleton handling (empty prediction vs. empty truth scores 1). No metric implementation defect was found.

No state-block or country-level validation split, and no cross-country transfer proxy, has been run or is claimed for this project.

---

## 6. Remaining loss / error anatomy

Not established. No error-anatomy breakdown (false-negative/false-positive decomposition, per-stratum miss rates, etc.) has been produced for this architecture. This remains an open item even after the final leaderboard score, since a per-entity or per-country breakdown of the official result requires labels the team does not have access to.

---

## 7. Submission package status

A submission was made and scored (Section 1), which implies `output/matching_results.tsv` and `output/candidate_pairs.tsv` were produced and passed the leaderboard's own format acceptance. This document does not, however, carry a record of the specific run that produced the submitted files — no source revision, model/config hash, wall time, or peak-RSS log for that run is on file here. Anyone needing to reproduce or audit the exact submitted output should treat it as unverified against this document until that run's artifacts are located or regenerated.

---

## 8. Known risks / open items

1. **France is unvalidated.** No France labels exist for calibration or for measuring transfer; the risk was real and unquantified at design time. The confirmed 0.971 final score suggests the realised penalty was smaller than the conservative estimate assumed, but no labeled measurement of the actual France-specific performance exists.
2. **Full-scale operational behaviour at design time was unverified**, and no run log for the actual submission run is on file (see Section 7). The host had 16 GiB RAM; eager representations over ~12.5M training source rows were a meaningful memory risk during development — whether it was reproduced or worked around for the submission run is not documented here.
3. **Stale artifacts (as of the design audit).** `models/config.json` and saved model pickles were stale at that point (schema 5 incompatibility) and were not to be presented as the trained model. Whether the model actually used for the scored submission matches the schema described in this document is not confirmed here.
4. **Score provenance.** The 0.951/0.952 scores still lack reproducible run artifacts and labels tying them to this architecture; the regression between them remains unlocalized. This is unrelated to the externally-confirmed 0.971 official result, which does not require internal artifact provenance to be trusted.
5. **No reproducibility log exists for the actual submitted run.** The release-run checklist below was the target at design time; confirm it was followed for the specific run that produced the submitted files, or reconstruct it, before treating this document as a full account of what was submitted: source revision, model/config hashes, threshold, candidate count/recall, per-country counts, precision/recall/macro F0.5 on the locked labeled audit, wall time, peak RSS, validator result, and output hashes.

---

## 9. Reproduce

Run from the `student_resource/` directory:
```
python3 run_final.py --mode full
```
or as separate steps:
```
python3 run_final.py --mode train
python3 run_final.py --mode test
```
This stores the Stage 1 model/config under `models/`, records a data/settings signature under `artifacts/`, and writes submission files to `output/`, then invokes the official validator and a disk-backed candidate-ID/one-owner check.

Validate independently at any time:
```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Before treating any local output as a match for the scored submission: confirm the model artifacts under `models/` match the current schema (see Section 8, item 3), and confirm `output/` is non-empty and validator-clean.
