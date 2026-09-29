# Final Research and Architecture Report

**Status:** final evidence-bounded design decision. This report supersedes V3 where its descriptions conflict with the current source. It does not claim a newly trained, full-scale model or a reproduced competition score.

## Decision

Use one LightGBM pair scorer with the corrected typed-token retrieval score, typed-token and exact-key candidate union, and deterministic **one-owner-per-target + NULL** assignment. Calibrate the global threshold on a calibration partition and report the result only on a separate locked audit partition. Keep the existing country partition and preprocessing. Do not use Stage 2 in the core. Do not add a France-specific model or threshold, character retrieval, new address parser, external data, embeddings, graph methods, or assignment optimization.

This is the final redesign from the previous multi-stage direction. The main evidence-backed risk is **domain/scale transfer**: France constitutes 259,452 of 1,732,544 test S1 records and is absent from training labels; additionally, no representative full-scale run established candidate/index behavior or memory requirements. The supplied 0.951 current and 0.952 previous scores cannot be independently attributed or reproduced from artifacts in this workspace.

## Research conclusion and honest score range

The small, ratio-matched screening audits produced Stage 1 macro F0.5 between **0.9924 and 0.9947** on 600-S1 locked samples. These are not competition estimates: they use a 6,000-S1 sampled index and sampled target corpus, and have limited coverage of country and rare-error strata. In these audits, adding only the genuine typed-token `tfidf_score` improved macro F0.5 by **0.003383** and **0.001991** on two seeds without changing candidate volume or candidate recall. That supports keeping this single feature, not projecting the sample score to the competition.

The best evidence-bounded estimate for full evaluation is **approximately 0.95–0.96 macro F0.5** (a rough range, not a measured confidence interval), centered near the supplied 0.951–0.952 results. The evidence does **not** support promising 0.99 or 0.97–0.98 on the full task. The sample/full-domain gap, unlabeled France cohort, and absent full-scale reproduced run are the concrete reasons. A new score claim requires labels or an official scored evaluation; changing the split or metric would be invalid.

## Completed forensic audit

### Ground truth and source-ID joins

The full GT/source-ID audit joined all 2,206,821 ground-truth S1 rows against source S1 and all 7,638,365 target mentions against the combined S2/S3 source table:

| Check | Result |
|---|---:|
| Ground-truth S1 rows / unique S1 IDs | 2,206,821 / 2,206,821 |
| GT target mentions / unique target IDs | 7,638,365 / 7,638,365 |
| GT S1 IDs absent from source S1 | 0 |
| GT target IDs absent from source S2/S3 | 0 |
| Target IDs with multiple S1 owners | 0 |
| Repeated target mentions beyond unique target IDs | 0 |
| Duplicate source S1 IDs | 0 |
| Duplicate S2/S3 target IDs across the target table | 0 |

This validates the source/label ID joins and supports one owner per target in the supplied training ground truth. It does not prove model correctness or test-label quality.

### Leakage, OOF, and feature audit

- Validation rows used to fit and score the earlier Stage 2 path; those scores were contaminated and are rejected.
- A later shared-universe audit found that Stage 2-fit entities' in-sample Stage 2 predictions participated in owner competition. Those ratio-matched Stage 2 claims are also rejected.
- Strict three-fold cross-fitted Stage 2 owner experiments were run on two independent seeds. Seed 2 moved from Stage 1 **0.994138** to Stage 2 **0.994617** (+0.000479); seed 3 moved from **0.992561** to **0.991372** (−0.001189). The two-seed mean change is negative, so Stage 2 is excluded.
- Training negative sampling excludes held-out positive target IDs. Calibration and audit S1 groups are separated. Candidate retrieval is unsupervised; sampled candidate-index limitations are retained as a scale caveat.
- Candidate union order had been exposed as if it were retrieval rank. That pseudo-rank and the default retrieval score are neutralized. The real typed-token weighted retrieval score is stored per candidate and exposed as `tfidf_score`; exact-only candidates receive zero for that feature.
- Target competition must use each target's highest-scoring S1 owner and threshold to either that owner or NULL. Ties are deterministic. S1 capacity remains unlimited. The target-side runner-up margin is not in core; the source-side gap-to-best is a different statistic and does not replace it.
- The independent macro F0.5 scorer matched the README formula on the bounded audits. It includes every S1, including singletons, and treats empty prediction/empty truth as score 1. No metric implementation defect was found.
- The full source-ID audit found no join, duplicate-owner, or duplicate-mention anomaly.

### Controlled evidence retained in the decision

| Change | Before → after macro F0.5 | Candidate recall / volume | Precision and recall effect | Runtime / peak memory | Decision |
|---|---|---|---|---|---|
| Add genuine typed-token `tfidf_score`, seed 2 | 0.991286 → 0.994669 | unchanged; audit 2,103/2,103; no edge change | P 0.996627 → 0.997118; R 0.983357 → 0.987161 | 62.4 s / 727.1 MiB | Retain provisionally |
| Same feature, independent seed 3 | 0.990398 → 0.992389 | unchanged; audit 2,088/2,089; no edge change | P 0.996109 → 0.996137; R 0.980373 → 0.987554 | 63.0 s / 616.8 MiB | Retain provisionally |
| Strict OOF Stage 2, seed 2 | 0.994138 → 0.994617 | unchanged | P 0.997114 → 0.997120; R 0.985735 → 0.987637 | 59.6 s / 510.6 MiB | Not enough to retain alone |
| Strict OOF Stage 2, seed 3 | 0.992561 → 0.991372 | unchanged | P 0.998055 → 0.992799; R 0.982767 → 0.989947 | 56.2 s / 646.6 MiB | Reject; mean effect negative |
| Typed + exact key deduplication, cumulative seed 2 | 0.989604 → 0.990522 | unchanged; no material candidate growth | P 0.992768 → 0.993263; R 0.979078 → 0.981455 | 75.0 s / 754.4 MiB | Retain provisionally; bounded sample only |
| Character retrieval 3–5 | 0.999820 → 0.999872 oracle ceiling on an earlier screen | +1 true pair / +81,654 edges | Candidate-pair precision 3.667% → 2.123% | Not comparable to final arms | Reject from core; tiny recall gain, large volume |
| Stage 2 top-five hard negatives | 0.990937 → 0.990571 | unchanged | P 0.991392 → 0.991404; R 0.985735 → 0.987161 | 80.7 s / 686.2 MiB | Reject; macro decreased |

All experimental scores above are sampled screening evidence. They are not directly comparable across harness versions or to the reported 0.951/0.952. The experiment ledger records the split, candidates, score, precision/recall, time, memory, and disposition for each experiment; superseded and invalidated runs remain labeled as such.

## Final architecture

```text
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

### Included

- Existing normalization, transliteration, address-number and legal-form features.
- Exact country routing: source audit found zero missing or unexpected country labels. France follows the same general model path; no France labels exist for supervised calibration.
- Existing typed-token and exact-key retrieval with token/key deduplication and deterministic tie breaks.
- Actual typed-token retrieval score (`tfidf_score`); union-order rank and fake/default retrieval score are not model evidence.
- Stage 1 LightGBM only.
- At most one S1 owner per target ID, with NULL when the winning score is below threshold. S1 may own multiple target IDs.
- Threshold selected on calibration S1s and frozen before locked audit evaluation.
- Candidate-constrained output and official format validation.

### Explicitly excluded

Stage 2, source-side rank masquerading as target margin, cross-source support with unresolved double counting, character retrieval, address-specific retrieval/parser, state/city/postal extraction, embeddings, cross-encoders, graph propagation, BM25/RRF, LLMs, external data, geocoding, and Hungarian assignment. None has adequate final-scale evidence here; some arms directly reduced macro F0.5 or expanded candidate volume substantially.

## Implementation and reproducibility status

The live source implements candidate score retention, the `tfidf_score` feature, one-owner-plus-NULL assignment, separate calibration/audit groups, and excludes held-out positive targets from sampled training negatives. The training path saves `use_stage2=false`; inference rejects incompatible schema/config artifacts. `models/config.json` and saved model pickles are stale (schema 5 incompatibility) and must not be presented as the trained model. A full training/retraining and test inference were not run in this finalization. The output directory was empty when audited, so the official submission validator cannot yet confirm a final prediction package.

Full-scale operational evidence is also absent. The host has 16 GiB RAM, while eager representations over 12.5M training source rows create meaningful memory risk. Any later release run must record source revision, model/config hashes, threshold, candidate count/recall, per-country counts, precision/recall/macro F0.5 on the locked labeled audit, wall time, peak RSS, validator result, and output hashes. Do not use the sample scores in place of the official metric.

## Final conclusion

The GT/source-ID audit passes with zero anomalies. The score regression from 0.952 to 0.951 remains unlocalized because the supplied scores lack reproducible run artifacts and labels. The defensible architecture is the simpler Stage 1 system with corrected retrieval provenance and one-owner/NULL decisions. Stage 2 is rejected after leakage was found and strict OOF results failed to reproduce a gain. Sample audits support the `tfidf_score` feature but do not establish 0.99. The honest expected full-evaluation range is roughly 0.95–0.96 until a representative labeled/official run proves otherwise.

---

## Post-submission outcome (addendum)

*Added after this audit was written and after the competition concluded — the sections above are left unchanged as the decision record at the time they were made.*

A final configuration was submitted and scored by the competition platform. The official final leaderboard result is macro F0.5 ≈ **0.971** (97.1/100), at a final rank of approximately **1500** (public, unofficial reports put total participation at over 10,500 teams). This is an externally-confirmed number from the competition platform, distinct in kind from the 0.951/0.952 figures discussed above — those remain unattributed to any reproducible artifact in this workspace and should still be treated as historical references only, not as evidence for the official result.

The official 0.971 outcome sits above the 0.95–0.96 evidence-bounded estimate this report reached from sampled screening audits. That estimate was framed throughout as a conservative bound given the evidence available at the time (no full-scale run, unlabeled France cohort), not as a ceiling, so a result above it does not contradict the reasoning above — it indicates the actual risk (particularly France transfer) was smaller than the worst case the estimate was bounding against.
