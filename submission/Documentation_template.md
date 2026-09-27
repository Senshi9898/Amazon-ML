# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Grokking
**Team Members:** Divyanshu (team leader), Aaditya Rawat, Shristi Chandra
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We reverse-engineered the data-generating process from the training labels before modelling. Candidate generation retrieves S1 entities for each S2/S3 record with typed-token TF-IDF plus exact structured keys, inside state-sized blocks. A two-stage LightGBM matcher scores each pair: stage 1 compares names, addresses and house numbers; stage 2 adds entity-level evidence (competing S1s, sibling records, the other source). Every record is assigned to at most one S1. Every component was kept or dropped by a pre-registered ablation on held-out states. Held-out macro F0.5 **0.971** with **22 candidates per S1** on the test set (3× fewer than without pruning).

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the full training labels (`src/er/forensics.py` reproduces every number):
- Each S2/S3 record matches at most one S1 (0 exceptions in 7.64M labels); 5.6% of S1 are singletons; 27% of S2/S3 records are distractors.
- **Near-miss decoys:** 35–45% of distractors copy a real S1's name, city and street with the house number shifted. In the US a shift of ≤10 occurs in 63.7% of decoys but 1.4% of true matches — the decisive precision signal.
- **Addresses are corrupted once per source** (siblings agree with each other 92% of the time when they disagree with S1); names are corrupted per record: typos, legal-form changes, reordering, ID tags, `.com` forms, `formerly/DBA` wrappers, 2.7% full rebrands, and transliteration into 9 Indic scripts (token-aligned with S1 in 100% of cases).
- 24% of India and 8% of US positives share no Latin name token with S1; 97.7% of empty-address records are true matches; no usable postal codes; no leakage from IDs or row order.
- France (15% of test) has no labels but follows the same corruption template.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier + constrained assignment.
**Core Innovation:** features that model the generator directly (house-number relation over the full number set, legal-form change, name out-of-vocabulary share) and a second stage that scores each pair with evidence from the record's rival S1s, its same-source siblings and the other source. Validation splits by whole states so decoys stay with the entities they imitate.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:** unaccent, lower-case; unwrap DBA/formerly; strip ID tags; segment domain forms with the S1 vocabulary; Indic→Latin dictionary learned from aligned training pairs (1,498 tokens, 96.4% test coverage); set of all address numbers.
- **Blocking keys used:** typed-token TF-IDF over name tokens, address tokens, numbers, number+street and number+name (df ≤ 100, top-8 S1 per record) ∪ exact keys (sorted name key, number+street, number+name; ≤50 S1 per key, top-8). Runs per country in blocks of ≈150k S1 so IDF and caps match the training scale; records without a parsable state are scored in every block of their country.
- **Rule-based pruning (no model):** a candidate is kept only if its TF-IDF score is ≥ 0.5 × the record's best score, or it has the record's most shared exact keys; then each S1 keeps its 50 best candidates, which removes generic-name "hub" S1s that attract thousands of unrelated records.
- **Candidate pairs generated:** 38.5M on the test set — **22.2 per S1** (median 19, 99th percentile 50); reduction ratio 0.999994 against all same-country pairs. 457 of 1.73M S1 have none.
- **How true matches were not lost:** measured on held-out states — pruning + cap cut candidates 3.4× (46.3 → 13.6 per S1) while candidate recall moved 0.971 → 0.967 and the perfect-scorer ceiling 0.990 → 0.988; end-to-end macro F0.5 0.9727 → 0.9710. Top-k cuts were worse at every budget (top-2: 12 per S1 but ceiling 0.980). Character 5-gram retrieval was tested and rejected (+0.0008).

---

## 4. Matching Model
**Features used (35 in stage 1, +10 in stage 2):**
- Name features: token Jaccard, token-set ratio, edit ratio, token counts, S1 name-key collisions, legal-form relation, out-of-vocabulary share.
- Address features: token and street Jaccard, street token-set ratio, empty-address flag; house numbers: coverage of S1's numbers, missing/extra numbers, |Δ|, truncation.
- Other: retrieval score/rank/exact hits, candidates per record and per S1, gaps to the record's best candidate, source. Stage 2: runner-up score, margin, rank, best sibling score, same/other-source support. No country feature (France is unseen).

**Model type:** LightGBM binary classifier at each stage (127 leaves, 400 rounds); stage 2 trained on out-of-fold stage-1 scores.
**Threshold selection method:** each record → its argmax S1 if p ≥ τ; τ = 0.65 chosen by maximising macro F0.5 on out-of-fold training scores; a 0.1 cross-block margin for records without a state.

---

## 5. Results & Error Analysis
- **F0.5 Score (macro):** final pruned configuration **0.971** on held-out states (256k S1, CI 0.971–0.972; unpruned 0.973); unpruned pipeline at full scale 0.961 on 1.94M unseen S1 (US 0.971, India 0.947). Rules baseline 0.640; ablation gains: all-number address set +0.197, learned scorer +0.106, legal/OOV features +0.009, stage 2 +0.003. Rejected (<+0.002): margins, per-S1 expected-F0.5 rule, calibration, char n-grams, hard-negative weighting.
- **Common false positives:** near-miss decoys with a small house-number shift; address-matching records with dissimilar names.
- **Common false negatives:** empty-address records whose name fits several S1s; records never retrieved (44% of misses); true matches with a genuine house-number typo (indistinguishable from decoys).
- **France:** train-on-US→test-on-India scores 0.912 vs 0.965, so an unseen country is expected to score below US/India; predicted France structure (5.4% singletons, 3.45 matches per S1) matches the training labels.

---

## 6. Conclusion
The gains came from understanding the corruption process, not from model size: comparing the set of house numbers alone was worth +0.197, and entity-level evidence added the precision F0.5 rewards. The pipeline scores the whole test set in ≈25 minutes on a 16 GB laptop with no GPU, no pretrained model and no external data.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/er/`: `data` (ingest), `split` (state subsets), `views` (normalisation), `retrieve`/`rules` (candidates), `features`, `learn` (stage 1), `stage2`, `pipeline` (full-scale blocks, outputs), `metric`, `exp` (experiment registry), `forensics`. `README.md` gives the six commands; `requirements.txt` pins polars, numpy, scipy, scikit-learn, rapidfuzz, LightGBM (all MIT/BSD). Entry points: `python -m er.pipeline train`, `run test`, `finish test`.

### B. Additional Results
The full write-up with figures, equations and the complete experiment log is `Entity-Resolution-Whitepaper.pdf` in this package; `experiments/experiments.csv` holds every run.
