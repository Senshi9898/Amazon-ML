# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task as inverting a data-generating process. Before modelling, we reverse-engineered it from the training labels. Candidate generation retrieves S1 entities for each S2/S3 record: typed-token TF-IDF retrieval combined with exact structured keys, in partitions of whole states. A two-stage LightGBM matcher then scores each pair. Stage 1 compares the pair's names, addresses and numbers. Stage 2 adds entity-level evidence: competing candidates, sibling records that share a corrupted address, and support from the other source. Every S2/S3 record is assigned to at most one S1. Every component was kept or removed by an ablation measured in macro-F0.5 on held-out states.

---

## 2. Methodology

### 2.1 Problem Analysis
Every number below was measured on the training data.
- **Structure.** Each S2/S3 record matches at most one S1 (0 violations across 7.6M labels). An S1 has 0–11 matches (mode 3), and 5.6% of S1 are singletons. Country always agrees between matches. IDs, row order and file order carry no signal.
- **Distractors.** 27% of S2/S3 records match nothing. Of these, 35–45% are *near-misses*: same name, city and street as a real S1, with the house number changed. In the US, a house-number difference of 10 or less occurs in 64% of near-miss decoys but in only 1.4% of true matches.
- **Address corruption is shared per source.** When two records of the same S1 from the same source both disagree with S1's house number, they agree with each other 92–98% of the time. Names, by contrast, are corrupted independently per record.
- **Name noise.** Typos (insertion/deletion 25%, substitution 15%, leet-style 7%), legal-form add/drop, token reordering, noise prefixes, ID tags, `.com`/handle forms, `formerly known as` / `DBA` / `t/a` wrappers, and random rebrands (2.7%). Indian names are transliterated into 9 Indic scripts, token-aligned with the S1 name.
- **Address noise.** Abbreviations, reordered or dropped components, state code vs full name vs native script, leading-zero and truncated house numbers, injected `DOOR NO <random>` prefixes (India), and `<NULL>` placeholders. 3–5% of addresses are empty, yet 97.7% of those records are true matches. There are no usable postal codes.
- **France** (15% of test) has no labels. It follows the same corruption template: case mix, accent injection and component drops at the same rates as the US and India.

### 2.2 Solution Strategy
**Approach type:** multi-retriever blocking → two-stage gradient-boosted matcher → constrained assignment.
**Core innovation:** features that model the generator directly (house-number relation type, changes in legal form, name out-of-vocabulary share), plus entity-level second-stage evidence (siblings that share a corrupted base address, cross-source support). Validation uses held-out *whole states*, so every near-miss decoy stays in the same split as the S1 it imitates.

---

## 3. Candidate Generation (Blocking)
- **Normalisation.** Unaccent and lowercase. Unwrap DBA/formerly/t-a. Strip ID tags. Split domain and handle forms using the S1 vocabulary. Map Indic→Latin with a token dictionary learned from positionally aligned training pairs (1,498 tokens; covers 96.4% of test Indic tokens). Drop `null` placeholders. Extract the set of all numbers in the address.
- **R-sparse.** Record → S1 retrieval with TF-IDF over typed tokens: name tokens, alphabetic address tokens, numbers, number+street pairs and number+name pairs. Postings with df > 100 are dropped, and the top 8 S1 per record are kept.
- **R-exact.** Exact keys: sorted name key, number+street token and number+name token. Keys shared by more than 50 S1 are dropped, and the top 8 S1 per record (by number of shared keys) are kept.
- **Union.** Both retrievers run within country and within state blocks of about 150k S1. Each record is routed by its address state, using a map learned from the training labels (0.04% wrong routes); records without a state are sent to every block of their country.
- **Candidate pairs generated (test):** ≈ 116M; 343 of 1.73M S1 have no candidate.
- **Measured recall (held-out states):** 97.1% of true pairs retrieved. The best possible F0.5 with this candidate set is 0.990. Retrieval experiments: exact keys alone 0.906 recall; TF-IDF alone 0.945; union 0.971.

---

## 4. Matching Model
**Stage-1 features (35):**
- *Name:* token Jaccard, token-set ratio, Levenshtein ratio, token counts, and name-key collisions among S1.
- *Legal form:* same, added, dropped or changed; "Group" added. A record that adds "Group" was never a true match (0 of 837).
- *OOV share:* the share of the name's tokens that no S1 uses; high values indicate a rebrand.
- *Address:* token and street Jaccard, token-set ratio, empty-address flag.
- *House numbers:* coverage of S1's numbers, missing and extra numbers, whether the first number is equal, absolute difference, truncation relation.
- *Retrieval:* score, rank, relative score, exact-key hits.
- *Context:* candidate counts per record and per S1, and the gap to the record's best name/address similarity.
- *Other:* source. There is deliberately no country feature (France is unseen).

**Stage-2 features (+10), computed on stage-1 scores:**
- the record's runner-up score, margin and rank;
- sibling records of the same source with the same address signature, and their best score for this S1;
- support for this S1 from the same source and from the other source (count, sum and maximum).

**Model type:** LightGBM binary classifier at each stage (127 leaves, 400 rounds). Stage 2 only sees pairs with stage-1 p ≥ 0.01, which keeps 8% of pairs and 99.94% of retrieved positives.
**Threshold selection:** each record is assigned to its argmax S1 if p ≥ τ. τ is chosen on out-of-fold training scores by maximising macro-F0.5 directly (τ = 0.65).

---

## 5. Results & Error Analysis
Macro-F0.5 on held-out states (256k S1, never used for training or tuning), one change at a time:

| Step | Macro F0.5 |
|---|---|
| Rules baseline | 0.640 |
| + all-number address comparison | 0.837 |
| + name normalisation, Indic dictionary, top-8 exact keys | 0.858 |
| + TF-IDF retrieval ∪ exact keys, LightGBM | 0.961 |
| + legal-form and OOV features | 0.970 |
| + stage 2 (competition, siblings, cross-source) | **0.973** (95% CI 0.972–0.973) |
| **Full-scale run** (test-identical block pipeline, 1.94M train S1 never used for training) | **0.961** (US 0.971, India 0.947) |

Rejected by ablation (gain < +0.002): top-1/top-2 score margins, per-S1 expected-F0.5 decision, isotonic calibration, looser posting cap (memory), character 5-gram name retrieval for empty-address records (candidate recall on that stratum 78% → 87%, but end-to-end +0.0008 at 7.2 GB), dense embeddings (only about 1% of possible F0.5 is lost to retrieval).

- **Why the full-scale score is lower than the dev score:** in the dev subsets, a record without a state only competed with the S1s of its owner's states. At full scale it competes with same-name S1s in every state. We added a cross-block rule: such a record abstains unless its best block wins by a margin of 0.1 (+0.0025). We also gave S1s with an unparsed state a fallback from a learned address-component→state map, or scored them in every block of their country.
- **Common false positives:** near-miss decoys whose house number moved by a small amount, and address-matching records whose names are dissimilar.
- **Common false negatives:** empty-address records whose name matches several S1s, records that retrieval missed (44% of false negatives), and positives with a genuine typo in the house number (indistinguishable from decoys).
- **Domain shift (France proxy):** trained on the US only and tested on India, the model scores 0.912 (the model trained on both scores 0.965). Trained on India only and tested on the US, it scores 0.958 (vs 0.972). So France, being unseen, is expected to score below the US and India. Its predicted singleton rate (5.4%) and matches per S1 (3.45) are in line with the US (5.8%, 3.40) and India (6.7%, 3.35).

---

## 6. Conclusion
The largest gains came from understanding how the data was generated, not from model size. Comparing all numbers in the address alone added +0.197. Legal-form and rebrand signals, and entity-level evidence, added the precision that F0.5 rewards. Every component was kept or dropped by an ablation on held-out states.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/er/` holds the source: `data`, `split`, `views`, `retrieve`, `rules`, `features`, `learn`, `stage2`, `pipeline`, `exp`, `metric`, and `forensics` (reproduces every data-analysis number in §2.1 in about 12 s).
- `README.md` gives the exact commands.
- `requirements.txt` pins the versions: polars, numpy, scipy, scikit-learn, rapidfuzz, lightgbm. All are MIT/BSD licensed, and no pretrained models are used.

Full-scale inference processes state blocks of about 150k S1 (the same scale the model was trained at), one subprocess per block, in about 4 minutes and at most 7 GB RAM per block. Entry points: `python -m er.pipeline train`, then `run test`, then `finish test`.

### B. Additional Results
The full experiment log, with hypothesis, change, metrics, runtime and memory per experiment, is in `experiments/experiments.csv`.
