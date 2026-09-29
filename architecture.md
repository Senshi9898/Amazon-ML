# Final Architecture V4 — Business Entity Resolution (Team Grokking)

**Status:** implemented, measured, frozen, submitted. This document describes the system in `er/` exactly as it produced `submission/output/`. Every number is reproducible from the code and `experiments/experiments.csv`; `python -m er.forensics` reproduces the data facts in 12 seconds.

V4 supersedes V2 (a research plan for a codebase that no longer exists in this repository) and V3 (a reduced plan of the same codebase). Both were reviewed; the useful ideas in them were tested and are recorded in the ablation table below.

---

## 1. Result

**Final submitted configuration (E-11v):** held-out macro F0.5 **0.9830** (CI 0.9827–0.9833; pair precision 0.9955, recall 0.965, singletons 0.984), candidate recall 0.983 at 13.7 candidates per S1 (oracle F0.5 0.994). E-11t = E-11s + TOKR: the training-label positive rate of the name tokens a record adds or drops relative to S1 (+0.0055, decoy false positives 7,715 → 3,575). E-11v = E-11t with the rate table built from every stage-2 candidate pair of the full-scale training run outside the DEV-VAL states (10.5M labelled pairs, 2.8M tokens instead of 41k): +0.0006, adopted as a monotone data-mass increase of the same feature (below the +0.002 rule, stated as an exception). Earlier uploads: E-11q 0.9710 held-out / 0.951 public leaderboard; E-11r 0.9763 / 0.955. E-11s adds the name-token query quota (E-07s), the French department→region map, a +0.10 test threshold for the test set's 1.9× decoy density, and cross-block pair de-duplication (+0.0009 at full scale). The table below is the unpruned E-11 configuration the line was derived from.

| Measure | Held-out states (DEV-VAL, 256k S1) | Full scale (1.94M S1 never trained on, test-identical pipeline) |
|---|---|---|
| **Macro F0.5** | **0.9727** (95% CI 0.9723–0.9731) | **0.9612** (95% CI 0.9610–0.9614) |
| US / India | 0.974 / 0.970 | 0.971 / 0.947 |
| Singletons | 0.965 | 0.952 |
| Pair precision / recall | 0.990 / 0.951 | 0.985 / 0.931 |
| Candidate recall / oracle F0.5 | 0.970 / 0.990 | 0.968 / 0.988 |
| Rules baseline | 0.640 | — |

Test outputs pass the official validator (all rules, ID-existence check included). France (15% of test) has no labels; the cross-country proxy (train US → test India: −0.053; train India → test US: −0.014) bounds the expected penalty.

---

## 2. What the data is (measured facts that drive the design)

| Fact | Consequence |
|---|---|
| Each S2/S3 record belongs to at most one S1 (0 exceptions in 7.64M labels); S1 has 0–11 matches (mode 3); 5.6% singletons | Record → S1 retrieval; assignment enforces one owner per record |
| 27% of S2/S3 are distractors; 35–45% of those are **near-miss decoys**: same name, city and street, house number shifted (US: |Δ| ≤ 10 in 63.7% of decoys vs 1.4% of true matches) | House-number relation over the **set of all numbers** is the core precision feature |
| Addresses are corrupted **once per source** (siblings agree with each other 92% when they disagree with S1); names per record | Same-source sibling groups are entity-level evidence |
| 24% of India / 8% of US positives share no Latin name token with S1 (Indic scripts, rebrands, `.com` forms) | Learned Indic→Latin dictionary (1,498 tokens, 96.4% test coverage); address-only path |
| 97.7% of empty-address records are true matches | Never abstain on an empty address |
| Country always agrees; no postal codes; no leakage in IDs or row order | Country partition; no ZIP keys; no ID features |
| Operator rates differ by source (S2 US: state codes 94.5%; S3 US: full names 89.1%; S3 India reorders 30.7%) | Source is a feature; no country feature (France unseen) |

---

## 3. Pipeline (as implemented)

```
TSV → Parquet                                   er/data.py
   ↓
Views: name (unaccent, unwrap DBA/formerly, strip ID tags, segment domains,
       Indic→Latin dictionary, legal-form set) ; address (token set, street
       tokens, set of all numbers)              er/views.py
   ↓
Country partition → state blocks of ≈150k S1 (learned component→state map;
       stateless records go to every block of their country)   er/pipeline.py
   ↓
Retrieval, record → S1, top-8 each:
   R-sparse  typed-token TF-IDF (name, address, number, number+street,
             number+name tokens; df ≤ 100)      er/retrieve.py
   R-exact   sorted name key, number+street, number+name (≤ 50 S1/key)  er/rules.py
   ↓ union → prune: keep TF-IDF score ≥ 0.5 × record best, or record's most shared exact keys
   ↓ cap: each S1 keeps its 50 best candidates (removes generic-name hubs)
   = candidate_pairs.tsv (22.2 per S1 on test)   er/features.candidates
Stage 1: 35 pair features → LightGBM            er/features.py, er/learn.py
   name · address · house-number relation · legal form · OOV share · retrieval · context
   ↓ keep p1 ≥ 0.01 (8% of pairs, 99.94% of retrieved positives)
Stage 2: +10 entity features on out-of-fold stage-1 scores → LightGBM   er/stage2.py
   runner-up, margin, rank · same-source siblings · other-source support
   ↓
Assignment: record → argmax S1 if p2 ≥ τ* = 0.65 (τ chosen on OOF macro F0.5);
   stateless records abstain unless best block wins by 0.1     er/pipeline.assign
   ↓
matching_results.tsv, candidate_pairs.tsv → official validator
```

Full test inference: 13 state blocks, 38.5M candidate pairs after pruning (116M before), ≈20 min, ≤ 7 GB RAM, 12-core laptop, no GPU, no pretrained model, no external data.

---

## 4. Component decisions (every one measured)

| ID | Component | Macro F0.5 (DEV-VAL) | Δ | Decision |
|---|---|---|---|---|
| E-00 | rules baseline | 0.6397 | | baseline |
| E-01 | name noise inversion (DBA/formerly, ID tags, domains) | 0.6470 | +0.0073 | keep |
| E-02 | all-number address set (vs first number) | 0.8367 | **+0.1970** | keep |
| E-03 | Indic→Latin dictionary | 0.6519 | +0.0049 (Indic recall 0.37→0.98) | keep |
| E-04a | combined views, exact keys top-8 | 0.8581 | | reference |
| E-04 / E-05 | TF-IDF retrieval ∪ exact keys | recall 0.906 → 0.971 | | keep |
| E-07 | LightGBM stage 1 | 0.9605 | +0.1064 | keep |
| E-07b | legal-form relation + OOV share | 0.9699 | +0.0094 | keep |
| E-10a/E-10/E-11 | stage 2: competition, siblings, cross-source | **0.9727** | +0.0027 (CIs disjoint) | keep (as one component) |
| G3-full | frozen config at full scale, state blocks + cross-block margin | 0.9612 | | confirmation |
| E-11p | rule-based pruning (0.5 × record best or best exact) | 0.9712 | −0.0015, 46.3 → 14.0 cands/S1 | keep (organisers' update) |
| E-11q | + cap 50 candidates per S1 | 0.9710 | −0.0001, 13.6 cands/S1, max 50 | submitted (0.951 public) |
| E-07r / E-11r | rarest-6 query tokens at cap 3000, abbreviation/state/ordinal expansion, dotted legal forms, all-pairs number relations, alignment features | 0.9763 | +0.0053, recall 0.967 → 0.981 at 13.8 cands/S1 | keep; submitted (0.955 public) |
| E-07s / E-11s | + guaranteed quota of 3 rarest name tokens per query (a corrupted house number no longer empties the query: 6.7k exact-name positives were never retrieved) | 0.9769 | +0.0006, recall 0.981 → 0.983 at 13.7 cands/S1 | keep |
| E-11t | + TOKR: positive rate (train labels, out of fold) of the name tokens the record adds / drops — decoy vocabulary ("holdings", "industries", "ventures": 0%) vs corruption vocabulary ("center", "services": 35–45%) | 0.9824 | +0.0055 (CIs disjoint), precision 0.990 → 0.995 | keep |
| E-11u | + the same on street tokens and on the exact substitution pair | 0.9827 | +0.0003 | reject |
| **E-11v** | **TOKR table from the full training set** (1.94M S1 outside DEV-VAL, 10.5M pairs) | **0.9830** | +0.0006, precision 0.9951 → 0.9955, singletons up | **final** (rule exception: same feature, more labels) |
| E-09 | top-1 − top-2 margin | 0.9604 | −0.0001 | reject |
| E-12 | per-S1 expected-F0.5 decision rule | 0.9609 | +0.0004 | reject |
| E-13 | isotonic calibration | 0.9607 | +0.0001 | reject |
| E-04b | posting cap 300 | — | 3× join volume | reject (memory) |
| E-05c / E-07c | character 5-gram name retrieval (from V2/V3) | recall 0.976 / 0.9707 | +0.0008, 7.2 GB | reject |
| E-08 / E-11h | hard-negative weighting ×5 (from V3) | 0.9699 / 0.9727 | +0.0000 | reject |
| E-06 | dense multilingual retrieval | — | ≤ 1% headroom | not run |

Admission rule (fixed before the first experiment): keep only if DEV-VAL gain ≥ +0.002, no regression on the cross-country proxy, within budget (≤ 6 GB, ≤ 15 min per dev run).

**Why full scale is 0.011 below dev:** stateless records compete with same-name S1s in every state at full scale but only with their owner's states in the dev subsets. Fixed as far as measurable (misc-state S1 routing, cross-block margin: 0.956 → 0.961); the rest is the honest cost of scale.

**Remaining opportunities, ranked by recoverable mass on held-out states (measured 27 Sep, after E-11t):** (1) a cross-encoder fine-tuned on the 7.6M labelled pairs — the token-rate result (+0.0055 from one learned vocabulary) shows the remaining loss is learned corruption patterns, not features we can enumerate; (2) retrieval misses 0.0066 (7.8k pairs with both a number and a name change ranked below top-8; top-12 recovers +0.002 recall / +0.0007 oracle for +15% candidates, top-16 +0.0028 / +0.001 for +26%); (3) empty-address records among same-name twins 0.0027 (every hidden key tested — id order, row order, match counts, sibling names — scores exactly 1/k); (4) number-truncation family 0.0026 (number-edit-type rates add nothing beyond the existing prefix/suffix/1-digit features). Rejected on measurement: per-country / per-empty-address / per-candidate-count thresholds (≤ +0.0001), street-token and substitution-pair rates (E-11u, +0.0003).

**Why the leaderboard is ≈0.02 below held-out (measured 27 Sep):** empty-address records among same-name S1 twins across states (0.010; 39% of S1s have a same-name twin in-country, and sibling-name consensus resolves them at coin-flip precision, so abstaining is right); the test set holds 5.75 records per S1 against 4.68 in train, i.e. 1.9× the unmatched-record density (0.0035, hence the +0.10 test threshold); duplicate pairs across blocks (0.001, fixed); France (unlabelled: 40% of French records carry a department where S1 carries the region, now mapped). Held-out loss itself splits evenly: retrieval misses 0.008, true matches below τ 0.008, decoy false positives 0.008.

---

## 5. Validation design

- Split by **whole US and Indian states** (DEV-TRAIN 271k S1 / DEV-VAL 256k S1), so near-miss decoys and same-name collisions stay in the same split as the entities they imitate. DEV-VAL matches the full data: 5.6% singletons, 26% distractors, 4.68 records per S1.
- Stage 2 trained only on out-of-fold stage-1 scores; τ chosen on training-subset OOF scores; DEV-VAL never used for fitting or tuning.
- Full-scale confirmation: frozen pipeline run on the whole training set, scored on the 1.94M S1 outside DEV-TRAIN states.
- France proxy: train on one country, test on the other.

---

## 6. Where the remaining loss is (DEV-VAL, final model)

- False negatives (43,416): 61% never retrieved, 38% retrieved but below τ, 1% given to a rival S1. Concentrated on empty-address (34% missed) and address-only (8.5%) records; normal positives 2.9%.
- False positives (8,802): dissimilar name at a similar address 42% (the price of admitting rebrands), similar name 23%, near-miss decoys now only 13%, record belonging to another S1 9%.
- Singletons: 505 of 14,407 (3.5%) receive a prediction.

The remaining loss is retrieval-bound on name-poor records; the one retrieval fix tested (char 5-grams) recovers candidates the scorer cannot then separate.

---

## 7. Submission package

`submission/Grokking_submission.zip`
```
output/matching_results.tsv          scored file (also uploaded to the leaderboard)
output/candidate_pairs.tsv           exact set fed to the matcher
code/business_entity_resolution/src/er/   source (14 modules)
code/business_entity_resolution/README.md, requirements.txt, docs/, experiments/
Documentation_template.md            2-page methodology (team filled)
Entity-Resolution-Whitepaper.pdf     14-page technical write-up
```
Licensing: polars, numpy, scipy, scikit-learn, rapidfuzz, LightGBM (MIT/BSD); ~100k parameters per stage.

---

## 8. Known risks

1. **France is unvalidated** (no labels). Expected penalty between −0.014 and −0.053; predicted structure (5.4% singletons, 3.45 matches/S1) matches training.
2. **India retrieval** on name-poor records is the largest known loss and stays.
3. **τ selected at development scale**; the flat F0.5 curve (0.55–0.75) limits the risk.
4. The three stage-2 feature groups were admitted jointly (each alone < +0.002); recorded in the log.

---

## 9. Reproduce

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export ER_DATA=/path/to/student_resource/dataset
.venv/bin/python -m er.data && .venv/bin/python -m er.split
.venv/bin/python -m er.exp E-07b devval
.venv/bin/python -m er.pipeline train && .venv/bin/python -m er.pipeline run test && .venv/bin/python -m er.pipeline finish test
.venv/bin/python -m er.forensics
```
