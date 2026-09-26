# Business Entity Resolution: Research Specification

**Scope.** This document is the contract for the research phase of the Amazon ML Challenge 2026 entry. It defines the components, the data contracts between them, how experiments are run and judged, and the order they run in. It does **not** fix the final architecture. A component makes it into the final system only if an experiment defined here shows that it raises macro-F0.5.

**Evidence base.** The full forensic report is in `~/.claude/plans/pasted-content-id-e0dd-you-are-shimmying-summit.md` (sections 2–5). This spec repeats only the facts that constrain the design (§1).

---

## 0. Ground rules

1. **Keep/kill rule.** A component is kept only if (a) its paired macro-F0.5 gain on DEV-VAL is at least **+0.002**, (b) it does not lower the cross-country proxy (§4.3) by more than 0.002, and (c) it fits the compute budget (§7). Otherwise it is removed, even if it sounds sophisticated.
2. **Metric.** Every decision is made on **macro F0.5 per S1 entity, singletons included**, computed exactly as in the challenge README (`er/metric.py`). Pair AUC, pair accuracy and log-loss are diagnostics only.
3. **Data.** Only the provided files may be used. No geocoding, no registries, no web lookups. Maps learned from the provided test files without labels (for example French abbreviation or region↔department co-occurrence) are allowed and must be documented.
4. **Models.** Only MIT or Apache-2.0 licensed models of at most 8B parameters. Libraries used so far: polars (MIT), numpy/scipy/scikit-learn (BSD), rapidfuzz (MIT), lightgbm (MIT).
5. **Leaderboard.** The public leaderboard is a sanity check (a sixth fold). Never tune thresholds or features on it.

---

## 1. Facts that constrain the design (measured)

| ID | Fact | Design consequence |
|---|---|---|
| F1 | Each S2/S3 record belongs to at most one S1; an S1 can have 0–11 records (mode 3); 5.6% of S1 are singletons | Retrieval runs **record → S1** (top-k per record); assignment enforces ≤1 S1 per record |
| F2 | 27% of S2/S3 records are distractors; 35–45% of those are near-misses: same name, city and street, with the **house number shifted by 1–100** | A house-number relation feature is core. Near-miss false positives are the main KPI |
| F3 | US house relation: delta ≤10 in **1.4%** of positives vs **64%** of near-miss negatives. India: an injected `DOOR NO <rand>` prefix means the first number is unreliable | Compare **multisets of numeric tokens**, not the first number |
| F4 | Address corruption is shared **per source** (siblings that disagree with S1 agree with each other 92–98% of the time); names are corrupted per record | Within-source sibling groups are a candidate entity-level signal |
| F5 | Country always agrees between matches; test adds France (15%) with no labels | Partition by country (open set); run a cross-country proxy |
| F6 | 24% of India and 8% of US positives share **no** Latin name token with S1 (Indic script, rebrands, `.com` concatenations) | An address-only path is needed. Indic dictionary learned from labels covers 96.4% of test Indic tokens |
| F7 | Non-empty addresses of positives always share at least one token with S1 | Address token retrieval is complete for records with an address |
| F8 | 3–5% of S2/S3 addresses are empty; 97.7% of those records are true matches | Name-only path with an ambiguity check. An empty address is not a reason to abstain |
| F9 | No usable ZIP/PIN: 6-digit numbers essentially never appear, 5-digit numbers appear in about 10% of US addresses and are mostly house numbers | No postal-code blocking key |
| F10 | IDs, row order and ground-truth ordering carry no signal | No leakage features |
| F11 | Exact-key union recall: 99.8% on normal positives, 81–84% on address-only, 94% on name-only | Retrieval research concentrates on the two weak classes |
| F12 | S1 name keys collide heavily (205k groups; 16k collide within the same city) | S1↔S1 ambiguity must be resolved by a score margin |

---

## 2. Component catalogue

Each component has a contract (inputs and outputs), the variants to test, the experiment that decides it, and a kill criterion. Components talk to each other only through the tables in §3, so any variant can be swapped without touching the others.

### C1 Ingest *(harness: exists)*
TSV files → Parquet `cache/raw/{split}_s{1,2,3}.parquet` with columns `id, name, addr, country` (read with tab separator and no quoting), plus `cache/raw/train_gt.parquet` (`s1_id, rec_id`). Every S1 row is kept; S1s with no matches have no rows in the ground-truth table.

### C2 Name view (normaliser)
**Input:** raw name. **Output:** `name_norm` (string), `name_toks` (list), `name_alt` (the second name from DBA/formerly/t/a wrappers, or null), `name_concat` (all alphanumerics joined, used for domain and handle matching), and flags (`is_domain`, `is_handle`, `has_idtag`, `script`).
**Variants:**
- V0: lowercase, keep alphanumerics only.
- V1: V0 plus unaccent, noise-prefix and ID-tag stripping, bracket unwrapping, legal-form canonicalisation.
- V2: V1 plus DBA/formerly unwrapping and domain splitting.
- V3: V2 plus Indic→Latin dictionary (C4).

**Decided by:** E-01, E-03.

### C3 Address view (structured parser)
**Input:** raw address and country. **Output:** `nums` (multiset of numeric tokens, keeping `/`, `-` and letter suffixes), `house` (primary number), `unit` (unit, PMB, floor and flat tokens), `street_toks`, `city`, `state` (canonical), `addr_toks`, and `addr_empty`.
**State canonicalisation:** a map learned from ground-truth co-occurrence (the record's component → owner's S1 state) covers codes, full names and native-script names.
**Variants:**
- A0: token set only.
- A1: A0 plus first-number house.
- A2: A1 plus numeric multiset plus unit/PMB.
- A3: A2 plus learned state/city canonicalisation.

**Decided by:** E-02.

### C4 Transliteration
A token-level Indic→Latin dictionary built from positionally aligned training pairs (551k pairs, 1,498 Indic tokens). Unknown tokens pass through unchanged.
**Kill criterion:** India recall on the Indic stratum rises by less than 2 points (E-03).

### C5 Retrievers (a multi-retriever made of independent modules)
Every retriever has the same contract: `retrieve(records, s1_table, k) -> candidates(rec_id, s1_id, retriever, score, rank)`. Retrieval direction is record → S1 (F1), within one country partition.

| ID | Retriever | Target class | Notes |
|---|---|---|---|
| R-exact | exact keys K1/K4/K7/K9 with frequency caps | normal | baseline |
| R-sparse | TF-IDF over name and address tokens plus composite tokens (house+street, name bigram); sparse dot product, top-k | all | main candidate |
| R-char | TF-IDF over character 3-grams of name / `name_concat` / address | typos, domains, rebrands | only for records where R-sparse is weak |
| R-struct | house+street+city key | address-only | cheap, high precision |
| R-dense | multilingual embedding ANN (e5-small, MIT) | Indic not in dictionary, rebrands | **only if** R-sparse ∪ R-char leaves more than 2% recall on the table |

**Decided by:** E-04 to E-06.
**Kill criterion:** the retriever's marginal recall is below 0.3 points at the fixed candidate budget.

### C6 Candidate union
Union of the retriever outputs, deduplicated by `(rec_id, s1_id)`. The per-retriever rank and score are kept as features. Budget: at most **8 S1 candidates per record** and at most about 40 records per S1 on average. The union is exactly what goes into `candidate_pairs.tsv`.

### C7 Pair representation
Output table `pairs(rec_id, s1_id, feat_*)`. Feature families are turned on and off as groups so they can be ablated:

| Group | Contents |
|---|---|
| NAME | token Jaccard, token-set ratio, Jaro-Winkler, character cosine, alt-name max, concat partial ratio |
| ADDR | token Jaccard, street overlap, city and state equality, character cosine |
| HOUSE | relation type (equal / suffix / leading zero / truncation / range / delta≤10 / delta≤100 / far / missing), numeric multiset Jaccard |
| RET | retriever ranks and scores, candidate count per record and per S1, top-1/top-2 score margin |
| META | source, country, address empty, name out-of-vocabulary, S1 name-key collision count |
| DENSE | embedding cosine (only if R-dense survives) |

### C8 Scorer
Candidate families: rules, logistic regression, LightGBM binary, LightGBM LambdaRank grouped by record, and an optional stacked second stage. Every family is trained on DEV-TRAIN and compared on DEV-VAL *after* C11 and C12 (E-07).

### C9 Hard-negative mining
- Round 0: natural negatives from the candidate union, labelled by stratum (near-miss / same-name-other-street / key collision).
- Round 1: add the previous model's top false positives.
- Synthetic near-misses (true positive with the house number shifted by 1–10) only if a stratum has fewer than 50k natural examples.

**Decided by:** E-08 (KPI: near-miss false-positive rate at fixed recall).

### C10 Calibration
Isotonic regression on out-of-fold scores, fit globally, then per country and per source (E-13).

### C11 Entity-level inference
- (a) Record argmax with threshold τ and margin (E-09).
- (b) Within-source sibling groups: group size, group max score, consensus house relation (E-10).
- (c) Cross-source corroboration features, S2↔S3 (E-11).

(b) and (c) are implemented as second-stage features on out-of-fold scores, never as hand-written overrides.

### C12 Macro-F0.5 decision
Two options, decided by E-12:
- A global threshold.
- A per-S1 expected-F0.5 prefix: sort the S1's candidates by calibrated probability and choose the prefix length k that maximises expected F0.5, allowing k = 0.

### C13 Outputs
`matching_results.tsv` and `candidate_pairs.tsv`, checked with `utils/validate_submission.py --check-ids`. *(Final system only; not part of the harness.)*

---

## 3. Data contracts (Parquet, polars)

| Table | Columns |
|---|---|
| `raw/{split}_s{n}` | `id: str, name: str, addr: str, country: str` |
| `raw/train_gt` | `s1_id: str, rec_id: str` |
| `dev/{subset}_meta` | `s1_id: str, rec_id: str (nullable), is_pos: bool`, plus the subset's S1 and record id lists |
| `views/{split}_s{n}_{variant}` | `id` plus the C2/C3 outputs |
| `cands/{exp}_{subset}` | `rec_id, s1_id, retriever: str, score: f32, rank: i16` |
| `pairs/{exp}_{subset}` | `rec_id, s1_id, label: i8 (train only), feat_*: f32` |
| `preds/{exp}_{subset}` | `rec_id, s1_id, p: f32` |
| `sub/{exp}_{subset}` | `s1_id, matched: list[str]` → the metric |

---

## 4. Validation protocol

### 4.1 Development subsets (fast iteration)
The training data is split into two **disjoint groups of states**: DEV-TRAIN and DEV-VAL, each about 10% of S1, with US and India both represented. A subset contains:
- all S1s in its states;
- all records owned by those S1s (through the ground truth);
- all **unmatched** records whose parsed state is in its states;
- unmatched records with no parsable state, sampled at the subset's S1 fraction.

**Why states:** near-miss distractors and same-name S1 collisions are within-city, so partitioning by state keeps *all* of the hard competition inside the subset. Random S1 sampling would remove most competing S1s and give optimistic results.

Held-out states also give a mild geographic-shift test for free.

### 4.2 Full cross-fit (confirmation runs only)
Split S1 into two halves by hash, run on the whole training set, train on half A and score half B and vice versa, then evaluate macro-F0.5 over all 2.2M S1s. This mirrors the test structure exactly and produces the out-of-fold scores for stacking. It is run only on configurations that passed on DEV-VAL.

### 4.3 Proxies
- **Cross-country:** train on US states only, evaluate on India DEV-VAL, and the reverse. This stands in for France.
- **Unseen noise:** hold one corruption signature out of training (DBA/formerly records, or all Indic-script records) and measure the drop on that stratum.
- **Distractor stress:** DEV-VAL with the unmatched records duplicated at a 2× rate.

### 4.4 Trust rule
A change reaches the leaderboard only if it passes §0.1 on DEV-VAL **and** does not regress the full cross-fit. Every result is reported with a bootstrap 95% confidence interval over S1s (1,000 resamples).

---

## 5. Metrics (all in `er/metric.py`)

- **End metric:** macro F0.5, precision and recall per S1 (the README definition). Also reported separately for singletons and non-singletons, and per country.
- **Blocking:** recall of positive pairs overall and by stratum (country × source × {normal, address-only, name-only, Indic, rebrand}); candidates per record and per S1; reduction ratio; **oracle macro-F0.5** (the score if a perfect scorer ran on this candidate set).
- **Scorer diagnostics:** near-miss false-positive rate, and PR curve at pair level.

---

## 6. Experiment protocol

Every run goes through `er.log.log_experiment()`, which appends one row to `experiments/experiments.csv`:

`exp_id, parent_id, timestamp, git_rev, hypothesis, change, subset, n_s1, n_cands, cands_per_s1, blocking_recall, oracle_f05, precision, recall, macro_f05, f05_ci_lo, f05_ci_hi, f05_us, f05_india, f05_singletons, runtime_s, peak_rss_mb, conclusion(KEEP/KILL/INFO), notes`

Each experiment changes **one** component relative to its parent. Conclusions are written the same day.

---

## 7. Compute budget (machine: 12 cores, about 9 GB usable RAM, RTX 4050 with 6 GB, 300 GB disk)

- DEV experiment end to end: **≤ 15 min**, ≤ 6 GB RSS.
- Full test inference: **≤ 2 h**.
- A component that adds more than 25% to the test inference time needs a gain of at least **+0.005** to be kept.

---

## 8. Experiment matrix (run in this order; each row names its parent)

| ID | Parent | Hypothesis | Change | Keep if |
|---|---|---|---|---|
| E-00 | — | Rules give a usable floor and a sanity check for the harness | V1 name, A1 address, R-exact, rules, argmax assignment | INFO (baseline) |
| E-01 | E-00 | Inverting the generator's noise helps | name V0 → V2 | §0.1 |
| E-02 | E-00 | A numeric multiset beats the first number (India) | A1 → A2/A3 | §0.1, near-miss FP ↓ |
| E-03 | best | The Indic dictionary recovers India positives | V2 → V3 | +2 pts Indic recall |
| E-04 | best | TF-IDF retrieval beats exact keys at an equal budget | R-exact → R-sparse | oracle F0.5 ↑ at ≤8 candidates per record |
| E-05 | E-04 | Character retrieval closes the address-only and name-only gaps | + R-char | marginal recall ≥ 0.3 pts |
| E-06 | E-05 | Dense retrieval adds recall | + R-dense | marginal recall ≥ 0.3 pts and budget |
| E-07 | best | A learned scorer beats rules | rules → LR → LightGBM → LambdaRank | §0.1 |
| E-08 | E-07 | Hard-negative round 1 helps | + mined false positives | near-miss FP ↓, §0.1 |
| E-09 | E-07 | A margin reduces S1↔S1 confusions | + margin | §0.1 |
| E-10 | E-09 | Sibling-group consensus helps | + group features (stage 2) | §0.1 |
| E-11 | E-10 | Cross-source corroboration helps | + S2↔S3 features | §0.1 |
| E-12 | best | The per-S1 expected-F0.5 decision beats a global τ | decision rule | §0.1 |
| E-13 | best | Per-cell calibration helps | isotonic per country × source | §0.1 |
| E-14 | best | A country-agnostic model transfers better | drop country features | cross-country proxy ↑ |
| E-15 | best | Seed bagging reduces variance | 3–5 seeds | CI narrows, F0.5 not lower |

**Gates:**
- **G1** (after E-00): the harness agrees with the README worked example (0.714) and the validator passes on a DEV-shaped output.
- **G2** (after E-05): oracle F0.5 is at least 0.97 on DEV-VAL.
- **G3** (after E-12): the configuration is frozen for the full cross-fit and the test run.

---

## 9. What the harness provides now (`er/`)

| Module | Purpose |
|---|---|
| `er/data.py` | Paths, TSV→Parquet ingest, loaders, official-format writer (`write_id_lists`) |
| `er/split.py` | Learned state canonicalisation, DEV-TRAIN / DEV-VAL state subsets, positive-pair strata |
| `er/metric.py` | Macro F0.5 (with self-check), bootstrap CI, blocking recall, oracle F0.5 |
| `er/log.py` | Experiment logging |
| `er/e00_baseline.py` | E-00 only: minimal rules baseline that exercises the harness end to end |

**Not included yet (deliberately):** the production normaliser, the retrievers beyond R-exact, the learned scorer, entity-level inference, test inference and submission packaging.

**Run:** `.venv/bin/python -m er.data` (ingest) → `-m er.split` (subsets) → `-m er.metric` (self-check) → `-m er.e00_baseline devval`.

---

## 10. Status (25 Sep, DEV-VAL, full log in `experiments/experiments.csv`)

| Exp | Change | Macro F0.5 | Verdict |
|---|---|---|---|
| E-00 | rules baseline | 0.6397 | baseline |
| E-01 / E-02 / E-03 | name noise inversion / all-number set / Indic dictionary | 0.6470 / 0.8367 / 0.6519 | KEEP all three |
| E-04a | V3 + A2, exact keys, top-8 per record | 0.8581 | reference |
| E-04, E-05 | TF-IDF retrieval, ∪ exact keys | recall 0.971, oracle 0.990 (G2 passed) | KEEP |
| E-07 | LightGBM pair scorer | 0.9605 | KEEP |
| E-09 / E-12 / E-13 | margin / per-S1 expected-F0.5 / isotonic | +0.0000 / +0.0004 / +0.0001 | KILL |
| E-07b | legal-form relation + name OOV features | 0.9699 | KEEP |
| E-10a → E-11 | stage 2: record competition, siblings, cross-source | 0.9727 | KEEP (as one component) |
| E-14 | cross-country transfer (France proxy) | US→India 0.912, India→US 0.958 | INFO |
| G3-full | frozen E-11 at full scale, state blocks | **0.9612** (1.94M unseen S1) | confirmation |
| E-05c / E-07c | R-char (name 5-grams, empty-address records) | recall 0.976 / F0.5 0.9707 (+0.0008) | KILL |

Modules added: `views.py` (C2–C4), `rules.py`, `retrieve.py` (C5), `features.py` (C6–C7), `learn.py` (C8), `decide.py` (C10–C12), `stage2.py` (C11), `exp.py` (registry + runner).
