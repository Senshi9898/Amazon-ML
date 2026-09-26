# Review of FINAL_ARCHITECTURE_V2 — pros and gaps

**Reviewed:** `FINAL_ARCHITECTURE_V2.md` and the Whimsical board "Final Architecture V2".
**Checked against:** the official README (`ref/.../student_resource/README.md`), the Unstop page (`ref/contest.md`), and the implemented pipeline in `er/` (measured 0.961 macro F0.5 at full scale, validator PASS).
**Date:** 27 Sep 2026, ~23 h before the deadline (27 Sep, 23:59 IST).

## Verdict

V2 is a sound **research design** and a poor **plan for this deadline**. Its method (out-of-fold scoring, entity-disjoint validation, candidate audit before matcher tuning, one ablation at a time) is right. But it proposes rebuilding, from a 0.836-scoring codebase with known leakage, a pipeline that already exists in this repository, is measured at 0.961, and passes the official validator. The gaps that would actually cost leaderboard points (scale consistency, the submission package) are not in V2 at all.

## Pros

| # | What V2 gets right | Why it matters |
|---|---|---|
| P1 | Diagnoses the leakage in the `src/pipeline.py` codebase: Stage 2 fitted on the rows it is scored on, empty `sibling_probs`, constant `retrieval_score`, union-order `retrieval_rank`, no one-owner step | If accurate, that codebase's 0.836 is optimistic and its Stage 2 is decorative. Fixing this comes before any new feature. |
| P2 | Reads the rules correctly: one row per S1, empty lists valid, `candidate_pairs.tsv` = exact set fed to the matcher, no external lookups, France unlabeled, license/parameter limits | Avoids rejection and disqualification. |
| P3 | Out-of-fold Stage 1 scores as the only input to Stage 2 | The only honest way to train an entity-level stage. |
| P4 | Reports candidate recall and oracle F0.5 before tuning the matcher | Retrieval is the ceiling; V2 refuses to tune below it blind. |
| P5 | Keeps S2 and S3 ownership separate, S1 capacity unlimited | Matches the data structure (each record has ≤ 1 owner; an S1 has 0–11 records). |
| P6 | Refuses to invent a France score or a France-specific threshold without labels | Correct and rare. |
| P7 | Dense retrieval, BM25, graph propagation and cross-encoders pushed to late ablations or rejected | Matches the measured headroom: a perfect scorer on current candidates reaches 0.990, so retrieval upgrades can add at most ~1%. |
| P8 | Demands that the one-owner audit be reproduced before it becomes a hard invariant | Fair. **Reproduced on 27 Sep:** 7,638,365 labels, 7,638,365 distinct record IDs, 0 records with more than one owner (`cache/raw/train_gt.parquet`). |
| P9 | "Same address ≠ same business" and "one numeric span ≠ the house number" | Both correct and both already encoded in `er/` (number-set comparison; OOV-gated address-only matches). |
| P10 | Proposes character n-gram name retrieval as the first retrieval addition | The one V2 item worth testing now. **Tested 27 Sep (E-05c/E-07c):** name 5-grams on empty-address records lift their recall 0.781 → 0.873 and overall candidate recall 0.971 → 0.976, but end-to-end macro F0.5 moves only 0.9699 → 0.9707 (+0.0008, CIs overlap) at 7.2 GB. Rejected under the +0.002 rule; the frozen configuration stays. |

## Gaps

| # | Gap | Evidence | Consequence | Fix |
|---|---|---|---|---|
| G1 | **Rebuilds what exists.** Every "ADD" item (OOF Stage 2, true margins, channel provenance, one-owner + null, candidate audit, ledger) is implemented and measured in `er/` | `learn.oof`, `stage2.context` (`p_margin`, `p_r2`, `p_rank`), `features.candidates` (`sp_score`, `sp_rank`, `ex_hits`), `learn.assign`, `metric.blocking`, `experiments/experiments.csv` (21 runs) | P0 "reproduce 0.836" spends the last day re-deriving a 0.961 result | Adopt `er/` as the submission pipeline; treat V2's P1 as done |
| G2 | **No scale-consistency step.** V2 says "country-at-a-time" | Model trained on 76k–180k-entity state subsets; a test country has up to 810k. IDF, posting caps, collision counts and the OOV vocabulary all shift. First full-scale run dropped 0.973 → 0.956 for exactly this reason; state blocks (~150k S1) + cross-block margin brought it to 0.961 | Following V2 literally reproduces the 0.017 drop | Process in ~150k-entity state blocks; route stateless records to every block of their country with a 0.1 cross-block margin (`er/pipeline.py`) |
| G3 | **Submission package not mentioned.** README requires `output/`, `code/business_entity_resolution/{src, README.md, requirements.txt}`, filled `Documentation_template.md`, one zip | V2 §9 checklist has no packaging item | A perfect model with a wrong package is not evaluated | `submission/team_submission.zip` already has the required layout; rename to `<team_name>_submission.zip`, fill team fields |
| G4 | **Document length conflict unresolved.** Unstop page: "between 1 and 2 pages"; README: "no page limit" | `ref/contest.md` line 100 vs README §Submission Requirements | Risk of a formal rejection or a reviewer reading only page 1 | Make the first two pages of the filled template self-contained; keep the rest as appendix |
| G5 | **Roadmap does not fit the time.** P0–P5 is a multi-week programme | 23 h remain at review time | Only P0–P1 are feasible, and both are already done | Submit from `er/`; run the char n-gram ablation only if time remains |
| G6 | **Treats reproducible results as unverified.** The whitepaper numbers are labelled "friend-reported" | Fair on one point: the one-off forensic scripts were run as heredocs and not saved | Team may discard a measured 0.961 pipeline for an unmeasured plan | **Done 27 Sep:** `python -m er.forensics` (12 s) reproduces every fact F1–F10 and C4: 0 multi-owner records, 5.6% singletons, 26.6%/25.4% distractors, 100% country agreement, decoy signature, 92.2% sibling agreement, 24.5%/8.3% no-name-overlap, 97.7% of empty-address records matched, 100% Indic token alignment, no postal codes, row-order correlation +0.001 |
| G7 | **Stateless / unparsed-state records not addressed.** V2 has missingness flags but no routing rule | 13.8k Indian S1s with unrecognised state names scored 0.37 in the first full-scale run; ~26k unmatched records have no parsable state | Silent loss concentrated in one subgroup | Learned component→state map with fallback to "score in every block of the country" (`pipeline.plan`) |
| G8 | **No measured numbers anywhere.** Every capability is "hypothesis", every cost "moderate" | V2 §2.1: "the metric is not independently verifiable from this workspace" | Cannot prioritise: char n-grams vs address-first vs parser are all "ABLATION, high priority" | Use the existing strata: normal 98.2% / address-only 95.8% / Indic 98.5% / empty-address 78.1% retrieved. The weak class is empty-address → name char n-grams first, address-first path second |
| G9 | **Deterministic "safe" address views could double-count.** V2 proposes numeric exact/prefix/nearby relations as separate features | Already in `er/` as `ncover`, `ndiff`, `ntrunc`, `first_eq`; the near-miss decoy signature is Δ ≤ 10 (63.7% of decoys vs 1.4% of positives) | Fine as designed; only a gap if implemented without the "all numbers" set | Reuse `views.addr_view("A2")` |
| G10 | **Ranking objective listed as an ablation without a decision unit** | LambdaRank optimises NDCG per record; the metric is macro F0.5 per S1 with an absolute threshold | Likely to spend a run on a mismatched objective | Deprioritise; the per-S1 expected-F0.5 rule was already tested (+0.0004, rejected) |

## Items where V2 and `er/` agree (no action)

- Separate S2/S3 owner choice ≡ global argmax over unique record IDs.
- Support features exclude the candidate edge itself; other-source evidence kept as max/count/sum separately.
- No country feature in the model; France handled by language-neutral features only.
- Dense retrieval not in the core path.

## Recommended order for the remaining time

1. Submit from `er/` (0.961 full-scale, validator PASS). Do not rebuild.
2. Fill team name and members; decide document length (G4); upload the zip (G3).
3. ~~Char n-gram ablation~~ — run, rejected (+0.0008 < +0.002; see P10).
4. ~~Save `er/forensics.py`~~ — done (see G6).

## Note on scope

V2 reviews a codebase at `/Users/divyanshu/.../src/pipeline.py` that is not on this machine. Its description of that code's bugs is taken at face value here. If that codebase is the one being submitted, its Stage 2 leakage is the first thing to fix, and `er/learn.py` + `er/stage2.py` are a drop-in reference for the correct OOF construction.
