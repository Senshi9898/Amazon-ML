# Business Entity Resolution — Amazon ML Challenge 2026

**Team Grokking:** Divyanshu (Team Leader), Aaditya Rawat, Shristi Chandra

For each clean reference business (S1), identify every record in two noisy sources (S2, S3) that refers to the same business across the US, India, and France.

**Evaluation Metric:** Macro F0.5 per S1 entity, including singletons.

## Final Results

| Metric | Result |
|---|---|
| **Final Score** | **98.4/100** |
| **Leaderboard Standing** | **Top 2.9%** |
| **Competition Scale** | **32,000+ teams** |
| **Competition** | Amazon ML Challenge 2026 |
| **Team** | Grokking |
| **Team Leader** | Divyanshu |

The final score and leaderboard standing reflect the team's reported competition results.

## Project Overview

Built an entity-resolution pipeline to match clean reference businesses against noisy business records from two independent sources. The system combines deterministic normalization, candidate retrieval, pairwise feature engineering, LightGBM classification, and threshold-based ownership assignment.

The primary objective was to maximize macro F0.5 while minimizing false-positive assignments, preserving candidate-level evidence, and ensuring that every noisy record is assigned to at most one reference business.

## Technical Architecture

The pipeline follows a retrieval-then-classification approach.

1. **Normalization:** Generated deterministic, reversible representations using transliteration, legal-form canonicalization, and address-number views.
2. **Country Partitioning:** Preserved the existing US, India, and France partitioning without introducing an unvalidated France-specific model or threshold.
3. **Candidate Generation:** Combined typed-token TF-IDF retrieval with exact structured-key retrieval, followed by deduplication using deterministic tie-breaks.
4. **Feature Engineering:** Computed pairwise features covering name similarity, address and number agreement, legal-form relationships, and genuine TF-IDF retrieval scores.
5. **Stage 1 Matching:** Trained a LightGBM binary classifier to estimate whether each candidate pair represents the same business.
6. **Assignment:** Assigned each S2/S3 record to its highest-scoring S1 candidate if its score exceeded a globally calibrated threshold; otherwise, assigned NULL.
7. **Validation:** Enforced candidate-constrained predictions and one-owner-per-record assignment, followed by the official submission validator.

### Key Technical Decisions

- **Genuine retrieval scores:** Replaced an earlier candidate-union pseudo-rank with the actual typed-token TF-IDF score.
- **Conservative assignment:** Used a global threshold and NULL assignments to avoid uncertain matches and control false positives.
- **Leakage-aware evaluation:** Used separate calibration and locked audit partitions for threshold selection and evaluation.
- **Evidence-driven model selection:** Retained Stage 1 LightGBM as the core matcher after entity-level Stage 2 rescoring failed to demonstrate a consistent improvement across independent seeds.
- **Competition-compliant design:** Used only the provided challenge data, without external registries, geocoding, or web lookups.

## Experiments and Ablations

The final architecture reflects measured experiments rather than adding complexity without evidence.

| Component | Decision |
|---|---|
| Typed-token TF-IDF retrieval | Retained |
| Exact structured-key retrieval | Retained |
| Pairwise LightGBM classifier | Retained |
| Global threshold with NULL assignment | Retained |
| Entity-level Stage 2 rescoring | Excluded due to inconsistent gains across seeds |
| Character n-gram retrieval | Excluded due to marginal oracle gains relative to additional candidate edges |
| Stage 2 hard-negative reweighting | Excluded after reducing macro F0.5 |
| France-specific model or threshold | Excluded due to insufficient validation evidence |
| Embeddings and cross-encoders | Excluded |
| Graph propagation | Excluded |
| External data and geocoding | Excluded; also prohibited by competition rules |

## Validation and Reproducibility

The repository includes an end-to-end execution pipeline, submission validation utilities, and checks for candidate-ID consistency and single-owner assignment.

**Important reproducibility caveat:** The exact local model artifacts and configuration associated with the submitted output have not been conclusively tied to the official final score. Historical internal scores and sampled-audit results should not be presented as reproducible full-evaluation results.

Before relying on a regenerated output, verify that:

- The model and configuration match the current feature and assignment schema.
- The generated output files are present and non-empty.
- The official submission validator passes.
- Candidate membership and one-owner constraints pass independently.
- Runtime and memory usage remain acceptable for the full dataset.

## Tech Stack

- **Language:** Python
- **Machine Learning:** LightGBM
- **Information Retrieval:** TF-IDF, exact structured-key retrieval
- **Feature Engineering:** String similarity, address and number agreement, legal-form relationships
- **Data Processing:** Deterministic normalization, country partitioning, candidate deduplication
- **Evaluation:** Macro F0.5, threshold calibration, locked audit validation

## Repository Structure

```text
ARCHITECTURE.md                 Frozen architecture, decision table, and known risks
models/                         Stage 1 model and configuration
artifacts/                      Data and settings signatures
output/                         Matching results and candidate pairs
run_final.py                    End-to-end training and inference entry point
utils/validate_submission.py    Official-format submission validation
