# Business Entity Resolution (Amazon ML Challenge 2026)

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.

## Setup
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export ER_DATA=/path/to/student_resource/dataset   # default: ref/.../student_resource/dataset
```
Machine used: 12 CPU cores, 16 GB RAM. No GPU and no external data or services.

## Run (end to end)
```
.venv/bin/python -m er.data                 # TSV -> Parquet (cache/raw)
.venv/bin/python -m er.split                # DEV-TRAIN / DEV-VAL state subsets (cache/dev)
.venv/bin/python -m er.exp E-07b devval     # builds DEV pair features + stage-1 OOF scores
.venv/bin/python -m er.pipeline train       # fits stage-1 / stage-2 LightGBM + threshold (cache/models)
.venv/bin/python -m er.pipeline run test    # scores the test set in state blocks
.venv/bin/python -m er.pipeline finish test # writes submission/output/*.tsv
```
Validate: `python3 utils/validate_submission.py -m submission/output/matching_results.tsv -c submission/output/candidate_pairs.tsv -t <dataset>/test`

## Layout
`er/` holds the source. `docs/RESEARCH_SPEC.md` has the method and the experiment protocol. `experiments/experiments.csv` logs every experiment.
