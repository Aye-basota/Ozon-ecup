# Ozon E-Cup clean research workspace

This repository is the minimal reproducible environment for new model research.
The task is to predict each eligible user's next-30-day GMV. The competition
metric is RMSLE; models and blends operate in `z = log1p(prediction)` space.

## Ground truth and validation

- Raw events: 30,631,006 daily rows, 250,000 users, `2025-01-01..2026-02-13`.
- Test cutoff: `2026-02-13`.
- Eligibility: at least one observed day in each of the three latest 30-day blocks.
- Target: GMV sum in `(cutoff, cutoff + 30 days]`.
- Validation folds: `2025-09-04`, `2025-09-18`, `2025-10-02`, `2025-10-16`.
- A train cutoff is legal only when `train_cutoff + 30 days <= validation_cutoff`.
- wCV: per-fold RMSLE after an optimal global log offset, weighted `1:2:4:8`.
- Row key: `(cutoff, user_id)` for OOF and `user_id` for TEST.

Feature code must enter through `src.features.build_features(cutoff_date)`.
That function reads only rows with `event_date <= cutoff_date`.

## Two baselines that must not be conflated

**Public incumbent:** `1.6466079084`. It is a TEST-only submission-geometry
result. No correct fold-safe OOF equivalent is available, so it is not an
offline comparison baseline.

**Canonical offline baseline:** `EXP_037_STRONGEST_CURRENT`, frozen wCV
`1.7475098625201952` on 770,616 aligned OOF rows. Its log-space recipe is
`0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-AVG3 + 0.225 ETX-AVG3`.
The compact canonical artifact is
`artifacts/oof/EXP_037_STRONGEST_CURRENT.parquet`.

## Setup and validation

1. Copy `config/paths.example.yaml` to `config/paths.local.yaml` and set absolute
   external paths. The local file is ignored by Git.
2. Install the unchanged minimal dependency set: `python -m pip install -r requirements.txt`.
3. Run:

```text
python scripts/validate_environment.py
python scripts/validate_baseline.py
```

Neither command trains a model.

## New experiment

1. Copy `experiments/TEMPLATE` to `experiments/EXP_XXX_name`.
2. Fill its hypothesis, exact change, validation, and success gate before running.
3. Run a tabular challenger with:

```text
python scripts/run_tabular_experiment.py --config experiments/EXP_XXX_name/config.yaml
```

Add `--test` only after the experiment's OOF decision permits TEST inference.
The runner writes:

- `artifacts/oof/EXP_XXX_NAME.parquet`;
- `artifacts/test/EXP_XXX_NAME.parquet` when requested;
- experiment-local `metrics.json` and `report.md`.

Allowed experiment verdicts are `PASS`, `WEAK_SIGNAL`, `REJECT`, and `INVALID`.
After review, update `registry/experiments.csv` explicitly.

## Layout

- `config/`: competition truth and external path template.
- `src/`: reusable data, features, models, validation, metrics, blending, and artifact APIs.
- `baselines/`: canonical baseline definition and provenance.
- `artifacts/`: standardized predictions; only the compact canonical OOF is versioned.
- `experiments/`: new experiment folders and template.
- `registry/`: curated experiment, model, submission, and source provenance tables.
- `research_packets/`: references to separate research lines.
- `reports/`: repository-level reports only.

Raw data, historical caches, checkpoints, old experiment reports, and submission
dumps remain external. The separate geometry workspace is
`C:/Users/Admin/Desktop/submission_geometry_research`; see
`research_packets/submission_geometry_reference.md` for the PASS handoff.
