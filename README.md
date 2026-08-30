# Ozon eCup 2026 — Team A solution

## Overview

This branch is the reproducible Team-A research and delivery package for the
30-day user GMV prediction task. The competition metric is RMSLE, so model
ensembles are assembled in `log1p` space. The repository preserves the actual
research history, runners, reports and manifests while keeping large raw/cache
artifacts outside Git.

## Final solutions

| Solution | Formula | Expected SHA256 | Public LB evidence |
|---|---|---|---|
| `SUBMIT_STRONGEST55_TEAMB45` | 55% `STRONGEST_CURRENT` + 45% level-aligned Team-B | `1ce85203…a14fb4` | No confirmed score found; geometry report forecast 1.64823 (estimate, not fact) |
| `SUBMIT_JOINT86_TEAMB14` | 86% `SUBMIT_JOINT_V2` + 14% level-aligned Team-B | `85d9cd64…dac02` | 1.6458200196207617, recorded in the teammate reproduction request |

Both final CSV files reproduce byte-for-byte from committed frozen component
predictions. The upstream generation script for the 86% `SUBMIT_JOINT_V2`
anchor was not found; that boundary is explicitly marked
`PROVENANCE_INCOMPLETE`, while its bytes, SHA, audits and public score are
preserved.

## Repository structure

- `src/` — clean baseline feature/model/validation utilities;
- `scripts/reproduce_final.py` — common final reproduction entry point;
- `experiments/` — 95-entry unified index and preserved `exp_001…exp_071` reports;
- `research/new_directions/` — later EXP069–EXP090 research, runners and reports;
- `research/legacy_team_a/` — exact source/report snapshot from the active Team-A worktree;
- `research/submission_geometry/` — geometry code, reports and two historical champions;
- `reproducibility/` — standalone packages for the two final submissions;
- `submissions/` — selected final and direct-parent CSV files only;
- `docs/TEAM_A_SOURCE_INVENTORY.csv` — 8,022-file SHA256 forensic inventory.

## Data

Place the competition data at:

```text
data/raw/train.parquet
data/raw/sample_submit.csv
```

Expected `train.parquet` SHA256:

```text
5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0
```

The feature builders enforce the family-specific cutoff boundary (`<` or `<=`)
and never read events after cutoff; targets use
`[cutoff, cutoff + 30 days)`. Raw data, generated panels, caches and large
research artifacts are ignored by Git and remain traceable through the source
inventory.

## Environment setup

The exact frozen rebuild uses Python 3.13:

```powershell
py -3.13 -m venv .venv-rebuild
.\.venv-rebuild\Scripts\python.exe -m pip install -r requirements.txt
```

Historical full training used two incompatible pinned environments. Install
`reproducibility/SUBMIT_STRONGEST55_TEAMB45/requirements-strongest.txt` with
Python 3.13/CUDA 12.6, and `requirements-team-b.txt` with Python 3.11. The
Team-B requirements are also stored in the JOINT package.

## Reproduce from precomputed predictions

```powershell
python scripts/reproduce_final.py --solution SUBMIT_STRONGEST55_TEAMB45 --from-precomputed
python scripts/reproduce_final.py --solution SUBMIT_JOINT86_TEAMB14 --from-precomputed
```

Each command validates columns, 250,000 unique users, order, finite/nonnegative
predictions, expected SHA256, byte identity, maximum absolute prediction
difference and RMS log-space difference.

## Reproduce from raw data

```powershell
python scripts/reproduce_final.py --solution SUBMIT_STRONGEST55_TEAMB45 --from-raw `
  --raw-data data/raw/train.parquet `
  --strongest-python .venv-strongest/Scripts/python.exe `
  --team-b-python .venv-team-b/Scripts/python.exe

python scripts/reproduce_final.py --solution SUBMIT_JOINT86_TEAMB14 --from-raw `
  --raw-data data/raw/train.parquet `
  --team-b-python .venv-team-b/Scripts/python.exe
```

For STRONGEST this runs raw → features → three tabular models → frozen
SEQ/ETX predictions → STRONGEST → retrained Team-B → final blend. Full neural
retraining commands are preserved in its package README but are intentionally
not the default multi-hour audit. For JOINT, raw mode retrains Team-B and blends
it with frozen `SUBMIT_JOINT_V2`; a fully raw JOINT rebuild is impossible until
the missing upstream anchor generator is recovered.

## Main model families and ensemble

`STRONGEST_CURRENT` combines CAP, UNC and DIST tabular LightGBM families with a
TCN-like SEQ ensemble and the sparse-event ETX transformer. The external
Team-B vector combines LightGBM regression/classification, a 16-bin
distribution head, XGBoost and CatBoost over recency, post-order and behavior
features. Submission-geometry, ORTH and A1/A2 research operate on prediction
vectors and are historical ancestors of the JOINT anchor.

See [solution architecture](docs/SOLUTION_ARCHITECTURE.md),
[experiment index](experiments/README.md) and
[reproducibility guide](docs/REPRODUCIBILITY.md) for details.

## Final submissions

- `submissions/SUBMIT_STRONGEST55_TEAMB45.csv`
- `submissions/SUBMIT_JOINT86_TEAMB14.csv`

Do not treat geometry forecasts as measured leaderboard scores. The exact LB
provenance and remaining gaps are recorded in each package manifest and in the
packaging report.
