# Initialization report

## Repository

- New repository: `C:/Users/Admin/Desktop/e-cup-research-clean`
- Source repository: `C:/Users/Admin/Desktop/OZON-E-CUP` (read-only during initialization)
- Initialization date: `2026-08-26`

## Included

- Configurable external data/artifact paths and canonical competition config.
- Canonical raw-data loader, cutoff-safe feature/target generation, fold definitions,
  RMSLE/wCV evaluation, key-safe OOF comparison, reusable tabular model families,
  log-space blending, and standardized artifact writers.
- Frozen `EXP_037_STRONGEST_CURRENT` OOF as a compact 9,518,701-byte Parquet.
- Curated prediction-source manifest for CAP, UNC, DIST, ETX, SEQ, major tabular,
  BTYD, occurrence, FRESH, SEQ65, and geometry sources.
- Experiment, model, submission, and source-provenance registries.
- Executable experiment template and environment/baseline validation scripts.

## Kept external

- Raw data and all processed feature/panel caches.
- Historical OOF/TEST component banks other than the compact canonical baseline.
- Model checkpoints, CUDA/sequence training artifacts, old reports, old experiments,
  submission dumps, worktrees, caches, notebooks, agent/state documents, and Git history.
- The complete 65+ source submission-geometry workspace and its fitted caches.

## Baselines

- Canonical offline baseline: `EXP_037_STRONGEST_CURRENT`.
- Expected and reproduced wCV: `1.7475098625201952`.
- OOF rows / unique row keys: `770616 / 770616`.
- Fold sizes: `188518, 191025, 193694, 197379`.
- Current public incumbent: `1.6466079084`, TEST-only geometry result.
- The public incumbent has no correct fold-safe OOF equivalent and was not used as
  an offline baseline.

## Validation result

- `python scripts/validate_environment.py`: PASS.
- `python scripts/validate_baseline.py`: PASS.
- Baseline absolute numerical error: `0.0`.
- Canonical folds and `T+30<=V` safety audit: PASS.
- Imports and dependency availability: PASS.
- Standardized OOF schema, row alignment, uniqueness, and finite-value checks: PASS.
- Experiment template and all four registries/manifests: present and parseable.

## Missing by design or unavailable

- No fold-safe OOF equivalent for public incumbent `1.6466079084`.
- No canonical OOF for teammate occurrence components `occ_meta_B` / `occ_raw_X3`.
- No standalone canonical OOF artifact for the prepared SEQ65 TEST submission.
- Neural checkpoints and full historical prediction banks were not copied; their
  reviewed sources are registered by external path and SHA256 where reusable.

## Important external paths

- Data: `C:/Users/Admin/Desktop/OZON-E-CUP/data`
- Historical artifacts: `C:/Users/Admin/Desktop/OZON-E-CUP/artifacts`
- Historical source repository: `C:/Users/Admin/Desktop/OZON-E-CUP`
- Submission geometry: `C:/Users/Admin/Desktop/submission_geometry_research`
- Teammate latest/occurrence bundle:
  `C:/Users/Admin/Desktop/OZON-E-CUP/пайплайн сокомандника/latest`

## Final verdict

**READY FOR EXPERIMENTS**
