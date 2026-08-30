# EXP070_COUNT_VALUE_MOE reconnaissance

Reconnaissance was completed before EXP070 implementation or training.

## Canonical lineage

- Historical workspace: `C:/Users/Admin/Desktop/OZON-E-CUP`.
- Clean research workspace: `C:/Users/Admin/Desktop/e-cup-research-clean`.
- Geometry workspace: `C:/Users/Admin/Desktop/submission_geometry_research`.
- Canonical feature/target implementation: historical `src/features.py`.
- Canonical folds and fold-offset evaluator: historical `src/validation.py`; validation cutoffs are `2025-09-04`, `2025-09-18`, `2025-10-02`, `2025-10-16`, with fold weights `1:2:4:8`.
- Canonical EXP-037 aligned OOF: clean `artifacts/oof/EXP_037_STRONGEST_CURRENT.parquet` and geometry packet `gpt_pro_research_packet/06_ALIGNED_OOF.parquet`.
- Canonical aligned TEST/65-source geometry input: `gpt_pro_research_packet/07_ALIGNED_TEST.parquet`; QR/SVD projection code is under `submission_geometry/` in the geometry workspace.

## Exact target and purchase-day semantics

The canonical target builder filters daily event rows to `(T, T+30]`, keeps rows with `gmv > 0`, groups by `user_id`, sums `gmv`, left-joins to the eligible panel, and fills absent users with zero. The raw event date is a timezone-free Polars `Date`, so the calendar-day boundary is the stored `event_date` with no timezone conversion. EXP070 uses the same positive-purchase predicate and window, and defines `N30` as the number of distinct `event_date` values satisfying that predicate. The source is a daily user-event table; `n_unique(event_date)` is nevertheless used explicitly rather than assuming uniqueness.

Features are produced only from rows with `event_date <= T`. Training cutoffs satisfy `T + 30 days <= V`. Validation uses the canonical three-block panel, training uses the canonical one-block panel, and the cutoff grid is restricted to the clean corridor ending `2025-10-16`.

## S1-E10 cache reuse

The exact normalized-long S1-E10 feature list is `artifacts/feats_S1-E10.txt`: 227 columns, SHA256 `bfc340662ae46276a909edbcc215fa4c6ed7df6f6ca5edd12360f6a870afc46a`. The S1-E11 and both S1-DIST feature manifests are byte-identical to this list. Existing matrices are `data/processed/feat_YYYYMMDD_LnormNone.parquet`; EXP070 reads these matrices directly and never writes duplicate feature matrices.

The latest-fold training cutoffs are the 24 canonical dates `2025-04-03..2025-09-11` on the seven-day grid. Earlier full-fold runs reuse the corresponding prefixes (18/20/22/24 cutoffs). Validation matrices for all four folds and the TEST matrix at `2026-02-13` already exist.

## Reusable prediction directions

- `oof_S1-E10.npz`: direct normalized-long S1-E10 OOF.
- `oof_S1-DIST.npz`: 16-bin S1-DIST OOF and the EXP-037 DIST slot direction.
- `oof_S1-E11.npz`: two-part S1-E11 OOF.
- The aligned OOF packet also provides EXP-037, ETX-AVG3, SEQ-AVG3, SEQ-D3A-AVG3, BTYD, BTYD05, FRESH-CONTRAST, MHZ-FULL, and HOLIDAY-YOY on the exact canonical row keys.

No public-LB data, incumbent TEST geometry, prediction disagreement, cohort identifier, or TEST statistic will enter fitting or selection.

## Geometry projection

The geometry workspace constructs a centered log-prediction matrix and uses QR/SVD bases in `submission_geometry/`. TEST projection is permitted only after a PASS candidate exists. EXP070 will project onto the frozen current 65-source/rank-57 span and will not refit geometry weights.
