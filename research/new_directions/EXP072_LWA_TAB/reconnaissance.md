# EXP072 LWA TAB — reconnaissance

## Status

**PASS**

All evaluator, row-key, feature-schema, cutoff-count, EXP-037 reconstruction, and EXP069 correction-parity gates passed. No model was trained during reconnaissance.

## First substantive check: EXP069 correction

The saved correction was reproduced with maximum absolute error `0.000e+00` (required `<=1e-9`).
The saved `raw_correction` equals `z_cond_fresh-z_cond_clean` exactly and already contains `p_dist`. Historical saved OOF used donor-fold winsor bounds but held-fold centering. EXP072 uses the newly specified donor-derived center and multiplies frozen `p_dist` after processing the raw mu difference.

## Canonical OOF/evaluator parity

- Rows: `770,616`; duplicate `(fold,user_id)` keys: `0`; fold sizes: `[188518, 191025, 193694, 197379]`.
- EXP-037 wCV: `1.747509862493`; folds: `[1.7668833567679014, 1.7605095767346348, 1.7486292239577714, 1.7412785664162476]`.
- EXP-037 reconstruction maximum log error: `5.912e-08`.
- Add-one `pred_fresh_contrast`: nested delta `-0.000224686`, improved `4/4`, alphas `[1.0, 1.05, 1.0, 1.05]`.
- Add-one `pred_btyd`: nested delta `-0.000269182`, improved `4/4`, alphas `[0.05, 0.05, 0.1, 0.1]`.
- Add-one `pred_hurdle_e11`: nested delta `+0.000005234`, improved `1/4`, alphas `[0.05, 0.05, 0.05, 0.0]`.

## Frozen features and cutoff construction

- `43` required feature caches exist; each has `user_id + 227` columns with byte-identical column order and dtype schema.
- CLEAN cutoff counts are `{'2025-09-04': 18, '2025-09-18': 20, '2025-10-02': 22, '2025-10-16': 24}`; NOOV counts are `{'2025-09-04': 13, '2025-09-18': 13, '2025-10-02': 12, '2025-10-16': 9}`.
- NOOV specification discrepancy: literal interval intersection would keep 11 cutoffs for `2025-10-02`, but the fixed arm definition requires 12 and drops only `2025-10-22`; the explicit membership was used.
- Every feature cache is the canonical `feat_*_LnormNone.parquet` output. No feature was rebuilt or written.

## Filesystem adaptation: EXTRA b3 panels

The review packet stated that the 13 EXTRA `panel_*_b3.parquet` caches existed, but `13` are absent. The experiment therefore applies the exact `panel_users` three-block rule in memory from the canonical raw events. The adapter was equality-checked against the existing `2025-10-16` and `2026-02-13` b3 caches. No cache is written outside EXP072.

| EXTRA cutoff | b3 rows | source |
|---|---:|---|
| 2025-10-22 | 199,025 | canonical_in_memory_rebuild |
| 2025-10-29 | 200,827 | canonical_in_memory_rebuild |
| 2025-11-05 | 202,532 | canonical_in_memory_rebuild |
| 2025-11-12 | 204,319 | canonical_in_memory_rebuild |
| 2025-11-19 | 206,668 | canonical_in_memory_rebuild |
| 2025-11-26 | 209,184 | canonical_in_memory_rebuild |
| 2025-12-03 | 212,048 | canonical_in_memory_rebuild |
| 2025-12-10 | 215,272 | canonical_in_memory_rebuild |
| 2025-12-17 | 218,437 | canonical_in_memory_rebuild |
| 2025-12-24 | 219,858 | canonical_in_memory_rebuild |
| 2025-12-31 | 222,367 | canonical_in_memory_rebuild |
| 2026-01-07 | 226,218 | canonical_in_memory_rebuild |
| 2026-01-14 | 233,152 | canonical_in_memory_rebuild |

## Estimator

The S1-E11 positive regressor configuration was recovered from `exp_013_s1_e11_two_part.md` and `src/config.py`: the frozen default LightGBM regression parameters for 600 rounds. The same configuration is fixed for all four arms and both user sides; no sweep or early stopping is permitted.

## Input integrity

`artifact_manifest.csv` records `128` independently SHA256-hashed inputs with row/column counts where applicable. Geometry scores and weights were not read or used. Public-LB use: **false**.
