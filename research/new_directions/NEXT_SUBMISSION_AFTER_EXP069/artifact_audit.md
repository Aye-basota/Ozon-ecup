# Artifact audit — EXP069 / EXP070 / EXP071 and the geometry inputs

Independent re-hash of every file under `research/new_directions/` plus the external
geometry inputs. Full table: `experiment_inventory.csv` (124 rows).

## 1. Checksum verification

Every hash was recomputed from the bytes on disk, not read from the experiments'
own ledgers.

| Experiment | files hashed | declared in `checksums.sha256` | mismatches | declared-but-missing |
|---|---|---|---|---|
| EXP069 | 35 | 30 | **0** | 0 |
| EXP070 | 58 | 28 | **0** | 0 |
| EXP071 | 20 | 19 | **0** | 0 |
| external geometry inputs | 11 | n/a | n/a | 0 (all present) |

Files present but not covered by a ledger are only `checksums.sha256` itself,
`__pycache__/*.pyc` (4 in EXP069, 1 in EXP070) and EXP070's 28 `_label_cache/*.parquet`
intermediate label caches. No experiment artifact is unhashed.

The geometry basis `submission_geometry/cache/Z.npz` hashes to
`443e80472f321e796faccfe0e9ea6954653397eb74311e7abef7471ea3064dfe`, exactly the value
EXP069 recorded in `test_span_projection.json`. The basis has not moved since EXP069 ran.

`SUBMIT_NEXT_BEST.csv` hashes to `95f3fa982d8173e5...`, matching the SHA that
`07_ALIGNED_TEST_COLUMNS.md` binds to the `1.6466079084` public result.

## 2. Parse / schema checks

All 20 JSON files parse. All 24 CSV files parse. All 9 parquet files open with the
row counts their reports claim:

- `btyd05_fresh1_OOF.parquet` 770,616 × 8, `fresh_conditional_OOF.parquet` 770,616 × 15
- `btyd05_fresh1_TEST.parquet` 250,000 × 6, `fresh_conditional_TEST.parquet` 250,000 × 13
- `count_value_moe_raw_OOF.parquet` 576,922 × 8, `count_probabilities_OOF.parquet` 576,922 × 11
- `etx_fresh_raw_OOF.parquet` 197,379 × 6

## 3. Missing, partial or stale artifacts

| # | Finding | Severity | Effect on this task |
|---|---|---|---|
| 1 | **EXP070 has no `2025-10-02` fold.** `count_value_moe_raw_OOF.parquet` and `count_probabilities_OOF.parquet` hold 576,922 rows = three folds. Canonical four-fold wCV is therefore not computable from any saved artifact. | high | EXP070 excluded from the submission (already its own recommendation). |
| 2 | **EXP070's standardized OOF parquet stores only the REAL arm** (`candidate_name` has the single value `count_value_moe_raw`). The shuffled control and the two derived endpoints are absent from the standardized file. | medium | Recovered instead from the un-ledgered-but-hashed `_fold_*.npz` caches, which do contain `z_shuffled`; REAL−SHUFFLED is fully verifiable. Noted as a standardization gap, not a data gap. |
| 3 | **EXP071 produced no TEST artifacts** (`etx_fresh_raw_TEST`, `etx_fresh_contrast_*`). Its report declares them `NOT PRODUCED` under the reject-pilot early-stop policy. | expected | Correct behaviour; nothing fabricated. |
| 4 | **EXP071's OOF file is pilot-scope only** — 197,379 rows, `scope = PILOT_ONLY_SEED42`, one fold. Its `fold_metrics.csv`, `nested_selection.csv`, `bootstrap_metrics.csv`, `oof_projection_metrics.json` and `test_span_projection.json` are stubs marked `NOT_RUN`. | expected | Correctly labelled; the donor-fold raw vectors were not retained, so the winsor bounds cannot be re-derived from scratch (see §4). |
| 5 | **EXP069 `preprocessing_parameters.json` global q005/q995/center differ from the per-fold nested values used in the saved OOF vector.** | benign | Verified: the difference is a *pure per-fold constant* (within-fold sd exactly `0.0`), and the bridge reproduces the historical wCV to `0.0`. See §4. |
| 6 | **EXP069 TEST extensive probability is a reconstruction.** `ZERO2D_DIST_test.npz` (`ef3e528c…`) is a same-recipe CLEAN-only S1-DIST rebuild, `reference_reproduced = false`, mean `|Δz| = 0.0445`, max `0.7334`, corr `0.99919`. | medium | Real limitation. It multiplies the FRESH correction per user, so it modulates the *new orthogonal component* too, not only the parallel part. Carried into the uncertainty budget. |
| 7 | No experiment writes into `current_best/`, `submission_geometry/` or `submissions/` of the geometry workspace. | — | The incumbent and previous incumbent are untouched. |

## 4. Items re-derived rather than trusted

- **EXP069 preprocessing bridge.** Applying each fold's registered donor-derived
  `q005/q995/center` to the historical raw contrast reproduces the saved nested
  correction up to a per-fold constant with within-fold standard deviation exactly
  `0.0` on all four folds, and the per-fold RMS differences `0.031244 / 1.856e-06 /
  0.018630 / 0.011840` match the registered values to 12 digits. Bridge wCV minus
  historical wCV is exactly `0.0`. The canonical per-fold log-offset evaluator removes
  those constants, so the claim is confirmed.
- **EXP069 combined-OOF decomposition.** `z_predict = z_exp037 + 0.05·(z_btyd − z_exp037)
  + correction` to `2.05e-15`, and `z_base` is byte-equal to the EXP-037 column. The
  file therefore carries the exact component split the deployment needs.
- **EXP070 label bins.** `count_class` reproduces exactly from `N30` with cuts
  `{0}, {1}, [2,3], [4,7], [8,∞)`; five bins retained; probability row sums deviate from
  1 by at most `5.157e-08` (report claims `5.16e-08`); all probabilities in `[0,1]`.
  Every EXP070 row is present in the canonical bank with target error `0.0`.
- **EXP071 alignment.** `etx_fresh_raw_OOF.parquet` keys equal the canonical
  `2025-10-16` fold exactly, and its saved `user_side` equals `splitmix64(user_id)&1`.

## 5. Things that cannot be independently re-derived

- EXP071's donor-fold winsor bounds (`lo/hi/center` for the REAL and VOL arms) come from
  the three donor folds' raw ETX contrasts, which were not saved. The bounds in
  `pilot_metrics.json` had to be taken as given; every downstream number was then
  recomputed from them and reproduced exactly, so the arithmetic is verified even though
  the bound-fitting step is not.
- EXP069's production encoder `model_SEQ-D3A-BASE-S42-TEST.pt` and the three-seed /
  two-side head training are not re-runnable inside this task's compute budget; the
  training audit's internal invariants (frozen-encoder checksum unchanged, positive-only
  target pool, cutoff-grid assertions) were read, not re-executed.
