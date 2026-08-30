# Occurrence cache missing

## Status

The teammate `_best_bas_research` cache was not found on local `C:`/`D:` project locations. The supplied bundle explicitly says it was excluded (~80 fold NPZ, ~15 TEST NPZ, ~9.7 GB feature cache). Exact TEST CSVs and validation summaries exist, but row-level canonical OOF does not.

## Missing artifacts

- cache root: `_best_bas_research/checkpoints/folds/` and `_best_bas_research/checkpoints/test/`;
- core folds: `cap/unc/dist/hurdle__{2025-09-04,2025-09-18,2025-10-02,2025-10-16}.npz` with `user_id,y,z` and `p,mu` for hurdle;
- occurrence folds (32 expected): `occ_r10_fast__<fold>.npz, occ_r16_bal__<fold>.npz, occ_r22_stable__<fold>.npz, occ_r14_multiscale__<fold>.npz, occ_r18_wide__<fold>.npz, occ_r24_multiscale__<fold>.npz, occ_r12_wide__<fold>.npz, occ_r20_shallow__<fold>.npz` for four folds, keys `user_id,y,p`;
- occurrence TEST (8 expected): `occ_<name>_test.npz`, keys `user_id,p`;
- helper/meta state: `meta_raw_test.npz`, `final_candidate_bank.npz`, saved stable-stack predictions/recipes needed for `best_bas` and `_best_bas_research` replay;
- row-level final sources needed for the unified audit: `occ_meta_B/final6h_B`, `occ_raw_X3/extra90_3`, and their exact base/friend OOF on `(cutoff,user_id)`.

## Recoverability and cost

The scripts are present and the cache is recoverable in principle. Historical runtime shows the eight occurrence families alone took ~4.50 CPU-hours once the 9.7 GB core cache existed; extra90 materialization took ~31 minutes. Rebuilding the missing core bank from scratch invokes the 23h/14h lineage and is conservatively **20–30 CPU-hours plus disk**, with no need for a new neural/GPU run if archived neural predictions remain usable. This exceeds the 6–10h automatic-run ceiling, so it was not started.

## Required handoff

Copy only compact `checkpoints/folds/*.npz`, `checkpoints/test/*.npz`, recipe manifests and selected row-level final OOF; do not transfer the 9.7 GB processed feature cache unless replay is actually required. Every fold artifact must carry explicit `user_id` and the fold/cutoff must be unambiguous.
