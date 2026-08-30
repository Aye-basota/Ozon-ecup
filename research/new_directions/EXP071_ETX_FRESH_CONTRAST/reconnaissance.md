# EXP071 reconnaissance

## Status

All hard prerequisites were located. The four `ETX-01-S42` fold checkpoints and the seed-42 TEST checkpoint expose the unmodified ETX model state and full config metadata. Every checkpoint matches the registered 5-block, 128-dimensional ETX-01 configuration. Checkpoint hashes are recorded in `artifact_manifest.csv` and `config.json`.

## ETX representation and production fix

`src/etx.py` builds sparse real-event tokens, causal SDPA, and calendar-time ALiBi. The query token is inserted at position `n_events`; after five blocks and the final LayerNorm, `zq = h[:, n_events, :]` has dimension 128. The original direct head consumes `[zq, event_mean, last_event]`. EXP071 hooks only `zq` and does not change any weight.

The exact EXP-037 static-context correction is implemented by `research/strategies/results/ETX2/depth_fix.py`: clipped event history and query depth are both 289 on TEST, and the historical DCW checkpoint-compatible TEST prediction shifts the query cutoff weekday from Friday back to the Thursday support seen at every encoder-training cutoff. EXP071 uses actual weekdays for CLEAN/OOF and EXTRA representations; EXTRA applies only the depth clip/cap. The registered TEST DCW shift is used only for parity with the frozen production checkpoint.

## Historical conditional-positive construction

`src/seq_cond.py` supplies the exact 13 EXTRA cutoffs, `splitmix64(user_id)&1`, positive-only target filter, per-cutoff target centering, equal-step training, and the equal-volume early-CLEAN control. `src/fresh_contrast.py` supplies symmetric two-sided user cross-fitting and the q0.5%/q99.5% winsor/clip/center preprocessing family. EXP071 restricts preprocessing to GLOBAL and makes every held-out center donor-derived.

## Baselines and geometry

The canonical baseline is `C:\Users\Admin\Desktop\submission_geometry_research\gpt_pro_research_packet\06_ALIGNED_OOF.parquet` (`pred_exp037`, 770,616 rows). Existing SEQ-FRESH is `pred_fresh_contrast`. TEST alignment is `C:\Users\Admin\Desktop\submission_geometry_research\gpt_pro_research_packet\07_ALIGNED_TEST.parquet` (250,000 rows). The existing TEST span is the deduplicated 65-source rank-57 basis in `submission_geometry/cache/Z.npz`, built by `submission_geometry/directions.py`.

## Hard-stop decision

Reconnaissance: **PASS**. Stable-hook parity is evaluated separately in `encoder_parity.json`; pilot execution remains blocked until that audit passes.
