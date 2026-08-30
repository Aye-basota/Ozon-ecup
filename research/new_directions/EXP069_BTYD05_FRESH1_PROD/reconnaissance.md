# EXP069 reconnaissance

Reconnaissance was completed before any EXP069 training or production code was written.

## Workspace topology

- Clean research workspace: `C:/Users/Admin/Desktop/e-cup-research-clean`.
- Historical source/artifact workspace: `C:/Users/Admin/Desktop/OZON-E-CUP`.
- Submission-geometry workspace: `C:/Users/Admin/Desktop/submission_geometry_research`.
- Canonical aligned banks: `gpt_pro_research_packet/06_ALIGNED_OOF.parquet` and `07_ALIGNED_TEST.parquet` in the geometry workspace.

The clean repository intentionally retains only the compact canonical EXP-037 OOF artifact. Historical checkpoints, component banks, old experiment code, and the 65-source TEST geometry cache remain in the two registered external workspaces.

## Exact historical lineage recovered

The exact saved historical OOF correction is recoverable from `artifacts/oof_FRESH_CONTRAST_MOE.npz:fresh_processed_nested`. Its implementation is `src/fresh_contrast.py` and its conditional-head implementation is `src/seq_cond.py`.

The recovered semantics are:

1. Raw conditional contrast `z_COND_FRESH - z_COND_CLEAN`.
2. Frozen `SEQ-D3A-BASE-S42-V*` encoder, trained on CLEAN only.
3. Positive-target conditional amount supervision; extensive probability is unchanged.
4. `splitmix64(user_id) & 1` user split and symmetric cross-fit assembly.
5. Donor-fold 0.5/99.5 percentile bounds.
6. Clip before gating/centering; selected historical variant is GLOBAL on every held-out fold.
7. Center after clipping; selected alpha is 1 on every held-out fold.
8. Historical productionized OOF vector uses no HIGH16 gate.

`src/seq.py` establishes the exact encoder family: a 365-day, 17-channel causal TCN. Stored channels are `present, cat, buy, ponly, searches, search_to_cart, search_to_ord, cat_to_cart, cat_to_ord, to_cart, to_ord, gmv_search, gmv_cat, gmv`; generated inputs are `avail, dow_sin, dow_cos`. The encoder is hidden 64, 8 dilated residual blocks, kernel 3, dropout 0.1, with pooled `[last, mean, max]` representation (192 dimensions). Historical training is AdamW, learning rate 0.003, weight decay 0.01, batch 1024, warmup 300, 4 epochs, seed 42, `aug=none`, `depth_aug=0`, workers 3. Feature scaling is frozen from data through 2025-07-31. TEST depth is clipped to 289 days.

The exact conditional head is `Linear(192,64) -> GELU -> Dropout(0.1) -> Linear(64,1)`. PyTorch default initialization is retained for the first linear layer; the final weight is zero-initialized and final bias is zero. AdamW uses learning rate 0.001, weight decay 0.01, betas `(0.9,0.98)`, batch 8192, four epochs, cosine decay with 200-step warmup, gradient clipping at 1.0. The historical matched-volume control samples with replacement from the earliest one-third of CLEAN positive cutoff slots using the head seed.

## EXP-037 and BTYD recovery

EXP-037 is reconstructed in `src/block4_saf.py` and in the aligned packet as the registered log-space blend of CAP, UNC, DIST, SEQ-AVG3, and ETX-AVG3. The canonical evaluator independently fits a fold-level log offset and computes 1:2:4:8 wCV.

Stable EXP-051 BTYD artifacts are complete. `artifacts/BTYD_STABLE_EXP051/oof_raw.npz:z_btyd` and `test_raw.npz:z_btyd` are the exact OOF/TEST direction. The fixed production endpoint is `0.95*z_EXP037 + 0.05*z_BTYD`; no public-LB quantity is needed or used.

## Production availability decision

The four historical fold encoders exist and match their registered hashes. The exact production checkpoint `artifacts/model_SEQ-D3A-BASE-S42-TEST.pt` does not exist, and EXP-040 did not save conditional-head weights. This is the exact missing element; no substitute checkpoint or guessed transformation is authorized.

The user-predeclared bridge is therefore required for TEST preprocessing. Its leave-one-fold-out emulation preserves historical calibrated wCV exactly and preserves all four fold signs. Per-fold RMS differences versus the saved centered correction are 0.031244, 0.00000186, 0.018630, and 0.011840; these differences are foldwise constants caused solely by donor-derived rather than held-out-derived centering and are absorbed exactly by the canonical fold log-offset evaluator. The bridge's predictive parity discrepancy is 0.0 wCV.

Because the exact TEST checkpoint is absent, the permitted next step is exactly one production encoder trained with the recovered BASE configuration, followed by the preregistered three-seed/two-side CLEAN, FRESH, and VOL conditional heads. No architecture or hyperparameter search is permitted.

## Geometry basis recovery

`submission_geometry/cache/Z.npz`, `submission_geometry/core.py`, and `directions.py` provide the exact 65-unique-vector geometry bank and rank-57 mean-metric orthonormal difference basis. This basis is used only for target-free TEST projection. Public scores and fitted geometry weights are excluded from EXP069 evaluation and selection.

## Initial parity facts

- Aligned OOF rows/unique keys: 770,616 / 770,616.
- Canonical fold sizes: 188,518; 191,025; 193,694; 197,379.
- EXP-037 component reconstruction maximum log error: `9.93e-8`.
- Saved FRESH aligned maximum log error: `4.10e-7`.
- BTYD05 reconstruction maximum log error: `1.14e-7`.
- EXP-037 wCV: `1.747509862493216` from the float32 aligned packet (registered canonical value `1.7475098625201952`).
- FRESH delta wCV: `-0.000224956127443`, 4/4; latest `-0.000253421885`.
- BTYD05 delta wCV: `-0.000320983015379`, 4/4.
- Fixed BTYD05_FRESH1 delta wCV: `-0.000466939738222`, 4/4.

These values pass the 2e-5 parity gate. TEST production is not selected or tuned with public LB.
