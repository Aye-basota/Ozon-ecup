# EXP-056 — LATE-UNLABELED-ETX-ADAPT

## Endpoint

- Verdict: **REJECT**; `PROMOTE_TO_FULL_FOLDS=NO`.
- LATE−CONTROL standalone: raw **-0.000001**, calibrated **+0.000032**.
- LATE_SLOT−BASE_SLOT: raw **+0.000004**, calibrated **+0.000007**.
- User halves, standalone: **+0.000058 / +0.000005**; slot: **+0.000013 / +0.000002**.
- `corr(correction, CONTROL residual)` = **+0.001432**; `Var(z_late-z_control)` = **0.00017742**.
- Clean direct holdout MSE BASE/CONTROL/LATE = **3.138318 / 3.141582 / 3.141207**.
- Frozen/adapted embedding MMD is in `embedding_mmd.csv`; full domain audit is in `domain_shift_summary.json`.

## Contract

Both arms used the exact `ETX-01-S42-V0904` checkpoint, support depth **212**, Thursday query context, one materialized epoch, identical direct rows/LR/masks/RNG/head/optimizer initialization and deterministic eager bf16/TF32 CUDA. The SSL objective read only input histories and reconstructed the exact 14 normalized behavioral channels. Validation target was first read by this analysis stage.

No full folds, production/test prediction, submission or LB action was performed.
