# FINGERPRINT_EXP058 — result summary

Verdict: **REJECT**.

- Integrity, exact-BASE replay, novelty, fixed-permutation marginals and test-metadata audits: **PASS**.
- 30 preregistered fingerprint fields → 15 after fixed novelty thresholds.
- Standalone calibrated RMSLE: PERM `1.745029349`, REAL `1.744899577`, REAL−PERM `−0.000129771`.
- Fixed strongest-slot RMSLE: PERM `1.741185002`, REAL `1.741256049`, REAL−PERM `+0.000071048`.
- Both fixed user-hash halves lose to PERM: `+0.000068771`, `+0.000073307`.
- REAL slot vs unchanged strongest: `−0.000022517` (noise); positive-segment RMSLE is worse by `+0.000206437`.
- Full folds, test inference, public LB and submission were not run.

Canonical machine-readable result: `summary.json`. Detailed experiment card: `experiments/exp_058_dataset_fingerprint.md`.
