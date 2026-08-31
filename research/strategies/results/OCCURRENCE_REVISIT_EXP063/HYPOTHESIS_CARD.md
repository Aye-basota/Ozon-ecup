# EXP-063 — OCCURRENCE-REVISIT Hypothesis Card (frozen before results)

- **Date:** 2026-08-25
- **Author:** A1
- **Development reference:** exact `STRONGEST-CURRENT / exp_037`, wCV `1.7475098625`

## Hypothesis

The existing two-part `S1-E11` prediction (purchase occurrence probability times conditional positive GMV) contains occurrence/intensity information that is complementary to the exact ETX+SEQ champion even though its earlier integration against a weaker mixture was below threshold. A small log-space weight should improve the exact four-fold reference more than the same-feature direct `S1-E10` control.

## Continuation tax / why this revisit is allowed

1. **Remaining uncertainty:** `S1-E11` was never evaluated as a member of exact exp_037; old `exp_016` used the weaker `S1-DIST-MIX` reference.
2. **Competing explanations:** (A) occurrence decomposition is genuinely complementary; (B) apparent teammate occurrence gains are weak-proxy deltas, production-regime effects, and/or public-LB calibration.
3. **One separating experiment:** artifact-only fixed-grid nested LOFO against exact exp_037, with `S1-E10` as a same-227-features direct control.
4. **Why highest EIV:** all OOF/test artifacts already exist, runtime is seconds, and the result resolves a provenance conflict before any retraining.
5. **Not disguised tuning:** no model retraining, new feature, leaderboard observation, segment gate, or adaptive fine grid. The alpha grid and gates below are frozen.

## Exact change and controls

For member `m`, form `z(alpha) = (1-alpha) * z_exp037 + alpha * z_m` in `log1p` space for fixed `alpha in {0, .025, .05, .075, .10, .15, .20}`. Evaluate `S1-E11` as REAL and `S1-E10` as the negative/direct control. CAP remains nonzero for every candidate. Exact `(cutoff,user_id)` alignment, `y`, finite values, fold sizes, and source hashes are mandatory audits.

For every held-out fold, choose alpha only on the other three folds using their corresponding canonical 1:2:4:8 weights; ties choose the smallest alpha. Report fixed curves and the assembled nested held-out result. Do not select from public LB.

## Success gate

- Alignment/target/finiteness/hash audits pass.
- Nested E11 `Delta wCV <= -0.0010`, improves at least `3/4` folds including `2025-10-16`, and every held-out selection is nonzero.
- E11 nested gain beats E10 control by at least `0.0005` wCV; selected alphas span at most two adjacent grid steps.
- If promoted to test integration, `Var(z_E11-z_strong)` test/OOF is within `0.6..1.2`, mean/quantile support is explained, level remains fixed at 2.3293, and submission schema audits pass.

Passing all OOF gates means `CONTINUE` to the already-available test-regime audit and canonical candidate build. Test support pass means `ACCEPT` as a production candidate; no leaderboard send.

## Development / kill gates

- Nested gain in `[-0.0010, -0.0005]` with latest-fold improvement is development-only: record `CONTINUE`, but it does not replace the production reference.
- `Delta wCV > -0.0005`, fewer than `3/4`, latest non-improvement, any zero held alpha, E11 not separating from E10, or any provenance audit failure => `REJECT` and close direct occurrence-member integration. Do not rescue with neighboring weights, occurrence-model retraining, segment gates, or public-LB calibration.
