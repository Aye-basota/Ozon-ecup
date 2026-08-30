# EXP-062 — PLATFORM-DETREND Hypothesis Card (frozen before results)

- **Date:** 2026-08-25
- **Author:** A1
- **Development reference:** exact `STRONGEST-CURRENT / exp_037`, wCV `1.7475098625`

## Hypothesis

A user's activity on a day has different meaning in a quiet versus platform-wide high-demand regime. Cutoff-safe user histories normalized by same-day platform Search/Cart/Order/GMV intensity should expose relative demand and explain champion residuals beyond the user's marginal window aggregates.

## Why genuinely new information

Canonical columns aggregate each user in isolation. The proposed representation introduces contemporaneous cross-user context from rows observed by the cutoff. It is not another transform of the 227 per-user totals: two identical personal histories on different platform-state days receive different values.

## Closest prior + critical difference

- `exp_057` matched training rows to the production distribution through target-free row weights; EXP-062 keeps training geometry fixed and changes each historical event's contemporaneous context.
- `exp_023` measured a specific 2025 holiday response; EXP-062 is a general daily market-state normalization evaluated on all canonical folds.
- ETX/TCN encode user event age but do not receive platform-wide daily intensity.

## Expected mechanism / segment

Raw recent activity may overstate intent when the whole platform is elevated and understate it on quiet days. Relative activity should matter most for recent Search/Cart users and should reduce the monotone platform-growth component of train-to-test shift.

## Cheapest falsification

For every canonical cutoff, compute daily platform factors using only rows and current-panel membership observable at `event_date <= cutoff`. Build fixed 30/90-day sums of user events divided by same-day platform per-active-user rates. The matched placebo jointly permutes daily factors within fixed 28-day calendar blocks before aggregation, preserving factor marginals and user event histories. Join both to exact exp_037 OOF and run the same cross-user/nested-scale residual preflight as EXP-061.

## Success gate

- Temporal, current-panel selection, alignment and factor-support audits pass.
- REAL improves at least `3/4` folds including `2025-10-16`.
- Nested two-sided `Delta wCV <= -0.0005` and REAL beats the date-shuffled placebo by at least `0.0003` wCV.
- Both user halves choose non-zero scale; the latest-fold gain is not confined to a tiny segment.

Passing means `CONTINUE` to a canonical model pilot plus explicit OOF-to-test support audit, not production acceptance.

## Kill gate

Any leakage/selection audit failure, latest-fold non-improvement, `Delta wCV > -0.0005`, REAL-placebo advantage smaller than `0.0003`, zero scales, or unexplained factor support outside the historical range => `REJECT`; do not rescue with factor/window/bin/model tuning in this exact family.
