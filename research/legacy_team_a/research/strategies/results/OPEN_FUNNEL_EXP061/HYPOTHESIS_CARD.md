# EXP-061 — OPEN-FUNNEL Hypothesis Card (frozen before results)

- **Date:** 2026-08-25
- **Author:** A1
- **Development reference:** exact `STRONGEST-CURRENT / exp_037`, wCV `1.7475098625`

## Hypothesis

The amount and recency of historical Search/Cart activity strictly after a user's last order measure unresolved purchase intent. This state should predict the signed residual of `STRONGEST-CURRENT`, especially for never/rare/recently inactive buyers.

## Why genuinely new information

The 227 canonical columns contain window marginals and independent recencies but not the temporal stock of funnel events accumulated after the last order. The new object depends on event order across days; it cannot be reconstructed from total Search/Cart/Order counts alone.

## Closest prior + critical difference

- `exp_038` supervised an encoder with **future** Search/Cart labels; EXP-061 uses only past input state at the cutoff and does not add future supervision.
- `exp_052` decomposed monetary value by Search/Catalog channels; EXP-061 measures unresolved transition state, not GMV attribution.
- `exp_054` used fixed activity episodes/gaps; EXP-061 anchors the episode boundary to the user's last realized order and preserves channel direction.

## Expected mechanism / segment

Recent Cart/Search without a subsequent order should raise near-term conversion probability, while a long unresolved funnel with repeated non-conversion may lower it. The largest effect is expected among `w90_days_buy <= 1`, never-buy users, and `rec_buy > 30` with recent Search/Cart.

## Cheapest falsification

For each canonical fold, build features from rows with `event_date <= cutoff` only: Search/Cart days and volumes after the last order, their age span, and open-funnel flags. Join to the exact aligned OOF baseline. Produce cross-user out-of-fold residual corrections and an identical control with the new columns shuffled within fixed baseline-state strata. Select correction scale on the opposite user half; evaluate calibrated RMSLE on the recipient half.

## Success gate

- Leakage/integrity audit passes exactly on all four cutoffs.
- REAL improves at least `3/4` folds including `2025-10-16`.
- Nested two-sided `Delta wCV <= -0.0005` versus `STRONGEST-CURRENT` and REAL beats matched SHUFFLED by at least `0.0003` wCV.
- Both user recipient halves have non-zero selected scale and the expected low-buy segment does not reverse sign.

Passing this gate yields `CONTINUE` to a canonical feature/model pilot; it is not yet production acceptance.

## Kill gate

Any leakage failure, latest-fold non-improvement, `Delta wCV > -0.0005`, REAL-SHUFFLED advantage smaller than `0.0003`, zero scales on either recipient half, or a mechanism confined to an unstable tiny segment => `REJECT`; no threshold/scale/model sweep in this exact family.
