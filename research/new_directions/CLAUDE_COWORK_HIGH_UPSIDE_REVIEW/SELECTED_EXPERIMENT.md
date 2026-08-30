# EXP072 — LWA-TAB: Late-Window Conditional-Amount Transfer (tabular)

**Status:** specified, not started. **Id check:** highest registered id is `EXP068` in
`02_EXPERIMENT_REGISTRY.csv` plus `EXP069/EXP070/EXP071` in `research/new_directions`; no `EXP072`
description exists anywhere in the three workspaces. `EXP072` is the next free id.

**Relationship to the originally proposed EXP072.** The proposal *"SEQ/ETX Temporal Distributional
Heads"* is **rejected as first priority** (see `ANALYSIS.md` §3: the residual-aligned component of a
distributional reformulation that the temporal models do not already contain measures `a = 0.00257`,
i.e. a best case of `−1.9e-6` wCV; and four of the five head-swap experiments on record lost to
their matched controls). The id is reused; the content is replaced.

---

## 1. One falsifiable hypothesis

> The legal late-window conditional-amount channel — the 13 `EXTRA` cutoffs `2025-10-22 … 2026-01-14`,
> which are the only training data outside the information set of all 65 geometry sources — is
> under-exploited by the current ~12k-parameter head on a frozen SEQ-D3A encoder. Replacing that head
> with a full-capacity **tabular** conditional-positive-amount model on the frozen 227-column
> `LnormNone` feature family, changing **nothing else**, yields a span-orthogonal nested
> `ΔwCV ≤ −0.0005` against `pred_exp037`, with `REAL − VOL ≤ −0.0003`, and retains at least half of
> that gain when every `EXTRA` cutoff whose 30-day target window overlaps the evaluated fold's
> target window is removed.

Falsified if any of: `REAL − VOL > −0.0001` on the pilot fold; the no-overlap arm retains `< 50 %`
of `REAL − VOL`; the centred correction's variance unexplained by the existing OOF direction span is
`< 0.5`; `|corr|` of the new correction with the existing EXP069 FRESH correction is `> 0.85`
(the direction is then a duplicate, not a basis vector).

The hypothesis is about **which rows the amount model is trained on**, not about LightGBM. LightGBM
is the instrument; the matched `VOL` control makes the training-row set the only difference between
arms.

---

## 2. Exact target

- Per training row `(T, user_id)` with `y30(T) = Σ gmv over (T, T+30]`:
  **`z_pos = log1p(y30)` restricted to rows with `y30 > 0`.** Conditional positive amount only.
- The extensive margin is **not** modelled or refreshed. It is taken unchanged from the existing
  frozen `p_dist` vector (see §6).
- Evaluation target: the canonical 30-day GMV in `06_ALIGNED_OOF.parquet:target`.

## 3. Exact baseline and metric

- Baseline: `pred_exp037` in
  `C:/Users/Admin/Desktop/submission_geometry_research/gpt_pro_research_packet/06_ALIGNED_OOF.parquet`
  (770,616 rows; folds `2025-09-04 / 2025-09-18 / 2025-10-02 / 2025-10-16` with
  `188,518 / 191,025 / 193,694 / 197,379` rows).
- `z_base = log1p(pred_exp037)`. Registered wCV `1.7475098625201952`; the float32 packet reproduces
  `1.747509867` (`4.6e-8`), which is the parity tolerance to expect.
- Metric: per fold, `r = log1p(target) − z`, subtract the fold mean of `r` (the project's global log
  offset), RMSLE `= sqrt(mean((r − mean r)²))`; aggregate with weights `1:2:4:8` oldest→newest.
- **Never** build a residual against `1.6466079084`. Never project geometry weights onto OOF rows.

## 4. Exact train construction

For each evaluated fold `V ∈ {2025-09-04, 2025-09-18, 2025-10-02, 2025-10-16}`:

- **CLEAN grid** — every step-7 cutoff `T` from `2025-04-03` with `T + 30 days ≤ V`
  (18 / 20 / 22 / 24 cutoffs for folds 1→4, matching the historical S04/EXP-032 convention).
  Rows: eligible panel at `T` (3 blocks × 30 days), restricted to `y30(T) > 0`.
- **EXTRA set** — the 13 preregistered cutoffs `2025-10-22, 10-29, 11-05, 11-12, 11-19, 11-26,
  12-03, 12-10, 12-17, 12-24, 12-31, 2026-01-07, 2026-01-14`. Rows: eligible panel at `T`,
  `y30(T) > 0` **only**. EXTRA never contributes non-positive rows, never contributes to eligibility,
  the extensive probability, feature scaling, EXP-037 components or validation labels.
- **NOOV subset** — EXTRA restricted to cutoffs with `(T, T+30] ∩ (V, V+30] = ∅`. Concretely:
  fold `2025-09-04` and `2025-09-18` drop nothing (13 kept); fold `2025-10-02` drops `2025-10-22`
  (12 kept); fold `2025-10-16` drops `2025-10-22, 10-29, 11-05, 11-12` (9 kept).
- **User cross-fit** — `side = splitmix64(user_id) & 1`. Every arm is trained twice: the model
  trained on side `s` produces predictions only for users with side `1 − s`. Both sides are assembled
  into a single full-coverage OOF vector. This is the registered safeguard that makes it impossible
  for a user's own future target to enter their own prediction, and it is what legitimises using
  EXTRA cutoffs whose target windows overlap the fold's.

## 5. Exact features / representation

- `data/processed/feat_{YYYYMMDD}_LnormNone.parquet` — the frozen normalised-long-window family
  (`normalize_long: true`, `history_days: null`), 227 columns, the S1-E10 representation.
  **Verified present for all 13 EXTRA cutoffs and for the TEST cutoff `20260213`**, plus the full
  CLEAN grid (100 cutoffs cached in total).
- Panel caches `data/processed/panel_{YYYYMMDD}_b3.parquet` exist through `20260213`.
- **No new feature engineering.** No feature is added, removed, transformed or rescaled. If a cache
  is missing it is rebuilt only through `src.features.build_features(cutoff_date)`.

## 6. Exact model and arms

Four arms, all identical except for their training rows:

| arm | training rows |
|---|---|
| `CLEAN` | CLEAN positive rows only |
| `FRESH` | CLEAN ∪ EXTRA positive rows |
| `FRESH_NOOV` | CLEAN ∪ NOOV positive rows |
| `VOL` (matched control) | CLEAN ∪ an equal number of positive rows (equal to the EXTRA contribution) resampled **with replacement** from the earliest one-third of CLEAN positive cutoff slots, RNG seed 42 |

- Estimator: LightGBM regression on `z_pos`, objective `regression`, metric `rmse`.
- Hyperparameters: reuse the historical positive-part configuration of the `S1-E11` two-part model
  verbatim if it is recoverable from the old repo; otherwise the exact
  `config/competition.yaml:lightgbm` block (`learning_rate 0.05`, `num_leaves 127`,
  `min_data_in_leaf 200`, `feature_fraction 0.7`, `bagging_fraction 0.8`, `bagging_freq 1`,
  `lambda_l2 5.0`, `max_bin 63`, `force_row_wise true`) with `num_boost_round = 400`,
  no early stopping, `seed = 42`, `deterministic = true`. **Whichever is chosen is used unchanged by
  all four arms and both sides.** No tuning, no sweeps, no per-fold selection.
- Correction, built exactly as EXP-032B/EXP-040/EXP-069 do:
  `d_raw = μ_arm − μ_CLEAN` (opposite-side predictions), then the historical processing chain in
  `OZON-E-CUP/src/fresh_contrast.py` — winsorise to the **donor-fold** `q0.005 / q0.995`, `GLOBAL`
  variant, centre on the donor-fold winsorised mean, and apply the extensive probability multiplier.
- **`p_dist` is reused unchanged** from EXP069 so the multiplier is byte-identical and `μ` is the
  only thing that changes: OOF from the same S1-DIST source EXP069 used, TEST from
  `EXP069_BTYD05_FRESH1_PROD/fresh_conditional_TEST.parquet:p_dist`.
- Candidate endpoint: `z_cand = z_base + α · d_processed`.

## 7. Matched control and required comparisons

Minimum comparison set, all on canonical folds with the canonical evaluator:

1. `pred_exp037` baseline.
2. `VOL` matched control run through the identical pipeline (`REAL − VOL` is the primary evidence).
3. `FRESH` — the new candidate.
4. `FRESH_NOOV` — calendar-overlap control.
5. Replacement comparison: `LWA` versus the existing EXP069 `FRESH` correction (which is better
   alone, on the same folds, same evaluator).
6. Add-one / nested 2-D blend: `z_base + α₁·d_FRESH_EXP069 + α₂·d_LWA` with **nested per-fold**
   selection of `(α₁, α₂)` on the three donor folds over `{0, 0.25, 0.5, 0.75, 1.0}²`.

## 8. Leakage safeguards (explicit, all must be asserted in code)

- Feature frames come only from `build_features(T)` / its caches, which read `event_date ≤ T`.
- `T + 30 ≤ V` for every CLEAN training cutoff.
- EXTRA rows: positive-target only, opposite user side only, and they update **only** the amount
  model. Assert row counts and that no EXTRA row shares a `user_id` side with its evaluation rows.
- No fold's own target participates in its own winsor bounds or centre — both are donor-derived.
- `p_dist` is frozen and identical across arms.
- No public-LB quantity enters any label, weight, bound, level or selection.
- Assert `770,616` rows, `0` duplicate `(fold,user_id)`, exact fold sizes, and EXP-037 reconstruction
  before anything is trained.

## 9. Cheap pilot and the exact continuation gate

**Pilot:** fold `2025-10-16` only, seed 42, four arms × two user sides = 8 LightGBM fits on cached
features. Expected ~35–50 minutes CPU. No TEST inference. No four-fold run.

Continue **only if all four hold** on the pilot fold, using the canonical single-fold evaluator with
the fold's own global log offset:

- **G1** `wCV_fold(FRESH) − wCV_fold(VOL) ≤ −0.00010`
- **G2** `wCV_fold(FRESH) − wCV_fold(EXP037) < 0` at fixed `α = 1`
- **G3** `(wCV_fold(FRESH_NOOV) − wCV_fold(VOL)) ≤ 0.50 · (wCV_fold(FRESH) − wCV_fold(VOL))`
- **G4** fraction of the centred correction's variance **not** explained by a least-squares
  projection on the 13 other aligned OOF difference directions `≥ 0.50`, **and**
  `|corr(d_LWA, d_FRESH_EXP069)| ≤ 0.85` on the pilot fold.

If any gate fails: **STOP**, write the report with verdict `REJECT`, record that the late-window
amount channel does not scale, and produce no TEST vector.

## 10. Full validation (only after pilot PASS)

- All four canonical folds, four arms, two sides (32 fits). Runtime budget ~2.0–2.5 h CPU.
- Honest nested LOFO: for each held-out fold select `α` on the other three folds
  (weights `1:2:4:8` renormalised over the donors) from `{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}`;
  report per-fold held-out deltas, the nested wCV delta, and the fixed-`α=1` delta separately.
- Span-orthogonalised variant: project the correction out of the span of the other aligned OOF
  difference directions (donor-fold fitted, applied to the held-out fold) and repeat the nested
  selection. **The orthogonal number is the one the gates in §12 are read against**, because only the
  out-of-span part can move the incumbent.
- User-cluster bootstrap (≥1,000 resamples over `user_id`) for the nested delta; report the 95 %
  interval and `P(Δ < 0)`.
- Both `splitmix64(user_id) & 1` halves reported separately; both must be negative.
- Seed robustness: repeat the pilot fold with seeds 43 and 44 for the `FRESH` and `VOL` arms only,
  and report the seed spread against the effect size.

## 11. Diversity and span analysis (mandatory outputs)

- Prediction correlation, log-prediction correlation and residual correlation of the candidate
  against `pred_exp037` and all 13 other aligned OOF sources.
- Correction correlation `corr(d_LWA, d_i)` for every aligned direction, and specifically against
  `d_FRESH_EXP069` and `d_BTYD`.
- Unexplained variance of `d_LWA` after least-squares projection on the other directions
  (donor-fitted, held-out evaluated).
- If and only if OOF PASSes: TEST span distance against the rank-57/58 basis rebuilt from
  `submission_geometry/cache/Z.npz` (65 unique vectors after dropping
  `C_lgbm_exp015_regen.csv` and `submission_BTYD05.csv`, reference `last (1).csv`, mean-N metric,
  double re-orthogonalisation). Report orthogonal RMS, orthogonal fraction, nearest source, and
  whether numerical rank increases. The basis is used **only** for target-free projection; geometry
  weights and public scores are never touched.

## 12. Success gates

Read against the **span-orthogonal nested** `ΔwCV`, on canonical four-fold wCV vs `pred_exp037`.

- **PASS_BIG** — `≤ −0.0015` (threshold derived in `ANALYSIS.md` §2.2: public `−0.001` needs a TEST
  unit alignment of `0.0574`, which at the project's confirmed gain transfer `0.55–0.67` means
  `−0.0015 … −0.0018` on OOF), **and** ≥3/4 folds improved including the latest, **and**
  `REAL − VOL ≤ −0.0008`, **and** bootstrap 95 % upper bound materially negative, **and** TEST
  orthogonal fraction `≥ 0.5`, **and** TEST correction magnitude within `±25 %` of the OOF magnitude.
- **PASS_BASIS** — `−0.0005 … −0.0015` with `REAL − VOL ≤ −0.0003`, ≥3/4 folds, both user halves
  negative, `|corr|` with the EXP069 FRESH correction `≤ 0.6`. Produce OOF **and** TEST vectors and
  register the direction as the second basis vector. **Do not present it as a standalone `0.001`
  solution.**
- **WEAK_SIGNAL** — `−0.0002 … −0.0005` with `REAL − VOL < 0` and ≥3/4 folds. Record, save the OOF
  vector, do **not** produce a TEST vector and do **not** deploy.
- **REJECT** — anything above `−0.0002`, or `REAL − VOL ≥ 0`, or the placebo wins on ≥2/4 folds, or
  G3/G4 fail on the full folds.

## 13. TEST production policy

Only after an OOF `PASS_BASIS` or better:

- Retrain the `CLEAN` and `FRESH` arms at the TEST cutoff `2026-02-13` with the **identical**
  estimator, identical hyperparameters, identical rounds, identical seed and the identical two-sided
  cross-fit assembly. **Magnitude parity is a hard requirement**: the TEST correction RMS must be
  within `±25 %` of the OOF correction RMS, and the run must report the ratio. EXP069's `0.405×` loss
  is the specific failure this design exists to avoid; if parity fails, the cause must be found and
  fixed, not absorbed into `α`.
- Winsor bounds and centre are the frozen full-OOF values; no TEST centring, no TEST variance
  matching, no TEST target calibration.
- `p_dist` for TEST is the saved EXP069 vector, unchanged.
- Fix `α` from OOF evidence **before** writing any file. No public score is consulted for `α`, for
  the level, or for anything else. No submission is uploaded by the experiment.

## 14. Runtime budget (hard)

| stage | budget |
|---|---|
| reconnaissance + parity replication of EXP069's correction chain | 20 min |
| pilot (fold `2025-10-16`, 4 arms × 2 sides) | 50 min |
| full four folds + controls + seeds | 2 h 30 min |
| OOF analysis, nested selection, bootstrap, diversity | 30 min |
| TEST production (only if PASS) | 30 min |
| **total** | **≤ 4 h 30 min, CPU only, ≤ 3 GB new persistent disk** |

## 15. Artifacts

`research/new_directions/EXP072_LWA_TAB/`:
`reconnaissance.md`, `config.json`, `parity_exp069_replication.json`, `pilot_metrics.json`,
`fold_metrics.csv`, `real_vs_vol.csv`, `noov_control.csv`, `nested_selection.csv`,
`orthogonal_metrics.csv`, `bootstrap_metrics.csv`, `user_half_metrics.csv`, `diversity_oof.csv`,
`oof_projection_metrics.json`, `seed_robustness.csv`, `runtime_resources.json`,
`artifact_manifest.csv`, `checksums.sha256`, `report.md`, and — only on PASS —
`lwa_tab_OOF.parquet`, `lwa_tab_TEST.parquet`, `lwa_tab_TEST.csv`, `test_span_projection.json`,
`production_regime.json`.

## 16. Honest expectation

Central estimate: span-orthogonal nested `ΔwCV ≈ −0.0004`, range `−0.0002 … −0.0008`.
`P(≤ −0.0005)` low-medium; `P(≤ −0.001)` low; `P(≤ −0.0015)` very low.
This experiment is selected because it is the highest-EV *basis* experiment available, not because
it is likely to deliver `0.001` public on its own. Its second value — closing the last open family
for ~45 minutes of CPU if it fails — is real and is part of the reason it is ranked first.

## 17. If it passes: how the two directions are combined honestly

1. Fit nothing on TEST. On canonical OOF, select `(α₁, α₂)` for
   `z = z_exp037 + α₁·d_FRESH + α₂·d_LWA` by **nested** per-fold selection on the three donor folds.
2. Require the 2-D nested delta to beat both 1-D nested deltas on ≥3/4 held-out folds.
3. Orthogonalise the combined correction against the 65-source span (`Z.npz`, mean-N metric, applied
   twice), verify the level shift is `≈0`, then and only then add it to the incumbent TEST geometry
   with an `α` fixed from OOF.
4. Per `ANALYSIS.md` §8, a two-direction basis at `a₁ = 0.0237`, `a₂ ≈ 0.045`, `ρ ≈ 0.5` implies a
   combined nested `≈ −0.00057` and public `≈ −0.00033`. Reaching `≈ −0.001` public would require
   **both** directions near `a ≈ 0.05` with `ρ ≤ 0.2`. Say this out loud in the report rather than
   letting the number drift upward.
