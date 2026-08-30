# Autonomous task prompt for Claude Code (Opus 5) — EXP072 LWA-TAB

Paste this file as the whole task. It is self-contained: it assumes no access to the conversation
that produced it. Read it fully before writing any code.

---

## 0. Your role and the hard rules

You are a principal ML researcher running one pre-specified experiment in an existing competition
repository. You are **not** authorised to redesign the research. The hypothesis, target, arms,
controls, gates and thresholds below are fixed. You may fix bugs, adapt to what the filesystem
actually contains, and stop early — you may not invent a different experiment, add arms, tune
hyperparameters, or soften a gate.

Absolute prohibitions:

1. **Never** build any residual, label, weight, bound or level against the public score
   `1.6466079084` or against any public-LB quantity. The only honest OOF anchor is `pred_exp037`.
2. **Never** project public-LB geometry weights onto OOF rows and never reconstruct a "champion OOF".
3. **Never** upload a submission, and never tune anything on a public score.
4. **Never** write into the geometry workspace's `submissions/` folder (its loader globs `*.csv`
   there and an unscored file corrupts the 65-source bank).
5. Do not overwrite or modify any existing artifact, manifest, report or submission. Check for
   pre-existing files before writing. All new output goes under
   `research/new_directions/EXP072_LWA_TAB/`.
6. If a gate fails, **STOP** at that point, write the report with the honest verdict, and produce no
   TEST vector. A clean REJECT is a successful outcome of this task.

---

## 1. Context you need

**Workspaces**

- Clean research repo (your working root): `C:/Users/Admin/Desktop/e-cup-research-clean`
- Historical artifacts/code: `C:/Users/Admin/Desktop/OZON-E-CUP`
- Submission-geometry workspace: `C:/Users/Admin/Desktop/submission_geometry_research`
- Paths are configured in `config/paths.local.yaml`.

**Task.** Predict each eligible user's total GMV in the next 30 days. Metric RMSLE; all models and
blends operate in `z = log1p(prediction)`. Raw events: 30,631,006 daily rows, 250,000 users,
`2025-01-01 … 2026-02-13`. Test cutoff `2026-02-13`, target window `2026-02-14 … 2026-03-15`.
Eligibility: at least one observed day in each of the three latest 30-day blocks.

**Two baselines that must never be conflated.**

- *Public incumbent* `1.6466079084` — a TEST-only affine log-space combination of 65 unique submitted
  vectors fitted to public-LB score equations inside a rank-57/58 span
  (`submission_geometry/SUBMIT_NEXT_BEST.csv`). **It has no fold-safe OOF equivalent and is not an
  offline comparison baseline.** It appears in this task exactly once, as a target-free projection
  basis (§8.6), and never as a label source.
- *Canonical offline baseline* `EXP_037_STRONGEST_CURRENT` — the log-space blend
  `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-AVG3 + 0.225 ETX-AVG3`, wCV
  `1.7475098625201952` on 770,616 aligned OOF rows. This is the only residual anchor.

**Canonical validation.** Folds `2025-09-04 / 2025-09-18 / 2025-10-02 / 2025-10-16` with
`188,518 / 191,025 / 193,694 / 197,379` rows. wCV = per-fold RMSLE after subtracting the fold's own
global log offset, aggregated with weights `1:2:4:8` oldest→newest. A train cutoff `T` is legal for
fold `V` only when `T + 30 days ≤ V`. The clean corridor ends `2025-10-16`.

**Why this experiment was selected (short version of the review that produced this prompt).**

- Every one of the 65 submitted vectors that forms the incumbent's span was trained inside the clean
  corridor. Projecting each model family's TEST vector onto that span gives orthogonal fractions of
  `0.6 %` (DIST) to `6.7 %` (CAP), and exactly `0 %` for every submitted source. **Anything built by
  recombining existing sources is already priced by the geometry** — a fold-safe linear stack of the
  whole aligned OOF bank is worth `−0.000796` wCV and an in-fold *oracle* over the same bank only
  `−0.001081`, and all of it is in-span.
- Exactly one object in the project's history is genuinely out of span: the EXP069 FRESH conditional
  correction, `87.5 %` orthogonal, which moved the numerical rank `57 → 58`. Orthogonalising every
  aligned OOF direction against the other thirteen, FRESH is the **only** family with both a large
  unexplained fraction (`0.943`) and a residual-aligned orthogonal component
  (partial unit alignment `a = 0.02374`, best-case `ΔwCV −1.61e-4`). Next best is `SEQ-D3A` at
  `a = 0.0158`; `MHZ-FULL` is `0.0014`; CAP/UNC/DIST/SEQ/ETX/BTYD are ~0.
- FRESH's out-of-span-ness comes from its **data**, not its architecture: it is the only model ever
  trained on the 13 `EXTRA` cutoffs `2025-10-22 … 2026-01-14`, using positive-target
  conditional-amount rows only (the full target is illegal there — the 250k users were selected on
  guaranteed activity in `2025-11-16 … 2026-02-13`, so a full-target late model costs `+0.054 RMSLE`).
- Direct measurement on the raw panel over 41 weekly cutoffs shows the conditional amount regime
  really moves: `E[log1p y | y>0]` runs `4.264` (Apr) → `4.294` (Oct 16) → `4.410` (Nov 27) →
  `4.161` (Jan 8), with the cross-segment spread moving `0.13`. The level part is worthless (it is
  removed by the fold offset and by the geometry intercept); the **differential** part is the prize,
  and today it is extracted by a single `Linear(192,64)→GELU→Linear(64,1)` head (~12k parameters) on
  a frozen encoder trained for a different objective.
- So: point full tabular capacity at the same legal channel, change nothing else, and measure
  honestly.

**Threshold you are measuring against.** For a step `D`, `ΔRMSLE = (−2A + Q)/(2S)` with
`A = mean(r·D)`, `Q = mean(D²)`; with `a = A/rms(D)` the best achievable is `Δ = −a²/(2S)`. A public
`−0.001` therefore needs a TEST unit alignment `a = sqrt(2·1.6466·0.001) = 0.0574`. The project's
own confirmed OOF→LB **gain** transfer for large, genuinely new families is `0.55–0.67`
(`S1-DIST 0.64/0.71`, `EXP-037 0.564` recorded as the third independent confirmation of `≈0.57`),
which puts the required span-orthogonal nested `ΔwCV` at **`−0.0015 … −0.0018`**. FRESH today is
`a = 0.0237`, i.e. `−0.00016`. You are not expected to reach `−0.0015`; you are expected to find out
honestly whether this channel scales past `−0.0005`.

---

## 2. The one falsifiable hypothesis

> The legal late-window conditional-amount channel (13 EXTRA cutoffs `2025-10-22 … 2026-01-14`,
> positive-target rows only) is under-exploited by the current frozen-encoder head. Replacing that
> head with a full-capacity **tabular** conditional-positive-amount LightGBM on the frozen
> 227-column `LnormNone` features, changing nothing else, yields a **span-orthogonal nested
> `ΔwCV ≤ −0.0005`** against `pred_exp037`, with `REAL − VOL ≤ −0.0003`, and retains at least half of
> `REAL − VOL` when every EXTRA cutoff whose 30-day target window overlaps the evaluated fold's
> target window is removed.

The claim is about **which rows the amount model is trained on**. The `VOL` control makes the
training-row set the only difference between arms. "A new LightGBM" is not the mechanism and is not a
defence if the control wins.

---

## 3. Reconnaissance — do this before writing any experiment code

Write `research/new_directions/EXP072_LWA_TAB/reconnaissance.md` recording every finding, with
hashes and row counts. Do not train anything until it says PASS.

**3.1 Read these files (do not skim):**

- `README.md`, `config/competition.yaml`, `config/paths.local.yaml`
- `src/features/canonical.py` (`panel_users`, `target`, `build_features`), `src/data/loaders.py`,
  `src/validation/{folds,evaluate,workflow}.py`, `src/metrics/rmsle.py`, `src/models/tabular.py`
- `registry/experiments.csv`, `registry/models.csv`, `registry/submissions.csv`
- `research/new_directions/EXP069_BTYD05_FRESH1_PROD/` — **all** of `reconnaissance.md`, `report.md`,
  `config.json`, `run_oof_analysis.py`, `train_production_fresh.py`,
  `preprocessing_parameters.json`, `production_training_audit.json`
- `research/new_directions/EXP071_ETX_FRESH_CONTRAST/{report.md,reconnaissance.md}` — this is the
  failure you must not repeat
- `research/new_directions/NEXT_SUBMISSION_AFTER_EXP069/report.md` and
  `oof_component_metrics.csv`, `oof_nested_alpha.csv`, `exp069_component_decomposition.csv`
- In `C:/Users/Admin/Desktop/OZON-E-CUP`: `src/fresh_contrast.py`, `src/seq_cond.py` — these define
  the exact correction-processing chain, the 13 EXTRA cutoffs, the `splitmix64(user_id) & 1` split,
  the positive-only filter, the per-cutoff target centring and the equal-volume control.
- In `C:/Users/Admin/Desktop/submission_geometry_research`:
  `gpt_pro_research_packet/{06_ALIGNED_OOF.parquet,06_ALIGNED_OOF_COLUMNS.md,07_ALIGNED_TEST.parquet,07_ALIGNED_TEST_COLUMNS.md,15_VALIDATION_PROTOCOL.md,16_DO_NOT_REPEAT.md}`,
  `submission_geometry/{core.py,directions.py,geomlib.py}`, `submission_geometry/cache/Z.npz`,
  `submission_geometry/cache/Z_meta.json`.

**3.2 Confirm these prerequisites exist (they were verified to exist at review time):**

- `OZON-E-CUP/data/processed/feat_{YYYYMMDD}_LnormNone.parquet` for **all** of
  `20251022, 20251029, 20251105, 20251112, 20251119, 20251126, 20251203, 20251210, 20251217,
  20251224, 20251231, 20260107, 20260114` **and** `20260213`, plus the CLEAN step-7 grid from
  `20250403` to `20251016`. (100 cutoffs are cached in this family.)
- `OZON-E-CUP/data/processed/panel_{YYYYMMDD}_b3.parquet` through `20260213`.
- `EXP069_BTYD05_FRESH1_PROD/fresh_conditional_OOF.parquet` (770,616 × 15; columns include
  `raw_correction`, `z_cond_clean`, `z_cond_fresh`, `z_cond_vol`, `user_side`, `correction`) and
  `fresh_conditional_TEST.parquet` (250,000 × 13; includes `p_dist`).
- `gpt_pro_research_packet/06_ALIGNED_OOF.parquet` — 770,616 rows, columns `user_id, fold, target,
  pred_exp037, pred_cap, pred_unc, pred_dist, pred_etx_avg3, pred_seq_avg3, pred_seq_d3a_avg3,
  pred_ridge15, pred_hurdle_e11, pred_mhz_full, pred_holiday_yoy, pred_block4_saf,
  pred_fresh_contrast, pred_btyd, pred_btyd05`.

If a `feat_*` cache is missing, rebuild it **only** through `src.features.build_features(cutoff)` and
record that you did. Never hand-write features.

**3.3 Parity gate — replicate EXP069's own numbers before building anything new.**

Assert, and record the achieved errors:

- `06_ALIGNED_OOF.parquet` has 770,616 rows, 0 duplicate `(fold,user_id)`, fold sizes exactly
  `[188518, 191025, 193694, 197379]`, all `pred_*` finite and `≥ 0`.
- Your evaluator gives `wCV(pred_exp037) = 1.747509867 ± 5e-8` and per-fold
  `[1.7668834, 1.7605096, 1.7486292, 1.7412786]`.
- Add-one nested reproduction on the aligned bank (grid `α ∈ {0,0.05,…,2.0}`, held-out fold, donor
  weights renormalised): `pred_fresh_contrast → −0.000225 (4/4)`, `pred_btyd → −0.000269 (4/4)`,
  `pred_hurdle_e11 → +5e-6 (1/4)`. If you cannot reproduce these three, your evaluator is wrong —
  fix it before proceeding.
- **Reproduce EXP069's saved `correction` column** from its own saved `raw_correction`,
  `z_cond_*`, `user_side` and the processing chain in `src/fresh_contrast.py`, to `≤ 1e-9`. You must
  understand the exact order of winsorise / centre / extensive-probability multiplication *before*
  you build the new arm, because the new correction must use the identical chain with only `μ`
  changed. Record the recovered order in `parity_exp069_replication.json`.

Reconnaissance PASSes only if every item above passes. Otherwise stop and report.

---

## 4. Exact implementation

### 4.1 Target

Per training row `(T, user_id)`: `y30(T) = Σ gmv over (T, T+30]`; train on
**`z_pos = log1p(y30)` for rows with `y30 > 0` only**. The extensive margin is not modelled.

### 4.2 Cutoff construction, per evaluated fold `V`

- **CLEAN**: every step-7 cutoff from `2025-04-03` with `T + 30 ≤ V` (18/20/22/24 cutoffs for
  folds 1→4). Rows = eligible `b3` panel at `T`, filtered to `y30 > 0`.
- **EXTRA**: the 13 cutoffs `2025-10-22, 10-29, 11-05, 11-12, 11-19, 11-26, 12-03, 12-10, 12-17,
  12-24, 12-31, 2026-01-07, 2026-01-14`. Rows = eligible `b3` panel at `T`, `y30 > 0` only.
- **NOOV**: EXTRA restricted to `(T, T+30] ∩ (V, V+30] = ∅`. Folds 1 and 2 keep all 13; fold
  `2025-10-02` keeps 12 (drops `2025-10-22`); fold `2025-10-16` keeps 9 (drops
  `2025-10-22, 10-29, 11-05, 11-12`). Assert these counts.
- **User cross-fit**: `side = splitmix64(user_id) & 1` using the exact implementation in
  `src/seq_cond.py`. Every arm is trained twice; the model trained on side `s` predicts only for
  users with side `1 − s`; both halves are concatenated into one full-coverage vector.

### 4.3 Features

`data/processed/feat_{T}_LnormNone.parquet`, 227 columns, used unchanged. No feature is added,
removed, transformed or rescaled. Assert identical column order and dtype across all cutoffs.

### 4.4 Arms (four, identical except for training rows)

| arm | rows |
|---|---|
| `CLEAN` | CLEAN positives |
| `FRESH` | CLEAN ∪ EXTRA positives |
| `FRESH_NOOV` | CLEAN ∪ NOOV positives |
| `VOL` | CLEAN ∪ the same number of positive rows as EXTRA contributes, resampled **with replacement** from the earliest one-third of CLEAN positive cutoff slots, RNG seed 42 — use the exact rule already implemented in `src/seq_cond.py` |

### 4.5 Estimator

LightGBM regression on `z_pos`. Use the historical positive-part configuration of the `S1-E11`
two-part model verbatim if you can recover it from the old repo; record where you found it.
Otherwise use exactly `config/competition.yaml:lightgbm` — `objective regression`, `metric rmse`,
`learning_rate 0.05`, `num_leaves 127`, `min_data_in_leaf 200`, `feature_fraction 0.7`,
`bagging_fraction 0.8`, `bagging_freq 1`, `lambda_l2 5.0`, `max_bin 63`, `force_row_wise true` —
with `num_boost_round = 400`, no early stopping, `seed = 42`, `deterministic = true`.
**The same configuration is used by all four arms and both sides. No tuning. No sweeps. No per-fold
selection.** Record the chosen config in `config.json`.

### 4.6 Correction

`d_raw = μ_arm(opposite-side prediction) − μ_CLEAN(opposite-side prediction)`, then the historical
chain from `src/fresh_contrast.py`, GLOBAL variant: winsorise to the **donor-fold** `q0.005/q0.995`,
centre on the donor-fold winsorised mean, apply the extensive-probability multiplier.
**`p_dist` is reused unchanged from EXP069** — OOF from the same S1-DIST source EXP069 used, TEST
from `fresh_conditional_TEST.parquet:p_dist` — so `μ` is literally the only thing that changes
relative to EXP069. Candidate: `z_cand = z_base + α · d_processed`.

---

## 5. Cheap pilot and the exact continuation gate

**Pilot:** fold `2025-10-16` only, seed 42, four arms × two sides = 8 fits on cached features.
Budget ≤ 50 minutes CPU. No TEST inference, no other folds. Write `pilot_metrics.json`.

Continue only if **all four** hold, using the canonical single-fold evaluator (fold's own global log
offset) at fixed `α = 1`:

- **G1** `wCV(FRESH) − wCV(VOL) ≤ −0.00010`
- **G2** `wCV(FRESH) − wCV(EXP037) < 0`
- **G3** `(wCV(FRESH_NOOV) − wCV(VOL)) ≤ 0.50 × (wCV(FRESH) − wCV(VOL))`
- **G4** the centred correction's variance not explained by a least-squares projection on the 13
  other aligned OOF difference directions is `≥ 0.50`, **and**
  `|corr(d_LWA, d_FRESH_EXP069)| ≤ 0.85`

**If any gate fails: STOP.** Write `report.md` with verdict `REJECT`, state which gate failed and by
how much, record that the late-window amount channel does not scale in the tabular form, and produce
no TEST vector, no `_TEST.parquet`, no `_TEST.csv`. Do not "try one more variant".

---

## 6. Full validation — only after pilot PASS

- All four canonical folds × four arms × two sides (32 fits). Budget ≤ 2 h 30 min.
- **Honest nested LOFO**: for each held-out fold, select `α` on the other three folds (donor weights
  `1:2:4:8` renormalised) from `{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}`. Report per-fold held-out
  deltas, the nested wCV delta and the fixed-`α=1` delta separately. Never select `α` on the fold you
  score it on.
- **Span-orthogonalised variant**: project the correction out of the span of the other aligned OOF
  difference directions — coefficients fitted on donor folds, applied to the held-out fold — then
  repeat the nested selection. **The gates in §9 are read against this orthogonal number**, because
  only the out-of-span part can move the incumbent.
- **Bootstrap**: ≥1,000 resamples clustered on `user_id` for the nested delta; report the 95 %
  interval and `P(Δ < 0)`.
- **User halves**: report both `splitmix64 & 1` halves separately; both must be negative.
- **Seed robustness**: repeat the pilot fold with seeds 43 and 44 for `FRESH` and `VOL` only; report
  the seed spread against the effect size.

---

## 7. Leakage safety — assert all of these in code, and record the assertions

1. Every feature frame comes from `build_features(T)` or its cache, which reads `event_date ≤ T`.
2. `T + 30 ≤ V` for every CLEAN training cutoff of fold `V`.
3. EXTRA rows are positive-target only, from the opposite user side only, and update **only** the
   amount model — never eligibility, never `p_dist`, never feature scaling, never EXP-037 components,
   never validation labels.
4. No user appears in both the training set and the evaluation set of the same side.
5. Winsor bounds and centres are donor-derived; the held-out fold's own target never sets them.
6. `p_dist` is frozen and byte-identical across all arms.
7. No public-LB quantity enters any label, weight, bound, level or selection anywhere.
8. Canonical schema asserted before training: 770,616 rows, 0 duplicate keys, exact fold sizes,
   EXP-037 reconstruction error `≤ 1e-6`.

---

## 8. Required comparisons and diversity outputs

**8.1** `pred_exp037` baseline (four folds, canonical wCV).
**8.2** `VOL` matched control through the identical pipeline — `REAL − VOL` is the primary evidence.
**8.3** `FRESH` — the candidate.
**8.4** `FRESH_NOOV` — calendar-overlap control.
**8.5** Replacement comparison: `LWA` versus the existing EXP069 FRESH correction alone, same folds,
same evaluator; and the nested 2-D blend
`z_exp037 + α₁·d_FRESH_EXP069 + α₂·d_LWA` with nested per-fold selection of `(α₁, α₂)` over
`{0, 0.25, 0.5, 0.75, 1.0}²`.
**8.6** Diversity, all written to `diversity_oof.csv` / `oof_projection_metrics.json`:
prediction correlation, log-prediction correlation and residual correlation against `pred_exp037` and
all 13 other aligned sources; correction correlation `corr(d_LWA, d_i)` for every direction and
specifically against `d_FRESH_EXP069` and `d_BTYD`; unexplained variance of `d_LWA` after
donor-fitted projection on the other directions.
**8.7** TEST span distance — **only if OOF PASSes**. Rebuild the basis from
`submission_geometry/cache/Z.npz` (67 vectors; drop `C_lgbm_exp015_regen.csv` and
`submission_BTYD05.csv` → 65 unique; reference `last (1).csv`; mean-N inner product; eigen-decompose
`(Y Yᵀ)/N`; keep `λ > 1e-12 λ₀`; apply the projection twice for re-orthogonalisation). Verify the
constant vector lies inside the span (residual RMS `≈ 5e-12`). Report orthogonal RMS, orthogonal
fraction, nearest source and whether numerical rank increases past 58. **The basis is used only for
target-free projection; geometry weights and public scores are never touched.**

---

## 9. Success gates — read against the span-orthogonal nested `ΔwCV`

- **PASS_BIG** — `≤ −0.0015`, **and** ≥3/4 folds improved including the latest, **and**
  `REAL − VOL ≤ −0.0008`, **and** bootstrap 95 % upper bound materially negative, **and** TEST
  orthogonal fraction `≥ 0.5`, **and** TEST correction RMS within `±25 %` of the OOF correction RMS.
- **PASS_BASIS** — `−0.0005 … −0.0015`, `REAL − VOL ≤ −0.0003`, ≥3/4 folds, both user halves
  negative, `|corr(d_LWA, d_FRESH_EXP069)| ≤ 0.6`. Produce OOF and TEST vectors; register the
  direction as a **basis vector**. **Do not present it as a standalone `0.001` solution.**
- **WEAK_SIGNAL** — `−0.0002 … −0.0005`, `REAL − VOL < 0`, ≥3/4 folds. Save the OOF vector only; no
  TEST vector, no deployment.
- **REJECT** — above `−0.0002`, or `REAL − VOL ≥ 0`, or the placebo wins on ≥2/4 folds, or G3/G4 fail
  on the full folds.

---

## 10. Runtime budget (hard)

| stage | budget |
|---|---|
| reconnaissance + EXP069 parity replication | 20 min |
| pilot (fold `2025-10-16`, 4 arms × 2 sides) | 50 min |
| full four folds + controls + seeds | 2 h 30 min |
| OOF analysis, nested selection, bootstrap, diversity | 30 min |
| TEST production (only on PASS) | 30 min |
| **total** | **≤ 4 h 30 min, CPU only, ≤ 3 GB new persistent disk** |

Hardware is a 6c/12t Ryzen 5 7500F, ~31.6 GB RAM (17–20 GB typically free), RTX 4060 Ti 8 GB, and
constrained free disk. **This experiment needs no GPU.** Delete temporary caches at the end and
report `persistent_bytes`. If you exceed a stage budget by more than 50 %, stop and report rather
than continuing.

---

## 11. Production (TEST) — only after an honest OOF PASS_BASIS or better

- Retrain `CLEAN` and `FRESH` at the TEST cutoff `2026-02-13` with the **identical** estimator,
  hyperparameters, rounds, seed and two-sided cross-fit assembly used for OOF.
- **Magnitude parity is a hard requirement.** Report `rms(d_TEST) / rms(d_OOF)`; it must be within
  `±25 %` of 1. EXP069's TEST correction was only `0.405×` its OOF magnitude and that alone cost most
  of its deployed value. If parity fails, diagnose it — do not absorb it into `α`.
- Winsor bounds and centre = the frozen full-OOF values. No TEST centring, no TEST variance matching,
  no TEST target calibration.
- `p_dist` for TEST = the saved EXP069 vector, unchanged.
- Fix `α` from OOF evidence **before** writing any file. No public score is consulted for `α`, the
  level, or anything else.
- Save, with SHA256 recorded:
  - `lwa_tab_OOF.parquet` — `user_id, fold, target, z_base, correction, z_predict, predict, arm,
    user_side, raw_correction, vol_correction`
  - `lwa_tab_TEST.parquet` — `user_id, z_base, correction, z_predict, predict, p_dist,
    raw_correction`
  - `lwa_tab_TEST.csv` — exactly `user_id,predict`, 250,000 unique rows in the canonical
    `sample_submit.csv` order, finite, non-negative, no NaN/inf/duplicates.
- **Do not upload anything. Do not refit geometry weights. Do not write into the geometry
  workspace.**

---

## 12. Artifacts to write

Under `research/new_directions/EXP072_LWA_TAB/`:

`reconnaissance.md`, `config.json`, `parity_exp069_replication.json`, `pilot_metrics.json`,
`fold_metrics.csv`, `real_vs_vol.csv`, `noov_control.csv`, `nested_selection.csv`,
`orthogonal_metrics.csv`, `bootstrap_metrics.csv`, `user_half_metrics.csv`, `seed_robustness.csv`,
`diversity_oof.csv`, `oof_projection_metrics.json`, `runtime_resources.json`,
`artifact_manifest.csv` (every input path with an independently recomputed SHA256),
`checksums.sha256`, `report.md`; and only on PASS: `lwa_tab_OOF.parquet`, `lwa_tab_TEST.parquet`,
`lwa_tab_TEST.csv`, `test_span_projection.json`, `production_regime.json`.

Every script you write goes in the same folder so the run is reproducible from the repo alone.

---

## 13. Final report (`report.md`) — required contents

1. **Verdict**: one of `PASS_BIG` / `PASS_BASIS` / `WEAK_SIGNAL` / `REJECT`, with the gate arithmetic
   shown.
2. **Metrics**: canonical four-fold wCV and per-fold deltas for baseline, `CLEAN`, `FRESH`,
   `FRESH_NOOV`, `VOL`; the fixed-`α=1` delta, the nested delta, and the **span-orthogonal** nested
   delta; the selected `α` per held-out fold.
3. **Controls**: `REAL − VOL` per fold and aggregate; `FRESH_NOOV` retention ratio; both user halves;
   seed spread.
4. **Diversity**: correction correlations (especially with EXP069 FRESH), unexplained variance, and —
   on PASS — TEST orthogonal RMS/fraction, nearest source, rank change, and the OOF/TEST magnitude
   ratio.
5. **Runtime**: wall time per stage, peak RAM, persistent bytes written.
6. **Artifacts**: paths with SHA256.
7. **Recommendation**: one of `ADD_AS_BASIS_DIRECTION` / `RECORD_ONLY` / `DO_NOT_ADD` /
   `CLOSE_FAMILY`, plus one paragraph naming the single most informative next measurement. If the
   verdict is `PASS_BASIS`, state explicitly what the combined 2-D nested delta with EXP069 FRESH is
   and what public gain that implies at the project's `0.55–0.67` transfer — do not let the number
   drift upward.
8. **Limitations**: state every carried limitation plainly, including the reconstructed `p_dist`, the
   OOF/TEST magnitude ratio, the fact that the OOF folds end four months and one holiday season
   before the TEST cutoff, and — if `FRESH_NOOV` retention was between `0.5` and `0.8` — that part of
   the measured gain is calendar-contemporaneous and will not be available on TEST.

---

## 14. Start here

1. Create `research/new_directions/EXP072_LWA_TAB/`.
2. Do §3 reconnaissance in full and write `reconnaissance.md`. **The first substantive thing you do
   is reproduce EXP069's saved `correction` column from its own saved inputs to `≤ 1e-9`**, so that
   the new arm uses a chain you have verified rather than one you inferred.
3. Only then build the pilot. Only then look at G1–G4. Then stop, or continue, exactly as written.
