# Next submission after EXP069

# Final verdict

A new submission was created.

- **File:** `submissions/SUBMIT_NEXT_AFTER_EXP069.csv` (repo root: `C:/Users/Admin/Desktop/e-cup-research-clean`)
- **SHA256:** `714ec6eb5873389cc0586e4883d206f2a0ade3b4c41ff7e76eff56d1ef783d75`
- **Builder:** `submission_geometry/build_NEXT_AFTER_EXP069.py`
- **Status:** `UPLOAD_THIS` — one candidate, ready. It has **not** been uploaded.

Deliberate placement note: the CSV is written to the *clean repo's* `submissions/`, not to
the geometry workspace's `submissions/`. The geometry loader globs every `*.csv` in that
folder and looks each one up in the score registry, so dropping an unscored file there
would corrupt the 65-source bank on the next rebuild.

---

# Experiment review

| Experiment | Independent verdict | What was learned | Can it enter this submission? | Next relevance |
|---|---|---|---|---|
| **EXP071 ETX-FRESH** | **CONFIRMED_REJECT** — every pilot metric reproduces to `0.000e+00` | REAL is worse than EXP-037 (`+0.000110`), worse than its matched-volume placebo (`+0.000193`), and the ETX-orthogonal part makes SEQ-FRESH worse (`+0.000213`); both user halves agree in the wrong direction (`+0.000167`, `+0.000223`). The rejected object is **only** FRESH conditional heads on this frozen ETX-01 representation — not ETX representations (ETX-AVG3 still carries 0.225 inside EXP-037) and not FRESH supervision (EXP069 passes with it on the SEQ-D3A embedding). ETX's correction correlates only `0.240` with the working SEQ-FRESH correction and `0.004` with the base residual: a different, empty direction. | No | Low. Do not continue this pairing. |
| **EXP070 COUNT-VALUE MoE** | **COMPUTE_INCOMPLETE**, not a true reject | Everything saved reproduces exactly, but `2025-10-02` was never trained, so canonical four-fold wCV and honest LOFO do not exist. The pilot pass was marginal (`−0.000115` vs a `−0.00010` gate) and **the placebo beats the real model on 2 of 3 completed folds** (`+0.000180`, `+0.000086`, `−0.000116`) — the whole `−0.0000526` 1:2:8 advantage is the single weight-8 latest fold. Bootstrap of that 1:2:8 delta: 95% `[−0.000156, −0.0000073]`, `P(Δ<0)=0.983`. | No — no canonical wCV, no LOFO, no TEST vector, no production audit | **High**, as the cheapest decisive test. See "Next step". |
| **EXP069 BTYD05_FRESH1** | **PASS_TYPE_A independently confirmed** | 40/40 lineage tests + 116/116 component tests + production validator all pass. Every headline OOF number reproduces to ~`1e-15`; `−0.000466940` wCV, 4/4 folds, latest `−0.000397183`, REAL−VOL `−0.000232049`, both halves negative, bootstrap 95% `[−0.000585341, −0.000345622]`. TEST span numbers reproduce **exactly**: `0.010679245743308086` RMS, orthogonal fraction `0.12719419305538987`, rank `57 → 58`. | **Yes — its orthogonal FRESH component only** | The FRESH direction is the first genuinely out-of-span direction the geometry line has had. |

Decomposition that decides *what* is actually new (`exp069_component_decomposition.csv`):

| component | RMS | in-span RMS | out-of-span RMS | out-of-span fraction |
|---|---|---|---|---|
| `d_btyd` = 0.05·(z_btyd − z_base) | 0.027126 | 0.027125 | **0.000268** | **0.99 %** |
| `d_fresh` (saved correction) | 0.012200 | 0.005905 | **0.010675** | **87.5 %** |
| `d_combined` | 0.032424 | 0.030614 | 0.010679 | 32.9 % |

BTYD is essentially fully inside the existing span — `candidate_B_BTYD05_HEDGE.csv` is one
of the 65 sources, and on canonical OOF the BTYD correction's orthogonal component against
the eligible OOF basis is **exactly zero**. The entire rank increase and virtually the
entire orthogonal component are FRESH: `cos(d_fresh_⊥, d_combined_⊥) = 0.99968`, and the
two differ by only `0.000268` RMS (2.5 %). The S1-DIST reconstruction limitation is a
*multiplicative* per-user factor (`p_dist`) on the raw contrast, so it touches the new
orthogonal component too, not only the parallel part — carried into the uncertainty budget.

---

# Chosen submission

**Deployment anchor:** the current public incumbent `SUBMIT_NEXT_BEST.csv`
(`1.6466079084`, SHA `95f3fa98…`), used only as a TEST anchor.

**Correction:** the affine-span-orthogonal component of the EXP069 **FRESH-only**
correction (Candidate A). Candidate B (combined-orthogonal) is numerically the same
direction (cos `0.99968`); per the registered tie-break the simpler FRESH-only formulation
was chosen and the equivalence is documented above. Candidate C (full combined correction)
was rejected: it re-injects the in-span BTYD component — the exact direction whose
historical OOF→LB transfer *inverted* (`−0.000321` OOF → `+0.000274` public) — and it shifts
the global log level by `−0.00962` at α = 1 while the level of this family is already
essentially optimal on TEST (see below).

**Projection:** linear projection of the correction *direction* onto the row space of `Phi`
in the mean-N inner product, where `Phi` is the rank-57 orthonormal basis of
`span{z_i − z_ref}` over the 65 unique scored submissions (`Z.npz`, hash verified,
`tol = 1e-12`), applied twice for re-orthogonalisation. The constant vector lies inside that
space (residual RMS `7.89e-09`), so the residual is orthogonal to the intercept direction
and is level-neutral by construction: the realised mean log shift is `+9.88e-07`.

**Alpha:** `0.50` (frozen before the file was written; never touched by any public score).

**Formula:**

```
z_inc  = log1p(SUBMIT_NEXT_BEST.predict)
d      = fresh_conditional_TEST.correction              # saved standalone vector
d_perp = d - (Phi @ d / N) @ Phi                        # applied twice
z      = z_inc + 0.50 * d_perp
predict = max(expm1(z), 0)
```

**Why this avoids BTYD double-counting.** BTYD05 is already a geometry source and the
geometry's public-LB fit has already assigned it a coefficient. Deploying the combined
EXP069 vector would add a second, unpriced copy of it. The projection removes the in-span
part exactly; what survives of BTYD is `0.000268` RMS out of `0.010675`, and on canonical
OOF the BTYD orthogonal component is identically zero.

**Why the absolute EXP069 prediction was not blended in.** `z_exp069 = 0.95·z_base + 0.05·z_btyd + d`
is anchored on `pred_exp037_rebuilt`, which differs from the exact EXP-037 formula by a
level shift of `−0.135341` (post-mean-removal sd only `0.000830`). Blending the absolute
vector with the incumbent would (a) dilute a geometry solution that is already optimised
inside its own span with a much weaker standalone model, (b) import that baseline level
shift, and (c) re-import BTYD. Using only the additive correction sidesteps all three: `d`
is a standalone vector that never touches the baseline, so the reconstruction drift
provably cannot enter the deployed step.

**Why deploying on the incumbent is legitimate despite having no incumbent OOF.** Both
`z_inc` and `z_exp037_test` lie in the 65-source affine span, and `d_perp` is orthogonal to
it. The first-order gain `mean(d_perp·(t − z))` is therefore the same whether the anchor is
EXP-037 or the incumbent. Measured directly: `mean(d_perp·(z_inc − z_exp037_test)) = 1.9e-07`
against a Cauchy–Schwarz bound of `8.8e-04`. The OOF evidence, collected against EXP-037,
transfers to the incumbent anchor without ever needing an incumbent OOF.

**Realised step:** RMS `0.00533754`, max `|Δ log|` `0.0713`, mean shift `+9.88e-07`,
correlation with the incumbent `0.9999945`, zero predictions `965 → 319`, max prediction
`3420.75`, no NaN/inf/negatives.

---

# Expected leaderboard score

Exact local model (validated below): for an added step `D`,
`ΔRMSLE ≈ [−2·A + Q] / (2·S₀)` with `A = mean(residual·D)` and `Q = mean(D²)`.
`Q` is exactly computable; only `A` is unknown, and `A = τ · a_oof · rms(D)` where
`a_oof = 0.024628` is the per-unit alignment measured honestly on canonical OOF against
`pred_exp037` (bootstrap sd `0.002203`), and `τ` is the OOF→TEST transfer coefficient.

*Validation of the model:* `submission_LEVEL_MINUS_006.csv` is a pure level shift, so its
whole public loss should equal its own second-order term. Observed `+0.00108317`; predicted
`Q/(2·S) = +0.00108713`. Agreement to `4e-6` — and it also proves this family's TEST level
is already essentially optimal, which is why a level-neutral step was mandatory.

At α = 0.50: `Q = 2.849e-05`, second-order floor `+8.651e-06`, public 20 %-sampling noise sd
`2.135e-05`.

| | value |
|---|---|
| Current public LB | **1.6466079084** |
| Expected gain | **−7.4e-06** (i.e. a small improvement) |
| **Estimated public LB** | **≈ 1.646600** |
| 50 % interval | `[1.646579, 1.646622]` |
| 80 % interval | `[1.646559, 1.646642]` |
| P(beats 1.6466079084) | **0.59** |
| P(improvement ≥ 0.0001) | 0.003 |
| P(regression ≥ 0.0001) | **0.0006** |

Scenario table (deterministic, no sampling noise):

| τ | −0.344 (historical OLS) | 0 | 0.20 (prior mean) | 0.356 (JS shrinkage) | 1.0 (full transfer) |
|---|---|---|---|---|---|
| ΔLB at α=0.50 | +3.61e-05 | +8.65e-06 | −7.32e-06 | −1.98e-05 | −7.12e-05 |

Break-even τ is `0.108` at α = 0.50 (`0.054 / 0.163 / 0.217` at α = `0.25 / 0.75 / 1.00`).

**Why α = 0.50, from three independent channels** (`score_estimate.json`,
`historical_transfer.csv`, `external_direction_backtest.csv`, `basis_stability.csv`):

1. **Honest OOF nested evidence** — the orthogonal FRESH direction on canonical OOF gives
   nested alphas `[1.0, 1.0, 1.0, 0.75]`, nested Δ wCV `−0.000162`, fixed-α=1 Δ `−0.000173`,
   3/4 folds improved, bootstrap 95 % `[−0.000240, −0.000113]`, `P(Δ<0)=1.00`, both user
   halves negative (`−0.000177`, `−0.000168`). Its VOL placebo run through the *same*
   pipeline is `+0.0000176` (harmful), so REAL − placebo on the orthogonal component is
   `−0.000190`. Fold deltas improve monotonically with recency:
   `+1.19e-05 → −1.02e-05 → −1.27e-04 → −2.59e-04`. Basis sensitivity: `−0.000157`
   (3-source basis) to `−0.000182` (14-source basis), perp vectors correlated ≥ `0.986`.
   Because the TEST correction is only `0.405×` the OOF magnitude, this channel argues for
   the cap, α = 1.0.
2. **Historical CV→LB transfer** — six scored submissions are *exact* affine combinations of
   components that have canonical OOF, so their fold-safe OOF wCV can be computed and paired
   with a real public score. All five non-trivial ones have **negative** unit-alignment
   transfer: `−0.134, −0.310, −0.033, −6.04, −0.676`; median `−0.222`, OLS-through-origin
   `−0.344`, leave-one-case-out `[−0.52, −0.25]`. Worst case: BTYD05 went `−0.000321` OOF →
   `+0.000274` public. **Caveat that must not be swept away:** the anchor
   (STRONGEST_CURRENT) was itself plausibly chosen on public LB, so all five rivals inherit
   a winner's-curse penalty of roughly one public-noise σ (`~1e-04`); correcting for that
   leaves 2 helping and 2 harming, i.e. τ ≈ 0 rather than strongly negative. Either way this
   channel argues for α ≈ 0.
3. **External-direction geometry backtest** — the geometry's own leave-one-submission-out
   backtest was re-run unchanged. For 47 non-degenerate sources the realised out-of-span
   alignment on the real TEST target is indistinguishable from zero: mean `−0.00018`,
   median `+0.00052`, `z` mean `−0.03`, sd `1.25`; only 10/47 out-of-span residuals actually
   improved the public score. `var(z) = 1.55` implies an empirical James–Stein shrinkage of
   **0.356** for any external-direction gain estimate → α* `= 0.82`. *Underidentification,
   stated plainly:* the closer analogue — a historical direction with canonical OOF that is
   also out-of-span — **cannot be constructed**, because every source with canonical OOF is
   by construction inside the span of the OOF component basis (all five candidates returned
   an OOF orthogonal component of `~4e-15`). That channel is empty, and the uncertainty
   above is widened rather than narrowed to compensate.

Synthesis: the three channels imply α ≈ `1.00 / 0.00 / 0.82`; their average is `0.61` and
α = 0.50 is the adjacent grid point. Under the stated prior `τ ~ N(0.20, 0.30)` α = 0.50
also has the highest expected gain, and its worst historical-transfer case (`+3.6e-05`) is
about a third of one real leaderboard step. α = 0.25 has a higher P(beat) (`0.64`) but a
lower expected gain; α = 1.00 turns expected-positive only if τ > `0.217` and carries a
6.5 % chance of a ≥ 0.0001 regression. **Alpha was never selected from any public score of
this candidate.**

**Main uncertainty sources**, in order of contribution: (1) τ, the OOF→TEST alignment
transfer — the direct historical evidence is at best neutral and at worst negative, and the
folds end four months and one holiday season before the TEST cutoff; (2) the reconstructed
`p_dist` multiplier, which scales the new orthogonal component per user; (3) public 20 %
sampling noise (`2.1e-05` at α = 0.50); (4) OOF estimation noise on `a_oof` (sd `0.0022`);
(5) basis instability (14 %).

---

# Safety and reproducibility

- **OOF baseline rule.** Every OOF quantity uses `pred_exp037` from `06_ALIGNED_OOF.parquet`.
  The incumbent was used only as a TEST deployment anchor: no residual target was built
  against `1.6466079084`, no incumbent OOF was simulated, no TEST geometry weight was applied
  to OOF rows, and no public-LB equation produced any label.
- **No public-LB tuning.** The coefficient of EXP069, the correction scale, the
  preprocessing, the component decomposition, the candidate choice and the level shift were
  all fixed from OOF, geometry structure and pre-existing scored artifacts. No public score
  was invented for EXP069 and nothing was inserted into the geometry OLS. Geometry weights
  were not refit; only the fixed rank-57 basis was reused for projection.
- **No candidate zoo.** Three deterministic constructions were compared internally
  (`candidate_comparison.csv`), one CSV was produced, and no unrestricted sweep inside the
  existing 65-source span was run.
- **Nothing overwritten.** The incumbent, the previous incumbent, all original experiment
  artifacts, manifests and reports are byte-unchanged; `experiment_inventory.csv` re-hashes
  them. Output paths were checked for pre-existing files before writing.
- **Schema checks.** 250,000 rows; 250,000 unique `user_id`; columns exactly
  `user_id,predict`; byte-equal to the canonical `sample_submit.csv` order; 0 missing, 0
  duplicate, 0 NaN, 0 inf, 0 negative; min `0.0`, max `3420.7532`, mean `39.6926`.
- **Manifest checks.** `final_submission_manifest.json` records every input path with an
  independently recomputed SHA256, the exact formula, the projection definition, α, the
  output SHA256, row count, prediction statistics and the git working-tree state.
- **Rebuild status.** The builder was run twice; the two CSVs are **byte-identical**
  (`714ec6eb…` both times). A reconstruction written independently of the builder matches
  the shipped file to `5.0e-11` (the `%.10f` write format, same as the geometry's own
  builders).
- **Basis stability.** Leave-one-source-out over all 65 sources: drift median `1.22 %`, p90
  `6.84 %`, max `17.2 %`, min cos `0.9855`. Dropping each of the 10 largest-|weight| sources:
  median `2.55 %`, max `17.2 %`. Forty fixed-seed 80 % subsamples: drift median `13.95 %`,
  p90 `19.6 %`, cos min `0.9706`. For comparison the geometry's own in-span candidate moved
  `~56 %` of its step under leave-one-out and `~160 %` under 80 % subsampling: the out-of-span
  direction is an order of magnitude more stable, so no extra shrinkage was applied on this
  account. The direction definition was not changed and no favourable basis was selected.
- **Production limitation assessment.** Two carried limitations, both priced into the
  interval and neither invalidating the pass: the TEST extensive probability is a same-recipe
  CLEAN-only S1-DIST reconstruction (`reference_reproduced = false`, mean `|Δz| = 0.0445`)
  that multiplies the correction per user and therefore modulates the new orthogonal
  component; and the TEST correction is `0.405×` the OOF orthogonal magnitude — partly benign
  (TEST averages 2 donor sides × 3 seeds vs a single-seed cross-fit OOF) but not fully
  explained, which is itself a reason not to push α to the cap.
- **TEST regime.** `test_regime.json`: no winsor clipping active on TEST (clipped fraction
  `0.0`), `p_dist` mean `0.580` in `[0.0205, 0.9998]` with no zeros, no degenerate rows, and
  the level-neutrality of the deployed step verified at `+9.9e-07`.
- **Immutability re-check.** After all work, the 124 files recorded in
  `experiment_inventory.csv` — the three experiments' artifacts plus the incumbent, the
  previous incumbent, `last (1).csv`, `Z.npz`, both aligned banks and the canonical EXP-037
  OOF — were re-hashed: **0 drift**.
- **One disclosed side effect.** Running the geometry's existing `backtest.py` unchanged
  rewrote its own cache `submission_geometry/cache/loo_backtest.csv` in the geometry
  workspace. That file is a derived cache, not a manifest, report or submission, and it is
  byte-deterministic: re-running produces `1693515fd79b65a0994a83d1bd700d5e2b3d986093c995817c39145ade893b95`
  identically, at the same 23,991 bytes it had before. Nothing else outside this repo was
  written.
- **Compute.** No model was trained. Total additional compute was roughly one hour of CPU
  (test suites 40 s; OOF recomputation, bootstraps, projections, the LOO backtest and 115
  basis refits make up the rest), inside the two-hour budget, and no expensive sequence
  training was rerun.

---

# Next step

**Complete the missing `2025-10-02` fold of EXP070 under its frozen configuration — nothing else.**

- **Falsifiable hypothesis.** With all four folds trained under the unchanged EXP070 config,
  the DIST-replacement endpoint `z_exp037 + 0.25·(z_count − z_dist)` improves canonical
  four-fold wCV by at least `0.00010` **and** beats its matched shuffled control by at least
  `0.00010` on canonical wCV. If either fails, the count-regime mechanism is dead and no
  further count-MoE variant is authorised.
- **Why highest EV now.** It is the only open question on the board that a single fold
  settles. EXP071's mechanism is closed. EXP069 is already deployed here at a conservative
  scale and its next increment needs new evidence, not more geometry. EXP070 currently has a
  real but control-fragile signal (placebo wins 2 of 3 completed folds), and the missing fold
  is the exact observation that discriminates "genuine recency effect" from
  "one-fold fluke" — and it is also the fold that makes an honest LOFO possible at all.
- **Required inputs.** `run_experiment.py` unchanged; the three existing `_fold_*.npz` caches
  (`load_fold_cache` already supports resume); `_label_cache/label_20251002_b3.parquet`
  (already present); the frozen 227-column S1-E10 feature caches; `06_ALIGNED_OOF.parquet`.
- **Expected runtime.** ~40–50 minutes for the one fold (12 LightGBM fits: 2 multiclass ×
  300 rounds + 5 experts × 2 arms × 300 rounds, with the binned expert parent already shared
  between arms), versus ~116 min for all three. Disk: ~10 MB of new cache. No re-training of
  the completed folds.
- **Cheap pilot.** None needed and none appropriate — the "pilot" here *is* the single
  missing fold, and adding a pre-pilot would only re-spend the budget the fold itself costs.
  The only pre-flight check is a 5-minute assertion that the cached `2025-10-02` label and
  panel files reproduce their registered row counts and hashes before training starts.
- **Gates.** **PASS**: canonical four-fold wCV delta `≤ −0.00010`, REAL − SHUFFLED
  `≤ −0.00010` on canonical wCV, `≥ 3/4` folds improved, latest fold negative, and honest
  LOFO selects a non-zero β on `≥ 3/4` held-out folds. **WEAK SIGNAL**: wCV delta in
  `(−0.00010, 0]` with REAL − SHUFFLED `< 0` and `≥ 3/4` folds improved — record, do not
  deploy. **REJECT**: wCV delta `> 0`, or REAL − SHUFFLED `≥ 0`, or the placebo wins on
  `≥ 2/4` folds. No reduced rounds, no altered folds, no dropped placebo arm, and no verdict
  inferred from a partial fold set.
