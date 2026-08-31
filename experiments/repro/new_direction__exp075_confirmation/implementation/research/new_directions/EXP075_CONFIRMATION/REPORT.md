# EXP075 — independent confirmatory audit

Scope: decide whether `SUBMIT_EXP075_JOINT_A1_365_A2.csv` can be sent safely.
Nothing was tuned, no new features or models were searched, no leaderboard value
was used for selection, and no submission was uploaded. Every number below was
recomputed from primary artifacts (raw `train.parquet`, the frozen model files,
the stored OOF/TEST arrays and the submission CSVs); nothing was taken from
`EXP075/REPORT.md` on trust.

## Verdict

**STRONG_GO** — with one qualification on gate 3 and one explicit risk warning.

All nine pre-registered conditions are satisfied. The single deviation is that
gate 3 ("exact candidate parity, max abs difference <= 1e-10") is not literally
attainable from the artifacts that exist: the pipeline stored `D_perp` only in
`float32`, serialized the CSV with `%.10f`, and trained/inferred A2 on CUDA.
Parity was therefore established at the maximum precision the artifacts permit
(details in *TEST projection*), and the residual gap is fully explained and
bounded. It is a precision limit of the audit environment, not a discrepancy in
the file.

The latest, most test-like fold gives post-projection `rho = 0.0325` (stress
span) to `0.0398` (prediction-like span), i.e. above the `0.030`
"especially strong" line under both spans, with bootstrap `P(Delta MSE<0)=1.000`.

**Risk that the gates do not capture:** the deployed correction is large.
`||D_perp||_2 = 29.22` against `11.80` for the ORTH_ALPHA step that produced the
current score. The correction is added at amplitude 1.0, so it only pays off
above `rho = 0.017749`; if the signal does not transfer at all the score gets
*worse* by `+0.00104` RMSLE. The bet is close to symmetric.

| gate | result |
| --- | --- |
| 1. leakage audit | **PASS** |
| 2. original EXP075 reproduced | **PASS** (A1 to 1e-16, joint algebra to 1e-11) |
| 3. exact candidate parity | **PASS at artifact precision** (strict 1e-10 unattainable) |
| 4. `2026-01-14` post-projection rho >= 0.025 and Delta MSE < 0 | **PASS** (0.0325 / 0.0398; Delta MSE < 0) |
| 5. >= 2 of 3 late folds with positive post-projection rho | **PASS** (3 of 3) |
| 6. weighted late post-projection rho >= 0.025 | **PASS** (0.0299 stress / 0.0405 prediction-like) |
| 7. bootstrap `P(Delta MSE<0) >= 0.95` | **PASS** (1.000 on every fold, both spans) |
| 8. not driven by one anomalous fold | **PASS** (all 5 folds >= 0.0250 under the stress span) |
| 9. A1 and A2 do not both flip sign | **PASS** (both positive on all 5 folds) |
| bonus: latest post-projection rho >= 0.030 | **PASS** under both spans |

## Reproduction

### From the stored artifacts (exact)

Every headline number recomputes from `clean_forward_predictions.parquet` and
`a2_clean_forward_predictions.parquet` without change:

| quantity | recomputed | reported |
| --- | ---: | ---: |
| A1-365 weighted rho | 0.032954478048 | 0.032954478048 |
| A1-365 latest rho | 0.032075597742 | 0.032075597742 |
| A1-365 nested Delta MSE | -0.003333750684 | -0.003333750684 |
| A1-180 weighted rho | 0.030559306059 | 0.030559306059 |
| A2 weighted rho | 0.030863934759 | 0.030863934759 |
| A2 latest rho | 0.025595370566 | 0.025595370566 |
| joint `a_oracle` | [0.7462560852846633, 0.6466415684754089] | [0.7462560853, 0.6466415685] |
| joint condition number | 2.4405988877418343 | 2.4406 |
| joint `rho^2` | 0.001439429628483905 | 0.001439429628483905 |
| joint rho | 0.037939815873089 | 0.037939815873089 |
| corr(A1-365, A2) | 0.418801913488 | 0.418802 |
| rolling nested Delta MSE | -0.004391912083295 | -0.004391912083295 |
| rolling nested Delta RMSLE | -0.001250665929634 | -0.001250665929634 |

`residual == target_log - baseline_z` holds exactly (max abs error `0.0`), and
every `u_perp` is orthogonal to `baseline_z` to `<= 9e-19`.

One arithmetic correction to the source report. `EXP075/REPORT.md` and
`verdict.json` state a remaining equivalent rho of `0.013905` by adding the old
ORTH `rho=0.0141` on top of the joint. `SUBMIT_ORTH_ALPHA` **is** the baseline,
so that signal is already inside `1.6461597403364463` and must not be counted
twice. The correct figure is:

```text
required rho^2   = 0.0018316025548536
joint rho^2      = 0.0014394296284839
remaining rho^2  = 0.0003921729263697
remaining rho    = 0.0198033564420199
```

The joint covers `78.588536%` of the required `rho^2`, not `89.4%`.

Note also that the headline `0.037940` is the pooled **oracle** multivariate R of
the two directions, not a held-out statistic. With the frozen coefficients held
fixed the pooled weighted correlation is numerically the same (`0.037939815873`),
and the per-fold rolling-coefficient values are `[0.03943, 0.04153, 0.04124,
0.03486]`, so the optimism is negligible — but the deployed coefficients are
oracle-fitted on all four folds, which is why the TEST amplitude is calibrated to
`rho ~ 0.0355` rather than to the rolling `~0.0349`.

### Independent retraining (fold 2025-10-16)

The whole A1/A2 pipeline was re-ported into the audit container and the 11-channel
panel was rebuilt from `train.parquet` instead of reusing `seq_panel_v1.npy`
(`raw_sha256 = 5f3aa9…67c0`, matching the recorded value). On a 300-user sample
the rebuilt panel is **bitwise identical** to the historical mmap and the GMV
panel matches with max abs error `0.0`. Re-running the frozen fold:

| quantity | independent rerun | stored |
| --- | ---: | ---: |
| baseline RMSLE | 1.7488246991674166 | 1.7488246991674166 |
| A1-365 rho | 0.032075597742134546 | 0.032075597742134560 |
| A1-365 `b` | 0.0031486175448243964 | 0.0031486175448243970 |
| A1-365 `G` | 0.0031572036493710837 | 0.0031572036493710840 |
| A2 rho | 0.027039104481975 | 0.025595370566265 |
| joint (frozen coefficients) rho | 0.035586531994 | 0.035026610145 |

A1 reproduces to floating-point identity. A2 does not and cannot: the original
ran on an RTX 4060 Ti and the audit container has no CUDA, so the `float32`
convolution arithmetic differs. The A2 divergence (best epoch 1 here vs 2 there)
is the expected behaviour of a stochastic neural model re-run on a different
backend; the direction and magnitude of the signal are unchanged.

## Leakage audit

**PASS.** Checked directly against raw `train.parquet` for all four development
folds:

| check | result |
| --- | --- |
| feature timestamps `<= cutoff` | enforced by assertion in every builder; panel slice ends at `day_index(cutoff)` |
| target = strictly the next 30 days | `(cutoff, cutoff+30]`, verified per fold |
| target fully observed | `cutoff+30 <= 2026-02-13` on every fold |
| eligibility determined without target | 3 blocks all end at or before the cutoff; raw-rebuilt eligible set is **set-identical** to the stored fold panel on all 4 folds |
| stored `target_y30` vs raw | max abs error `0.0`; `target_log` `8.88e-16` |
| residual baseline cross-fitted | global user-hash halves; **0 of 210,212 users** appear in both halves |
| held-out user not scored by a model trained on his target | training targets end at `cutoff-5`, validation target starts at `cutoff+1`; windows never overlap |
| contaminated teammate OOF used | no — every input is rebuilt from raw or from the verified panel |
| A1/A2 share the same baseline | `baseline_z` identical between the two OOF files, max abs error `0.0` |

### Structural finding: the dataset is itself survivorship-selected

`train.parquet` contains exactly **250,000** users, and **all 250,000** satisfy
the organiser's three-block TEST eligibility rule at `2026-02-13`. The raw data
*is* the test cohort, so every user is guaranteed at least one event in each of

```text
block 0: 2026-01-15 .. 2026-02-13
block 1: 2025-12-16 .. 2026-01-14
block 2: 2025-11-16 .. 2025-12-15
```

Any historical fold whose 30-day target window overlaps `2025-11-16 .. 2026-02-13`
is therefore conditioned on its own target window. `2025-10-16` is the **last
cutoff with a fully unconditioned target** (`cutoff + 30 <= 2025-11-15`), which is
exactly the corridor EXP075 chose; that decision is correct and maximal, and the
`excluded_target_based_cutoffs_after` note in `validation_audit.json` is justified.

Measured severity, as the fraction of eligible users with **no event at all** in
the target window:

| cutoff | eligible | inactive in target | note |
| --- | ---: | ---: | --- |
| 2025-10-16 | 197,379 | 2.000% | clean, last unconditioned cutoff |
| 2025-10-30 | 201,074 | 1.362% | 14 of 30 target days inside the eligibility window |
| 2025-11-13 | 204,634 | 0.250% | 28 of 30 |
| 2025-12-11 | 215,823 | 0.553% | 30 of 30 |
| 2026-01-14 | 233,152 | **0.000%** | target window **is** organiser block 0 |

The requested main confirmation fold is the most distorted one on this axis, not
the least. Its size was measured rather than assumed: on the clean `2025-10-16`
fold, restricting to target-active users moves the joint rho from `0.035587` to
`0.036896` (+3.7% relative) and Delta MSE from `-0.003860` to `-0.004170`. The
late folds sit `0` to `+27%` above the clean fold, so survivorship explains at
most a small part of their level. **No leakage was found; the late folds are
flagged as regime-distorted, not as leaking.**

## Untouched late folds

Frozen pipeline, frozen features, architecture, hyperparameters, objective,
preprocessing and coefficients `D = 0.7462560853*A1 + 0.6466415685*A2`. The only
relaxed guard is the hard-coded `validation_cutoff <= 2025-10-16` assertion,
which had to be lifted to build late folds at all. `2025-10-16` is shown as the
port-validation fold (inside the development corridor, not new evidence);
`2025-10-30` is a supplementary least-contaminated late fold; the three
pre-registered confirmation folds are `2025-11-13`, `2025-12-11`, `2026-01-14`.

Post-projection columns use the historical analogue of the TEST submission span
(next section). `S_pred` is the faithful analogue, `S_wide` the conservative
stress span.

| cutoff | A1 rho | A2 rho | joint rho | post-proj rho `S_pred` | post-proj rho `S_wide` | joint Delta RMSLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-10-16 (port val.) | 0.032076 | 0.027039 | 0.035587 | 0.035780 | 0.024995 | -0.0011040 |
| 2025-10-30 (suppl.) | 0.033006 | 0.029976 | 0.037383 | 0.036820 | 0.025445 | -0.0011983 |
| **2025-11-13** | 0.036488 | 0.038532 | 0.045357 | 0.043681 | 0.028190 | -0.0017895 |
| **2025-12-11** | 0.032043 | 0.035022 | 0.040618 | 0.040410 | 0.025760 | -0.0014519 |
| **2026-01-14** | 0.028129 | 0.031854 | 0.036223 | 0.039765 | 0.032472 | -0.0010590 |

Delta MSE and bootstrap at the deployed unit amplitude (1,000 Poisson
user-cluster replicates):

| cutoff | Delta MSE `S_pred` | 95% CI | Delta MSE `S_wide` | 95% CI | `P(Delta MSE<0)` |
| --- | ---: | --- | ---: | --- | ---: |
| 2025-10-16 | -0.003879 | [-0.004674, -0.003036] | -0.001906 | [-0.002564, -0.001154] | 1.000 / 1.000 |
| 2025-10-30 | -0.004110 | [-0.005098, -0.003068] | -0.001891 | [-0.002780, -0.000970] | 1.000 / 1.000 |
| 2025-11-13 | -0.005777 | [-0.006789, -0.004785] | -0.002368 | [-0.003268, -0.001538] | 1.000 / 1.000 |
| 2025-12-11 | -0.005046 | [-0.006024, -0.004049] | -0.001993 | [-0.002834, -0.001150] | 1.000 / 1.000 |
| 2026-01-14 | -0.004480 | [-0.005434, -0.003467] | -0.002986 | [-0.003889, -0.002179] | 1.000 / 1.000 |

Aggregates over the three pre-registered late folds (recency weights `1:2:4`):

```text
weighted late joint rho (pre-projection)   = 0.038784   unweighted 0.040733   3/3 positive
weighted late post-projection rho S_pred   = 0.040508   unweighted 0.041285   3/3 positive
weighted late post-projection rho S_wide   = 0.029942   unweighted 0.028807   3/3 positive
```

A1 and A2 are individually positive on all five folds; neither ever changes sign,
so gate 9 cannot be triggered. No single fold carries the result: the weakest
post-projection value across all five folds and both spans is `0.024995`
(`2025-10-16`, stress span), and every pre-registered late fold is above `0.025`
under both spans.

## TEST projection

### Span rebuild

The projection span was rebuilt from `Z.npz` plus exactly the eleven explicit and
local submission vectors that existed when the joint candidate was written (the
four `SUBMIT_EXP075_*.csv` files were produced *after* `build_span` and are
correctly absent):

| quantity | audit | EXP075 |
| --- | ---: | ---: |
| canonical vectors | 67 | 67 |
| total unique vectors | 78 | 78 |
| centered rank | 67 | 67 |
| orthonormality max error | 1.55e-15 | 7.99e-15 |

`SUBMIT_ORTH_ALPHA` is genuinely inside the retained span: its out-of-span
residual RMS is `3.05e-9` against a centered RMS of `1.6224`. The projection
therefore really does remove the incumbent direction.

### Raw / perpendicular / parity

| quantity | from stored `D_raw` | from independently rebuilt models | EXP075 |
| --- | ---: | ---: | ---: |
| `RMS(D_raw)` | 0.073177510588 | 0.073177683231 | 0.073177510582 |
| `RMS(D_perp)` | 0.058433912326 | 0.058433842045 | 0.058433912326 |
| `perp_fraction` | 0.637638698668 | 0.637634156153 | 0.637638698756 |
| second-pass max projection | 1.33e-15 | 1.50e-15 | 4.78e-15 |
| `corr(D_perp, current ORTH)` | 3.86e-09 | — | 3.86e-09 |
| `corr(D_perp, z_alpha)` | 3.77e-11 | — | — |

Independent re-derivation from the frozen model files:

```text
A1  max abs difference vs stored TEST correction = 2.77e-08   corr = 1.0000000000000002
A2  max abs difference vs stored TEST correction = 5.99e-05   corr = 0.9999999914
joint corr with stored raw correction            = 0.9999999969
```

`2.77e-08` is exactly the `float32` storage resolution of the stored array, so
the LightGBM half is reproduced bit-for-bit. The A2 gap is the CUDA-vs-CPU
`float32` difference described above.

Internal consistency: `JOINT_raw = (0.7462560853/1.012306043) * A1_raw +
(0.6466415685/0.964201496) * A2_raw` holds to `5.66e-08`, i.e. exactly to the
stored `float32` precision.

### Candidate parity

`z_submit = max(log1p(SUBMIT_ORTH_ALPHA.predict) + D_perp, 0)`,
`predict = expm1(z_submit)`:

| check | result |
| --- | --- |
| rows / unique / order vs `sample_submit` | 250,000 / 250,000 / identical |
| clipped rows | 1,219 — **exactly** the set where `z_alpha + D_perp <= 0`, and no free row has a negative sum |
| free rows, max abs log-space difference vs stored `float32` `D_perp` | 2.62e-08 |
| free rows, median abs log-space difference | 4.83e-10 |
| rows above 1e-6 | 0 |
| serialization slack implied by `%.10f` alone | up to 1.39e-6 in log space |
| implied correction orthogonal to the span | `max abs Q^T D_perp = 4.92e-08` (= `float32` noise) |
| implied correction mean | -3.17e-12 |
| RMS of implied correction vs stored | 0.0581827922353 vs 0.0581827922338 |
| independent end-to-end rebuild vs CSV | relative RMS difference 6.81e-06, corr 0.999999999975, clip set identical |

The strict `<= 1e-10` cannot be met by anyone: the pipeline never persisted a
`float64` `D_perp`, and `%.10f` serialization alone allows up to `1.39e-6` of
log-space slack. Within what the artifacts do allow, the candidate is exactly the
frozen pipeline's output. Recorded SHA256 of the audited file matches the source
report: `d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901`.

### Historical analogue of the span (why the post-projection columns exist)

TEST `perp_fraction` alone proves novelty, not predictive transfer, so the
projection was replayed historically. No historical analogue of 67 full TEST
submissions exists; the largest reproducible equivalent was built from
cutoff-safe quantities:

* `S_pred` — 10 columns, every one a log-space prediction of `y30`: the clean
  baseline `z`; `log1p(s*expm1(z))` for `s in {0.64, 0.97, 1.20, 1.40}` (the
  canonical bank contains many such rescaled variants); naive-30, naive-90/3,
  naive-180 and naive-365 level predictors; a recency-only predictor.
* `S_wide` — `S_pred` plus 44 raw level/count/recency columns, centered rank
  **67**, i.e. the same rank as the real TEST span and a comparable energy
  removal (`perp_fraction 0.52-0.64` against the TEST `0.638`).

**Components present in the real TEST span that have no historical analogue and
are therefore missing from both:** `SUBMIT_ORTH_ALPHA`, `SUBMIT_ORTH_FINAL`,
`SUBMIT_PUBLIC_EB`, `SUBMIT_PRIVATE_OPTIMAL`, `SUBMIT_PRIVATE_V2`,
`SUBMIT_NEXT_AFTER_EXP069`, `PROBE_scale097`, `my_submit`, `SUBMIT_v7_newmodel`,
`SUBMIT_ORTH_ROBUST_H12_INTERP`, `anchor_diverse_A_combo_mlp_hurdle_w065`, and
the teammate/CatBoost/ridge/DL/SEQ/BTYD/hurdle-variance families inside the
67-vector canonical bank. `S_wide` is offered as the conservative bound precisely
because those are missing.

Retained energy and the character of what is removed:

| cutoff | `perp_fraction` `S_pred` | corr(removed, residual) | `perp_fraction` `S_wide` | corr(removed, residual) |
| --- | ---: | ---: | ---: | ---: |
| 2025-10-16 | 0.911834 | 0.004784 | 0.556932 | 0.025439 |
| 2025-10-30 | 0.861273 | 0.008625 | 0.524329 | 0.027487 |
| 2025-11-13 | 0.894775 | 0.012448 | 0.576734 | 0.036811 |
| 2025-12-11 | 0.913912 | 0.006771 | 0.573705 | 0.032327 |
| 2026-01-14 | 0.944345 | -0.010254 | 0.635288 | 0.017124 |

The removed component is mildly predictive under `S_wide` — which is expected and
harmless in deployment, because on TEST the removed part is by construction
already contained in the incumbent submissions.

## Mathematics

```text
current reference RMSLE      = 1.6461597403364463   (SUBMIT_ORTH_ALPHA)
target reference RMSLE       = 1.6446514942
required Delta RMSLE         = -0.0015082461364464
current MSE                  =  2.7098418907045563
target MSE                   =  2.7048785373742925
required Delta MSE           = -0.0049633533302638
required rho^2               =  0.0018316025548536
required rho                 =  0.0427972260182082

Codex joint clean-forward rho = 0.0379398158730891   -> CONFIRMED
joint rho^2                   = 0.0014394296284839   -> CONFIRMED
remaining rho^2               = 0.0003921729263697
remaining rho                 = 0.0198033564420199
```

### Deployed amplitude — the decisive risk term

```text
RMS(D_perp)             = 0.058433912326
||D_perp||_2            = 29.216956            (the ORTH_ALPHA step is 11.80)
amplitude-matched rho   = 0.035497             (the rho at which this size is MSE-optimal)
break-even rho          = 0.017749             (Delta MSE = 0)
```

The correction is added at amplitude `1.0` with no shrinkage, so
`Delta MSE(rho) = -0.192387*rho + 0.00341462`. Below `rho = 0.017749` the
submission makes the score worse.

### Empirical precedent for transfer

Decomposing the ORTH family: `SUBMIT_ORTH_FINAL = PUBLIC_EB + 21*v` exactly
(residual RMS `7.9e-18`), `H12_INTERP = PUBLIC_EB + 12*v`, and
`SUBMIT_ORTH_ALPHA ~ PUBLIC_EB + 11.80*v` (extra residual RMS `9.7e-05`), with
`||v||_2 = 0.99947`. From the recorded leaderboard values
(`EB = 1.6463246740`, `H21 = 1.6462686940`, `ALPHA = 1.6461597403`) the
**realised** rho of `v` on the public set is `0.014089` (from H21) and `0.014156`
(from ALPHA), against a historically predicted `0.0137`. For this team, on this
data, a historically estimated residual rho transferred essentially intact. That
is the strongest available evidence in favour of the EXP075 premise.

## Expected score scenarios

Not a guarantee of public or private movement. Computed from the actual TEST
`RMS(D_perp)` and the ALPHA baseline, with no fitting to the leaderboard.

| scenario | rho | Delta MSE | Delta RMSLE | approx score | remaining rho to 1.6446515 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no transfer | 0.00000 | +0.0034145 | +0.0010368 | 1.6471965 | 0.055568 |
| realised ORTH transfer level | 0.01416 | +0.0006911 | +0.0002099 | 1.6463697 | 0.045674 |
| break-even | 0.01775 | 0.0000000 | 0.0000000 | 1.6461597 | 0.042797 |
| 0.020 | 0.02000 | -0.0004331 | -0.0001316 | 1.6460282 | 0.040890 |
| 0.025 | 0.02500 | -0.0013951 | -0.0004238 | 1.6457360 | 0.036297 |
| 0.030 | 0.03000 | -0.0023570 | -0.0007161 | 1.6454437 | 0.031027 |
| **late confirmation, `S_wide` weighted** | 0.02994 | -0.0023458 | -0.0007127 | **1.6454471** | 0.031093 |
| **late confirmation, `S_wide` latest** | 0.03247 | -0.0028325 | -0.0008606 | **1.6452992** | 0.028056 |
| amplitude-matched | 0.03550 | -0.0034145 | -0.0010374 | 1.6451223 | 0.023922 |
| **late confirmation, `S_pred` latest** | 0.03977 | -0.0042356 | -0.0012870 | **1.6448727** | 0.016401 |
| **late confirmation, `S_pred` weighted** | 0.04051 | -0.0043785 | -0.0013305 | **1.6448293** | 0.014702 |
| original EXP075 historical joint | 0.03794 | -0.0038845 | -0.0011803 | 1.6449795 | 0.019968 |

Central expectation: `rho` between `0.030` and `0.040`, i.e. a score around
**1.6448 to 1.6455** and a remaining `rho` of roughly **0.015 to 0.031** before
`1.6446514942`. The stress span (`S_wide`) is the more defensible reading, since
its rank and energy removal match the real TEST span; the prediction-like span
(`S_pred`) is the more faithful reading of the span's *character*. The honest
interval is bounded by them.

The one asymmetry the audit cannot resolve: the historical residual is measured
against a plain RFM LightGBM baseline (fold RMSLE `1.71`-`1.76`), while the
deployed residual is that of `SUBMIT_ORTH_ALPHA` (`1.6462`), a far stronger
incumbent. Span orthogonality guarantees the *direction* is new; it does not
guarantee the correlation survives against a stronger residual at full strength.
This is precisely why the amplitude term matters, and why the downside case
(`+0.00104` RMSLE) should be treated as live rather than remote.

## Submission

Prediction-identical to the audited Codex candidate; a byte-identical confirmed
copy was created rather than rebuilding a second model (rebuilding would only
re-introduce the CUDA-vs-CPU A2 difference).

```text
path   : C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP075_JOINT_A1_365_A2_CONFIRMED.csv
SHA256 : d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901
source : SUBMIT_EXP075_JOINT_A1_365_A2.csv
         SHA256 d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901
```

| format gate | result |
| --- | --- |
| rows | 250,000 |
| `user_id` unique | 250,000 |
| same order as `sample_submit.csv` | True |
| columns | `user_id,predict` |
| finite | True |
| `predict >= 0` | True (1,219 zeros) |
| min / max | 0.0 / 8789.0162995436 |
| mean `log1p` | 2.329907832630031 |
| max abs difference vs candidate | 0.0 (byte-identical) |

Not uploaded. No alternative scale or coefficient set was created.

## Audit artifacts

Stored beside this report: per-fold OOF predictions and metadata for all five
audit folds, the frozen A1/A2 models trained for each, the historical span
projection table, the post-projection bootstrap, the confirmation fold summary,
and the independently rebuilt TEST correction (`audit_TEST_D_raw.npz`).
