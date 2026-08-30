# EXP-054 — BURST-GAP-ETX final report

- **Endpoint:** `NO_GO_PREFLIGHT`
- **GPU pilot:** not run by gate
- **Full folds / test inference / submission / public LB:** not run
- **Baseline revision:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Isolation:** EXP-054-only files in the current tree. A clean worktree could not contain the required uncommitted ETX/EXP-053 baseline.

## Exact baseline audit

The five OOF components align exactly on 770,616 unique `(cutoff,user_id)` rows with identical targets. Reconstructed

```text
STRONGEST_CURRENT = .10 CAP + .20 UNC + .25 DIST + .225 ETX-AVG3 + .225 SEQ-AVG3
```

gives fold scores `1.766883357 / 1.760509577 / 1.748629224 / 1.741278566`, wCV `1.747509863`. The required 2025-10-16 score matches within `4.5e-10`.

`ETX-01-S42-V1016` checkpoint config is exact: `d_model=128`, five blocks, eight heads, `head_dim=16`, FFN 384, dropout .1, AdamW `lr=1.5e-3`, `wd=1e-2`, warmup 500, four epochs, seed 42. Candidate type embeddings add 640 parameters: `1,116,073` versus `1,115,433`, or `+0.0574%`.

One code-level discrepancy was found: `n_tok=192` in current ETX means 192 history slots **plus** a separately appended query, so the implemented maximum is 193 positions, not “192 including query.” The candidate specification remains fixed at 191 history tokens plus query. This did not affect the CPU endpoint because no neural pilot ran.

## Structural audit

The fixed threshold-3 segmentation was built for all four folds. On 2025-10-16 the mean/median burst count is `18.74 / 20`, mean token count `37.47`, maximum 80, and overflow is exactly 0%. Singleton bursts are 36.5% on that fold (36.4% overall).

Novelty passes: not all key episode summaries are near-duplicates of existing state. Examples of maximum absolute Spearman correlation with the requested existing state families:

| Episode variable | max | closest existing |
|---|---:|---|
| `n_bursts` | 0.349 | `buygap_mean` |
| `last_burst_intensity` | 0.469 | `rec_buy` |
| intensity ratio | 0.346 | `trend_srch_7_30` |
| `median_closed_gap` | 0.846 | `gap_mean` |
| `last_burst_slope` | 0.189 | `trend_gmv_7_30` |

Two explicit redundancies were also exposed: `current_open_gap` equals `rec_any` (Spearman 1.0), and `current_gap_ratio` is highly correlated with it (0.977). Existing caches contain `buygap_mean/std`, but not the requested `buygap_median/max`; this absence is recorded rather than synthesized.

The strongest within-stratum mid-activity spread in `P(y30>0)` is only `0.01998`, below the fixed 0.03 gate.

## REAL vs SHUFFLED residual probe

The probe used the exact EXP-053 COMBINED base (227 state + 34 disagreement columns) plus the 12 fixed episode summaries. SHUFFLED permuted those 12 columns jointly inside fold × STRONGEST decile × fixed `rec_buy` bin × fixed `w180_days_buy` bin. User halves are disjoint by `splitmix64(user_id)&1`.

| Check | REAL | SHUFFLED | Required |
|---|---:|---:|---:|
| donor-selected scale, A→B | 0 | 0 | nonzero gain |
| donor-selected scale, B→A | 0 | 0 | nonzero gain |
| late-fold RMSLE delta | 0.000000 | 0.000000 | REAL ≤ −0.0005 |
| REAL − SHUFFLED | 0.000000 | — | ≤ −0.0004 |
| both recipient halves better | no (ties) | no | yes |
| selected correction/residual corr | 0 / 0 | 0 / 0 | >0 in both halves |
| raw unscaled probe/residual corr | +0.00699 / +0.01164 | +0.00693 / +0.01141 | diagnostic only |
| activity AUC | 0.847581 | 0.847560 | gain ≥ 0.002 if spread gate fails |
| AUC gain | **+0.000021** | — | ≥ +0.002 |

The weak raw-probe correlation is fully matched by SHUFFLED and donor-side selection collapses both scales to zero, so the actual correction correlation is zero. Every effect-size gate fails except structural novelty.

## Decision

`NO_GO_PREFLIGHT`. Explicit bursts/gaps are structurally novel in several axes, but the fixed summaries contain no usable late-fold residual or occurrence signal beyond the matched control. Per the preregistered gate, GPU pilot, one-fold neural training, full folds, test inference, and submission are prohibited.

This closes this exact threshold-3 BURST/GAP representation route without threshold, architecture, feature, or blend-weight tuning. It does not claim that every possible state-space model is impossible; it says the preregistered episode construction failed before neural compute.

## Artifacts

- `artifacts/BURST_GAP_EXP054/`: initial git status, exact baseline manifest, segmentation/cache manifests, episode cache, structural audit, shuffle audit, pre-flight verdict, GPU plan, hash replay and summary.
- `research/strategies/results/BURST_GAP_EXP054/`: structural statistics, novelty correlations, within-stratum diagnostics, recipient-half metrics, scale-selection curves and this report.
- No checkpoint or prediction artifact exists because the gate failed.

## Files changed by EXP-054

- New code/tests: `src/burst_gap_etx.py`, `src/test_burst_gap_etx.py`.
- New experiment record/report: `experiments/exp_054_burst_gap_etx.md`, `research/strategies/results/BURST_GAP_EXP054/REPORT.md` and the CSV diagnostics in that directory.
- New ignored artifacts/cache: `artifacts/BURST_GAP_EXP054/`.
- Shared registries updated only after the endpoint: `STATE.md`, `HISTORY.md`, `experiments/log.csv`.
- No shared production module, `src/config.py`, or `src/validation.py` was changed.

## Reproduction

```text
python src/burst_gap_etx.py
python -m pytest src/test_burst_gap_etx.py -q
```

Final focused/regression verification: `78 passed` (`test_burst_gap_etx`, `test_etx`, `test_residual_signal_discovery`, `test_validation`); analysis-only hash replay PASS.
