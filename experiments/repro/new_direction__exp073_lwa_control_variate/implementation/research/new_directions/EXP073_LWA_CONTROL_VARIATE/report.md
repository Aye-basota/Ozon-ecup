# EXP073 — LWA FRESH−VOL Control-Variate Correction

## Verdict

**REJECT_HEADROOM**. Failed Stage 0 gate(s): `corr_fresh_vol, theoretical_optimal_gain`. The protocol stops before the untouched confirmation fold, seed 43, full OOF, TEST inference, and submission creation.

## Stage 0 headroom (`2025-10-16`)

- EXP072 parity: **PASS**; maximum score error `0.000e+00` (tolerance `2e-6`).
- `corr(d_fresh,d_vol)`: `0.460682`.
- RMS of arm-preprocessed `d_fresh-d_vol`: `0.072438`.
- RMS of direct EXP073 correction: `0.071296`.
- `A`: `+7.824968987e-04`; `Q`: `5.083175057e-03`; analytic `alpha*`: `+0.153939`.
- Theoretical optimal delta: `-0.000034582` (gain `+0.000034582`).

The complete frozen alpha curve is in `stage0_alpha_curve.csv`; row-level parity and correction vectors are in `stage0_design_rows.parquet`.

The best point on the frozen grid is `alpha=0.25`, with delta `-0.000022017`; this is far below the preregistered theoretical-headroom requirement. An independent row audit passed canonical order, unique-user, target-alignment, finiteness, and correction-identity checks. Model levels are highly correlated (`corr(mu_fresh,mu_vol)=0.986808`), but their retraining deltas are not (`0.446959` raw; `0.460682` processed), directly contradicting the common-drift premise.

## Confirmation and full canonical validation

Not run by gate. There is no confirmatory estimate, four-fold wCV/LOFO result, REAL-vs-null estimate, fold-sign count, bootstrap interval, standalone/EXP069 combination, or TEST diversity/span measurement.

## Production audit

TEST inference was not authorized. No `exp073_lwa_cv_TEST.*`, builder, or `SUBMIT_EXP073_LWA_CONTROL_VARIATE.csv` was created. Existing artifacts were not overwritten. Public-LB information and reconstructed incumbent OOF were not used.

## Estimated public gain

Not estimated: the headroom gate failed, and the public leaderboard is not a model-selection source.
