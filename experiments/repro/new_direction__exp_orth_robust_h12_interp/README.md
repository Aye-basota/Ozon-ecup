# EXP ORTH-ROBUST H12-INTERP

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp_orth_robust_h12_interp`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP_ORTH_ROBUST_H12_INTERP`
- **Original source:** `research/new_directions/EXP_ORTH_ROBUST_H12_INTERP`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** Unknown / not recoverable from repository history
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Using baseline RMSLE `1.6463246740442117` and
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 2 files, 1 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP ORTH-ROBUST H12-INTERP

## Verdict

**GO_H12_INTERP**

The fixed one-sided gate passes for both the full-test and worst-case private
bounds. The only candidate created is the preregistered `h=12` observable
interpolation; no fitting, calibration, segment rule, alternative scale, or
leaderboard probe was used.

## Why previous gate was insufficient

The previous symmetric inequality

```text
abs(RMSLE(UPPER) - RMSLE(LOWER)) <= rms(z_upper-z_lower)
```

bounds either direction of score movement and discards the structure of the 129
ambiguous rows. Its value, `2.038127364742e-4`, is useful as a generic metric but
is not the relevant worst-case loss from choosing INTERP: on these rows the exact
H12 prediction is known to lie below INTERP, and RMSLE targets are nonnegative.
The corrected one-sided argument bounds only how much worse INTERP can be.

## One-sided loss proof

For an ambiguous row, H21 clipping gives

```text
clip(za + 21*v, 0) = 0  =>  v <= -za/21.
```

Therefore the unknown exact H12 log-prediction is

```text
q = clip(za + 12*v, 0),
0 <= q <= za - 12*za/21 = (3/7)*za.
```

The observable interpolation on the same row is

```text
u = za + (12/21)*(z21-za) = (3/7)*za,
```

so `0 <= q <= u` follows directly from clipping. Since the RMSLE-space target
`t=log1p(y)` is nonnegative,

```text
(u-t)^2 - (q-t)^2
= (u-q)(u+q-2t)
<= (u-q)(u+q)
= u^2-q^2
<= u^2.
```

Summing over ambiguous rows gives the deterministic one-sided bound

```text
extra_MSE(INTERP versus exact H12) <= sum(u_i^2) / evaluation_size.
```

No target values or public fitting enter this proof.

## Recoverability

The row classes were recomputed from the two source CSVs:

| class | count | result |
|---|---:|---|
| `z21 > 0` | `249185` | exact H12 from interpolation |
| `z21 = 0` and `za = 0` | `686` | exact clipped H12 is zero |
| `z21 = 0` and `za > 0` | `129` | `0 <= exact H12 <= INTERP` |
| exact rows total | `249871` | `99.9484%` |
| ambiguous fraction | `0.000516` | `0.0516%` |

The classes are exhaustive. On the 249185 non-clipped rows the two algebraic H12
constructions agree to max absolute error `2.22e-16` in log space; the 686
zero-anchor rows are exactly zero. There are no other differences.

## Correct uncertainty bound

For `u=z_interp` restricted to the 129 ambiguous rows:

| metric | value |
|---|---:|
| count | `129` |
| fraction of 250000 | `5.16e-4` |
| `sum(u_i^2)` | `1.0384907887280059e-2` |
| L2 norm of `u` | `1.0190636823712275e-1` |
| full-test extra MSE bound, `/250000` | `4.1539631549120235e-8` |
| private-only worst-case extra MSE bound, `/200000` | `5.1924539436400293e-8` |

Using baseline RMSLE `1.6463246740442117` and
`Delta RMSLE ~= Delta MSE/(2*baseline_RMSLE)`:

| bound | approximate RMSLE impact | exact square-root conversion |
|---|---:|---:|
| full test | `1.2615868608431212e-8` | `1.2615868572041222e-8` |
| private worst case | `1.5769835760539014e-8` | `1.5769835659540377e-8` |

The private bound deliberately assumes all 129 ambiguous users are among the
200,000 private users.

## Expected H12 gain

Only the preregistered most test-like historical value is used:

```text
rho = 0.0137
N = 250000
h = 12
anchor RMSLE = 1.6463246740442117
```

For the unit-direction geometry,

```text
Delta MSE = h^2/N - 2*h*rho*anchor_RMSLE/sqrt(N)
          = -5.0662310565147345e-4

Delta RMSLE
          = sqrt(anchor_RMSLE^2 + Delta MSE) - anchor_RMSLE
          = -1.5387207342953957e-4
```

The linear MSE-to-RMSLE approximation is `-1.5386488268044637e-4`. The public
H21 score was not used to choose or change `h`.

## Risk ratio

The fixed gate requires each MSE bound to be at most 1% of the expected H12 MSE
gain magnitude. The 1% threshold is `5.066231056514735e-6`.

| risk ratio | value | percent of expected gain | gate |
|---|---:|---:|---|
| full-test bound / expected MSE gain | `8.199316431828325e-5` | `0.0081993164%` | PASS |
| private bound / expected MSE gain | `1.0249145539785405e-4` | `0.0102491455%` | PASS |

Both are roughly two orders of magnitude below the fixed 1% allowance.

## Output

- Path:
  `C:/Users/Admin/Desktop/e-cup-research-clean/submissions/SUBMIT_ORTH_ROBUST_H12_INTERP.csv`
- SHA256:
  `1ab19eaaf78124d574f43d98c0585c3c2b58833a1aae52adf80a8cd07041d29f`
- Formula:
  `z = za + (12/21)*(z21-za)` and `predict = max(expm1(z),0)`.
- Rows: `250000`.
- Columns: `user_id,predict`.
- Same user order as anchor: `True`.
- Unique user IDs: `250000`.
- Finite predictions: `True`.
- Nonnegative predictions: `True`.
- Min / max predict: `0.0 / 3232.1837536418`.
- Zero predictions: `686`.
- Mean / std `log1p`: `2.32966289712869 / 1.62243748297175`.
- `rms(z_interp-za)`: `0.0239871870120033`.
- `corr(z_interp,za)`: `0.999890701532020`.
- `corr(z_interp,z21)`: `0.999938556134775`.
- Step RMS ratio: `0.571428571428453` versus `12/21`.
- Change-direction correlation: `1.0`.
- Max serialization difference from the in-memory formula: `5.01e-11` in
  prediction and `4.99e-11` in log space.
- Byte regeneration with the fixed CSV format is deterministic and reproduces the
  same SHA256.

The deterministic builder is
`research/new_directions/EXP_ORTH_ROBUST_H12_INTERP/build_submission.py`.
Existing submissions and `scores/` were not modified.

## Final conclusion

The corrected one-sided proof uses the facts that the exact ambiguous H12 values
lie in `[0,u]` and RMSLE-space targets are nonnegative. It reduces the relevant
worst-case full-test loss to `4.15396e-8` MSE (`1.26159e-8` RMSLE), only
`0.00820%` of the expected H12 MSE gain; even the all-private worst case is only
`0.01025%`. Both pass the fixed 1% gate, so the final verdict is
**GO_H12_INTERP**.
