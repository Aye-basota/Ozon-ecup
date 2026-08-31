# EXP088 — A1/A2 Residual-Plane Tomography

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp088_a1_a2_tomography`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP088_A1_A2_TOMOGRAPHY`
- **Original source:** `research/new_directions/EXP088_A1_A2_TOMOGRAPHY`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 3 files, 1 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP088 — A1/A2 Residual-Plane Tomography

Status: **PRE-LB / two probes created, not submitted**. No model was trained, no
feature was added, and no leaderboard score was used to choose the direction,
sign, robustification, or scale.

## Artifact reconstruction

The exact frozen EXP075 TEST artifacts were recovered. The saved standalone A1
and A2 `TEST_PERP` arrays already contain their separately fitted standalone
amplitudes. The joint, however, was defined from the underlying unit directions.
Therefore the required vectors were reconstructed as

```text
d_A1 = A1_TREE_TRAJ_365_TEST_PERP / 1.012306043162683
d_A2 = A2_WEEKLY_RESIDUAL_CNN_TEST_PERP / 0.9642014960450844
```

Source artifacts:

| vector | source path | source SHA256 |
| --- | --- | --- |
| A1 | `research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS/A1_TREE_TRAJ_365_TEST_PERP.npy` | `9b4527057531f6981744c1f2369136ef4446623675a416c5c21b1195112d01ae` |
| A2 | `research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS/A2_WEEKLY_RESIDUAL_CNN_TEST_PERP.npy` | `2ffc2ef15138ad1a64e95dac6373c7449006d5cfb8cc24df4ef1bf20c3d44e58` |
| joint | `research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS/JOINT_A1_365_A2_TEST_PERP.npy` | `e3667884a661adf64a6ce5f231956bab18e45a7e6f017e453506f5e93d3045da` |
| anchor | `C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv` | `9a8adb83e7b34bb6c12b7eb51584d1bf9a93825945d285258d4e1dd991f4b838` |

The recovered float64 vectors are frozen in this experiment:

| vector | path | SHA256 |
| --- | --- | --- |
| `d_A1` | `research/new_directions/EXP088_A1_A2_TOMOGRAPHY/d_A1_TEST.npy` | `052cd46640a119628bc2127cf90c71007aceb8bacff5d3154992e9c6606e6f61` |
| `d_A2` | `research/new_directions/EXP088_A1_A2_TOMOGRAPHY/d_A2_TEST.npy` | `f216c30bebd77cc3e45a6d3a352d217cfc1b9c99b715010a6e14e3d8aa7c63cd` |
| `d_joint` | `research/new_directions/EXP088_A1_A2_TOMOGRAPHY/d_joint_TEST.npy` | `781cd1af72e55d6a411707dc04d47c3c47829a1c88ab74f0f3ba8b14213d1248` |

Frozen reconstruction check:

```text
d_joint ~= 0.7462560853*d_A1 + 0.6466415685*d_A2
max absolute error = 5.148324044856878e-08
RMS error          = 1.9770832252922484e-09
mean error         = -5.096383053142421e-12
corr               = 0.9999999999999993
```

The maximum error is below the preregistered reasonable floating tolerance
`1e-7`; it is consistent with the three source arrays having been serialized
independently as float32.

## A1/A2 plane geometry

After projecting `d_A1` and `d_A2` out of the current sent span, constant, and
the exact pre-submit `d_joint`:

```text
RMS(w1)                         = 0.029430706054941275
RMS(w2)                         = 0.033964478253717350
RMS(d_A1)                       = 0.048769612958583350
RMS(w1) / RMS(d_A1)             = 0.6034640069818625
corr(w1,w2)                     = -0.9999999999999959
w2-on-w1 coefficient            = -1.1540490462686246
joint-implied coefficient       = -1.1540490461061350
RMS collinearity residual       = 3.0565672736697673e-09
```

Thus the genuinely new orthogonal axis before the fixed tail treatment has RMS
`0.029430706054941275`, retaining `60.3464%` of the A1 direction's RMS. This is
an axis-size statement only, not an estimate of predictive quality.

Canonical orientation is `w=w1`. No sign flip was needed; after the complete
robustification/reprojection procedure `corr(w,d_A1)=0.6031908143280144 > 0`.

## Basis/projection audit

The basis contains the exact 67-vector scored canonical TEST bank, 12 additional
exact sent/evaluated TEST submissions preserved outside that cache (including
`ORTH_ALPHA`, `ORTH_FINAL`, `PUBLIC_EB`, geometry v2/NEXT_BEST, and the actually
sent clipped EXP075 joint submission), plus the exact pre-submit `d_joint`.
The constant is handled separately by centering.

Prepared-but-not-sent EXP075 A1/A2 standalone files, EXP079 A040, EXP084, and
other explicitly documented PRE-LB candidates were deliberately not admitted
as sent directions. In particular, including the unsubmitted A1/A2 standalone
files would incorrectly annihilate the plane being measured.

```text
input vectors excluding constant = 80
centered numerical rank          = 71
rank including constant          = 72
SVD relative tolerance           = 1e-10
smallest retained singular ratio = 6.570694619747367e-10
largest rejected singular ratio  = 5.627235520588542e-13
basis orthonormality max error    = 1.2678746941219288e-13
```

Final robust axis audit after zero-centering and double projection:

```text
mean(w)                            = 1.900701818158268e-19
corr(w,d_joint)                    = -1.8367980107024134e-17
RMS(final projection into basis)   = 4.540863259696291e-19
projection RMS / RMS(w)            = 1.5457391942336813e-17
max absolute basis coefficient     = 1.487525380650112e-16
```

The complete source list, exact paths, and SHA256 values are in `audit.json`.

## Probe normalization

Distribution before the fixed robustification decision:

| statistic | value |
| --- | ---: |
| min | -0.4131117318967211 |
| max | 0.6659797859561243 |
| p0.01 | -0.1895193588092650 |
| p0.1 | -0.1133419832762535 |
| p1 | -0.0710189090619933 |
| p50 | -0.0013131481795983 |
| p99 | 0.0843301544232792 |
| p99.9 | 0.1455674266098815 |
| p99.99 | 0.2412392793497273 |
| RMS | 0.0294307060549413 |
| max_abs / RMS | **22.628739681367904** |

Since `22.6287 > 20`, the single allowed winsorization was applied at
`±10*RMS = ±0.29430706054941275`, followed by zero-centering and two projection
passes. No alternative threshold or scale was evaluated.

Distribution after the fixed robustification and reprojection:

| statistic | value |
| --- | ---: |
| min | -0.2956640374774447 |
| max | 0.2955678213558272 |
| p0.01 | -0.1906143919733099 |
| p0.1 | -0.1133423224979207 |
| p1 | -0.0709789233250067 |
| p50 | -0.0013132392189298 |
| p99 | 0.0843293497423523 |
| p99.9 | 0.1455885384617522 |
| p99.99 | 0.2415953588950147 |
| RMS | **0.02937664566335512** |
| max_abs / RMS | 10.064594878041525 |

Normalization and frozen probe:

```text
RMS(u)       = 1.0
probe_rms    = 0.025
RMS(d_probe) = 0.025
G            = mean(d_probe^2) = 0.0006250000000000001
```

Exact robust axis/probe paths:

```text
w_robust_TEST.npy SHA256 ad1f9297621f9b781aacae479c86fd7897ffcd38fa0c24ea633462b85d8a8995
d_probe_TEST.npy  SHA256 db230c1cdd233949d439829db9c17ef6391a6401f54296a6f9a7f15dc6011b0f
```

## Clipping audit

The prescribed un-clipped base

```text
z_base = z_ORTH_ALPHA + 0.50*d_joint
```

has 874 negative coordinates. Consequently clipping is materially nonzero for
both probes:

| audit | PLUS | MINUS |
| --- | ---: | ---: |
| rows with unclipped `z < 0` | **915** | **1,009** |
| RMS(clipped - unclipped) | 0.002835366858204795 | 0.002876569993635138 |
| max clipping difference | 0.23121463561765354 | 0.27215544340566220 |
| clipping RMS / probe RMS | 0.1134146743281918 | 0.1150627997454055 |

Therefore the nominal affine equations with `G=0.000625` are not registered as
exact for these files. All exact pre/post-clip float64 vectors were frozen before
LB in:

```text
research/new_directions/EXP088_A1_A2_TOMOGRAPHY/tomography_vectors.npz
SHA256 d9f8d35a2f69b5a393a2a2ceb08d185d7a3b10e8b59be0905442b447b41f1706
```

For the realized pair,

```text
z_mid = (z_plus + z_minus)/2
d_eff = (z_plus - z_minus)/2
```

and hence the two realized files are exactly symmetric around `z_mid`, even
though clipping makes them asymmetric around the prescribed un-clipped
`z_base`. Frozen full-TEST geometry:

```text
RMS(d_eff)                 = 0.024930207761303152
G_eff = mean(d_eff^2)      = 0.0006215152590217399
RMS(z_mid-z_base)          = 0.0024656910196435397
max_abs(z_mid-z_base)      = 0.17332565983892978
materially changed rows    = 1,476
RMS(d_eff-d_probe)         = 0.0014413007562938745
corr(d_eff,d_probe)        = 0.9983373899777038
```

## Preregistered decoding equations

If clipping had been negligible, the requested affine decode would have been

```text
b       = (S_minus^2 - S_plus^2)/4
R^2     = (S_plus^2 + S_minus^2)/2 - 0.000625
a_star  = b / 0.000625
gain_MSE = b^2 / 0.000625
```

The clipping audit rejects that as an exact decode. The preregistered
clipping-aware decode instead uses the exact realized pair above. Let

```text
r_mid = y_log - z_mid
b_eff = mean(r_mid*d_eff)
G_eff = mean(d_eff^2)
```

Then

```text
S_plus^2  = R_mid^2 + G_eff - 2*b_eff
S_minus^2 = R_mid^2 + G_eff + 2*b_eff

b_eff        = (S_minus^2 - S_plus^2)/4
R_mid^2      = (S_plus^2 + S_minus^2)/2 - G_eff
a_star_eff   = b_eff / G_eff
gain_MSE_eff = b_eff^2 / G_eff
```

with frozen full-TEST `G_eff = 0.0006215152590217399`. This is the exact
symmetric quadratic for the two actual clipped vectors; after the two LB scores
arrive, no direction, sign, threshold, or probe scale will be changed.

Because `z_mid != z_base` on the clipped rows, two scores identify the midpoint
score and effective clipped-axis covariance exactly under this geometry, not the
score of the un-clipped `z_base` without an additional target moment. The report
will keep `R_mid` and any numerically reconstructed/approximated `R_base`
explicitly separate rather than silently treating the non-affine correction as
affine.

## Probe files

Both files have 250,000 unique users in exact sample order, columns
`user_id,predict`, finite nonnegative predictions, and serialization error below
`5e-11` in log space. They were created together and were not uploaded.

```text
PLUS
path   submissions/SUBMIT_EXP088_TOMO_PLUS.csv
SHA256 25bc4973a333bd5e428b1448c673d6eaae112a32953abd8f0f611272375ef7bc
zeros  915

MINUS
path   submissions/SUBMIT_EXP088_TOMO_MINUS.csv
SHA256 ce4ea2521f56b4302fc58a3d06d05aeb3a76421fb0cf0075ae9fd72258f85e75
zeros  1009
```

Reproduction script: `research/new_directions/EXP088_A1_A2_TOMOGRAPHY/run_exp088.py`.
Machine-readable audit: `research/new_directions/EXP088_A1_A2_TOMOGRAPHY/audit.json`.
