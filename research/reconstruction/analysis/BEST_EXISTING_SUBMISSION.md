# Best existing-artifact submission

## Decision

If no new training runs are allowed, submit one locked compound assembled from existing primitive artifacts:

```text
COMPOUND_SEQ65_BTYD05 = 0.95 * SEQ65 + 0.05 * BTYD
```

The locked file has now been materialized as `analysis/BEST_EXISTING_SUBMISSION.csv` without fitting or LB-based choice. It contains `250,000` rows and has SHA256:

```text
4001d212aab708a5caef500fa7c6edf119d24969888fe45a0be2330347ba949f
```

The reproducible build manifest is `analysis/BEST_EXISTING_SUBMISSION.csv.manifest.json`; the realized one-shot global shift is `−0.109740343613`.

## Exact recipe

All operations are in `z = log1p(pred)` space.

### 1. Rebuild SEQ65 raw shape

```text
z_seq65_raw =
    0.10 * ztest_S1-CAP
  + 0.10 * ztest_S1-UNC
  + 0.15 * ztest_S1-DIST
  + (0.325 / 3) * ztest_SEQ-01
  + (0.325 / 3) * ztest_SEQ-C289-S43
  + (0.325 / 3) * ztest_SEQ-C289-S44
  + (0.325 / 3) * ztest_ETX-01-S42-DCW
  + (0.325 / 3) * ztest_ETX-01-S43-DCW
  + (0.325 / 3) * ztest_ETX-01-S44-DCW
```

For every component use its matching `uid_<name>.npy`; all must have identical UID SHA256 `50e5ba9b71a510b05126d5f325d9c63186ca09975680c66e4ee024e3e0fd576a` and exact sample order.

### 2. Load BTYD raw shape

From:

```text
C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\BTYD_STABLE_EXP051\test_raw.npz
```

use arrays `user_id` and `z_btyd`. Artifact SHA256:

```text
5222d26166c600ba201958937d7226ba535a49a1c7aeb2a8dc3b328b437e5a43
```

Require exact equality of BTYD `user_id` to the component UID array. Also verify stored `z_strongest` reproduces primitive `EXP-037` to `<=5e-7`.

### 3. Fixed compound

```text
z_raw = 0.95 * z_seq65_raw + 0.05 * z_btyd
shift = 2.3293 - mean(z_raw)
z_final = maximum(z_raw + shift, 0.0)
predict = expm1(z_final)
```

Write exactly two columns in sample order:

```text
user_id,predict
```

Do not:

- blend the already rounded submission CSVs and call that the same recipe;
- apply separate level shifts to SEQ65 and BTYD before the fixed blend;
- tune `0.65`, `0.05` or level after seeing LB;
- add ZERO2D or teammate latest.

## Offline evidence

| fold | STRONGEST | compound | delta |
|---|---:|---:|---:|
| 2025-09-04 | 1.766883360 | 1.765895441 | `−0.000987919` |
| 2025-09-18 | 1.760509577 | 1.759715031 | `−0.000794546` |
| 2025-10-02 | 1.748629224 | 1.748119421 | `−0.000509803` |
| 2025-10-16 | 1.741278566 | 1.740800534 | `−0.000478033` |
| **wCV 1:2:4:8** | **1.747509863** | **1.746947164** | **`−0.000562699`** |

Additional gates:

- correction Pearson between constituent gains: `−0.012633`;
- interaction vs arithmetic sum: `−0.000003818`;
- paired user-cluster SE: `0.000058635`;
- combined test/OOF correction variance ratio: `1.216712`;
- no public LB input was used.

## Why this one, not the alternatives

- `submission_BTYD05.csv` is safer but leaves the independent `EXP-059` gain unused.
- `submission_SEQ65_TEMPORAL_HEAVY.csv` leaves the stronger BTYD residual unused.
- FRESH triple is locally stronger (`−0.000721`) but has no exact production encoder/heads.
- teammate `latest.csv` has lower externally reported public LB, but canonical OOF is missing and selection/provenance risk is materially higher.
- ZERO2D has only `−0.000025` honest OOF and failed its mechanism controls.

## Expected leaderboard result

Likely gain vs `submission_STRONGEST_CURRENT.csv`: `−0.0002…−0.0005`; upside around `−0.0007`. The result must be treated as one locked measurement, not the first point of an LB weight search.
