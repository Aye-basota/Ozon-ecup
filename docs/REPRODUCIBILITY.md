# Reproducibility guide

## Reproduction levels

1. **Frozen exact rebuild** — committed component predictions are aligned and
   blended. This is fast and must be byte-identical for both finals.
2. **Audited raw/tabular rebuild** — raw parquet is checked by SHA, features are
   rebuilt and tabular/Team-B models are retrained. Historical STRONGEST booster
   weights were not retained, so its retrained result is compared numerically.
3. **Full neural replay** — exact SEQ/ETX commands and five available checkpoints
   are preserved in the STRONGEST package. It is multi-hour/GPU-sensitive and
   is not the default audit; the SEQ-01 seed-42 checkpoint is missing.

## Fast exact commands

```powershell
python scripts/reproduce_final.py --solution SUBMIT_STRONGEST55_TEAMB45 --from-precomputed
python scripts/reproduce_final.py --solution SUBMIT_JOINT86_TEAMB14 --from-precomputed
```

Expected results:

| Solution | EXPECTED_SHA256 | BYTE_IDENTICAL | Expected max abs diff | Expected RMS log diff |
|---|---|---|---:|---:|
| STRONGEST55/TEAMB45 | `1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4` | YES | 0 | 0 |
| JOINT86/TEAMB14 | `85d9cd645e14a7895da9ad8cc89065714606266be588c762d37487d2b4edac02` | YES | 0 | 0 |

## Raw data contract

`data/raw/train.parquet` must have SHA256
`5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0`.
`sample_submit.csv` must contain exactly 250,000 unique `user_id` values and the
columns `user_id,predict` in competition order.

## Raw mode and known limits

The common `--from-raw` entry point checks raw SHA before work begins. The
STRONGEST path retrains three tabular members and Team-B, while using frozen
SEQ/ETX for the bounded audit. The saved 2026-08-30 audit measured:

- Team-B: byte-identical, SHA `4ed2916b…44aba`;
- fresh STRONGEST versus production: RMS log difference 0.0272457903;
- fresh final STRONGEST55/TEAMB45 versus reference: RMS log difference
  0.0149851847.

The JOINT raw path retrains Team-B but necessarily retains frozen JOINT_V2.
Without the missing JOINT_V2 generator, claiming a raw-to-JOINT exact replay
would be false.

## Smoke checks

The audit enforces imports, columns, row count, unique IDs, sample order,
finite/nonnegative predictions and SHA256. The preserved neural smoke report
records 114 tests passed, one skipped, exact SEQ-43/44 checkpoint inference and
ETX error within the 0.05 log tolerance on a 512-row subset.

Package-level details:

- `reproducibility/SUBMIT_STRONGEST55_TEAMB45/AUDIT_RESULTS.md`
- `reproducibility/SUBMIT_STRONGEST55_TEAMB45/COMPONENT_MANIFEST.json`
- `reproducibility/SUBMIT_JOINT86_TEAMB14/COMPONENT_MANIFEST.json`
