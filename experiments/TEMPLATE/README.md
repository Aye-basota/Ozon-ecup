# EXP_XXX_NAME

## Hypothesis

State one falsifiable claim.

## Baseline

`EXP_037_STRONGEST_CURRENT` — wCV `1.7475098625201952`.

## Exact change

Describe the single intended difference from the baseline or control.

## Validation

Canonical four folds, per-fold log offset, weights `1:2:4:8`, row key
`(cutoff, user_id)`. State any additional control or LOFO protocol.

## Success gate

Write the exact numeric/consistency gate before running. Define whether PASS can
come from standalone gain, incremental blend gain, or orthogonality evidence.

## Commands

```text
python scripts/run_tabular_experiment.py --config experiments/EXP_XXX_name/config.yaml
```

Add `--test` only after the OOF decision permits TEST inference.

## Outputs

- `metrics.json`
- `report.md`
- `artifacts/oof/EXP_XXX_NAME.parquet`
- `artifacts/test/EXP_XXX_NAME.parquet` if approved

## Result

Verdict: pending. Replace with exactly one of `PASS`, `WEAK_SIGNAL`, `REJECT`, or `INVALID`.
