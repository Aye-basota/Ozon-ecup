# Canonical baselines

`EXP_037_STRONGEST_CURRENT` is the strongest frozen, fold-safe, reproducible OOF
baseline found in the audited repository. It is deliberately separate from the
current TEST-only public incumbent.

The canonical Parquet stores raw log predictions before fold calibration:

```text
cutoff: string
user_id: int64
y_true: float64
z_pred: float64
```

Evaluation must use `src.validation.evaluate.evaluate_oof`. Do not refit offsets
globally, change fold weights, or reconstruct an OOF for the geometry incumbent.
Full provenance and source hashes are in
`baselines/manifests/EXP_037_STRONGEST_CURRENT.json` and
`registry/source_manifest.csv`.
