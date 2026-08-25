# Prediction artifact standard

All predictions are stored in log space as `z_pred = log1p(prediction)`.

OOF schema:

```text
cutoff   string
user_id  int64
y_true   float64
z_pred   float64
```

TEST schema:

```text
user_id  int64
z_pred   float64
```

OOF row key is `(cutoff, user_id)`; TEST row key is `user_id`. Writers in
`src.utils.artifacts` reject duplicate keys and non-finite values. Filenames are:

```text
artifacts/oof/EXP_XXX_NAME.parquet
artifacts/test/EXP_XXX_NAME.parquet
```

Generated predictions are ignored by Git except for the reviewed frozen
`EXP_037_STRONGEST_CURRENT` OOF. External historical sources are indexed in
`artifacts/manifests/prediction_sources.csv`.
