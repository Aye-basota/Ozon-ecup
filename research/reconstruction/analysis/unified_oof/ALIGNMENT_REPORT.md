# Alignment report

Canonical key is `(cutoff, user_id)` for OOF and `user_id` for TEST. No positional merge was used.

Canonical rows: **770,616**; unique keys: **770,616**; unique users: **210,212**; folds: `2025-09-04, 2025-09-18, 2025-10-02, 2025-10-16`.

| source | rows | matched | missing | extras | duplicates | target equal | original order equal after canonical sort | status |
|---|---:|---:|---:|---:|---:|---|---|---|
| CAP | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |
| UNC | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |
| DIST | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |
| ETX | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |
| SEQ | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |
| BTYD_STABLE_EXP051 | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |
| fallback_occ_lgbm_residual | 770616 | 770616 | 0 | 0 | 0 | True | True | PASS |

EXP-037 primitive replay max absolute error: `4.530e-07`.
Latest TEST recipe replay max absolute log error: `8.882e-16`.

Teammate `occ_meta_B`, `occ_raw_X3`, and `latest` have exact TEST keys (250,000/250,000) but no OOF rows; they are not treated as aligned OOF sources.
