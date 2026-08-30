# Registries

Start with `experiments.csv` for analysis or `experiments.jsonl` when nested
FACTS/INTERPRETATION must remain lossless. Field definitions are in `SCHEMA.md`.

Supporting registries:

- `run_metrics.csv/jsonl`: 1,134 granular run/manifest rows;
- `components.csv`: 614 prediction/model component groups;
- `family_summary.csv/jsonl`: protocol-safe family aggregates;
- `deduplication.csv/jsonl`: 11 experiment-level duplicate/rerun/reuse clusters;
- `id_collisions.csv/jsonl`: cross-namespace bare-ID collisions;
- `report_catalog.csv/jsonl`: 124 primary report snapshots;
- `teammate_package_links.csv/jsonl`: seven package copies linked to canonical
  Team A experiments instead of duplicated in the central table;
- `experiments_without_canonical_numeric_metric.csv`: explicit completeness
  audit, including runtime-only and blocked units.

