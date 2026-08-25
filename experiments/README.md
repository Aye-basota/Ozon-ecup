# Experiment protocol

Create one directory per experiment: `experiments/EXP_XXX_name/`. Start from
`experiments/TEMPLATE` and decide the hypothesis and success gate before a run.

Required files:

- `README.md`: hypothesis, baseline, exact change, validation, gate, commands, outputs, result;
- `config.yaml`: complete executable model/feature configuration;
- `metrics.json`: machine-readable result;
- `report.md`: concise decision record;
- `scripts/` or an explicit source reference.

The canonical comparison is `EXP_037_STRONGEST_CURRENT`, wCV
`1.7475098625201952`. A TEST prediction is generated only when the experiment
prompt's gate permits it.

Use exactly one final verdict:

- `PASS`: OOF gain, proven incremental blend gain, or a demonstrated useful orthogonal direction;
- `WEAK_SIGNAL`: consistent evidence below the experiment's materiality gate;
- `REJECT`: valid experiment that fails its gate;
- `INVALID`: leakage, row mismatch, wrong folds, execution failure, or otherwise unusable evidence.

After review, add one row to `registry/experiments.csv`. Do not rewrite historical
rows or use public LB to redefine offline validation.
