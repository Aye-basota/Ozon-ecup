# Registry schema

The central table is `experiments.csv`; the JSONL sibling preserves lists and
nested FACTS/INTERPRETATION without CSV escaping ambiguity. Missing values are
the literal `unknown`, never an inferred zero.

Required core fields:

- `experiment_id`: globally namespaced canonical identifier;
- `canonical_name`, `family`, `date`, `parent_baseline`, `change`;
- `model_family`, `validation_protocol`, `comparison_class`;
- `cv_score`, `delta_cv`, `folds_positive`, `folds_total`, `lb_score`;
- `runtime`, `status`, `evidence_strength`, `artifacts`, `duplicate_of`;
- `compatible_tags`, `notes`.

Additional fields preserve train construction, features, target, folds, seeds,
hyperparameters, per-fold scores, submission, reproducibility, source report,
measured `facts`, author/agent `interpretation`, confounders, and conflicts.

One row is one primary experiment report or a separately evidenced run unit.
Duplicate documents and reruns remain as rows and point to a canonical unit with
`duplicate_of` and `relation_type`. Namespace prefixes prevent unrelated branch
experiments with the same local ID from colliding.

Supporting tables:

- `run_metrics.csv/jsonl`: granular arms, folds, sweeps, and machine run records;
- `components.csv`: OOF/test/model components and pairability;
- `../baselines/chronology.csv`: baselines within each comparable research line;
- `family_summary.csv`: aggregates that never pool incompatible score classes;
- `../ensembles/ancestry_edges.csv`: directed lineage and composition edges.
