# Start here

This repository is a neutral reconstruction of completed research. It contains
no recommendation for the next ML experiment.

1. Filter [`../registry/experiments.csv`](../registry/experiments.csv) by
   namespace, family, status, or `comparison_class`.
2. Open the corresponding card in [`../experiments/normalized/`](../experiments/normalized/)
   to read measured FACTS separately from the original INTERPRETATION.
3. Use [`../registry/run_metrics.csv`](../registry/run_metrics.csv) for fold,
   seed, arm, and sweep detail; use [`../registry/components.csv`](../registry/components.csv)
   for OOF/test/model artifacts.
4. Read [`../ensembles/SOLUTION_ANCESTRY.md`](../ensembles/SOLUTION_ANCESTRY.md)
   for parallel pipelines and composition ancestry.
5. Check [`../contradictions/registry.csv`](../contradictions/registry.csv) and
   [`../leaderboard/report_only_claims.csv`](../leaderboard/report_only_claims.csv)
   before quoting a verdict or LB value.

The audit of claims made by excluded old summaries is in
[`../evidence/secondary_summary_conflicts_audit.md`](../evidence/secondary_summary_conflicts_audit.md).
Those claims are comparison objects only and never experiment fact sources.

The authoritative navigation rules are simple: IDs must be namespaced; scores
may be compared only within one `comparison_class`; `unknown` is intentional;
and repository-internal LB links are not platform-independent verification.

For scope and completeness, read
[`REPOSITORY_RECONSTRUCTION.md`](REPOSITORY_RECONSTRUCTION.md).
