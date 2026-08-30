# domain_shift_unlabeled_and_dataset_identity

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 2
- Positive / negative / inconclusive: 0 / 2 / 0
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:single_fold_protocol": {
      "score": 1.7422688089459835,
      "experiment_id": "team_a_current:EXP-056",
      "n": 1
    }
  },
  "median": {
    "team_a_current:single_fold_protocol": 1.7422688089459835
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| independent_calendar:EXP-029 | 2026-08-13 | rejected_stop_calendar | historical AUC means 7d=.52219,28d=.55294,56d=.58186,105d=.62339; real120d=.64434668 | `independent_calendar:exp-029:comparability_unconfirmed` |
| team_a_current:EXP-056 | 2026-08-24 | rejected | 1.7422688089459835 | `team_a_current:single_fold_protocol` |
