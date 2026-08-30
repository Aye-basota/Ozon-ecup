# train_example_construction

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 7
- Positive / negative / inconclusive: 1 / 6 / 0
- Saturation evidence: several_negative_results; evidence_is_implementation_and_protocol_specific

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7523386485160688,
      "experiment_id": "team_a_current:EXP-022",
      "n": 1
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.71704,
      "experiment_id": "team_b_core:EXP-005",
      "n": 2
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.7523386485160688,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.71704
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-002 | 2026-08-10 | rejected | 1.76981 | `team_a_current:exp-002:comparability_unconfirmed` |
| team_a_current:EXP-003 | 2026-08-10 | accepted | mean=1.76182; OOF=1.76165 | `team_a_current:exp-003:comparability_unconfirmed` |
| team_a_current:EXP-004 | 2026-08-10 | mixed_negative | E03a=1.76787; E03b=1.76999; E03c=1.79931; E04_two_fold=1.75705 | `team_a_current:exp-004:comparability_unconfirmed` |
| team_a_current:EXP-015 | 2026-08-11 | rejected | incomparable_unknown | `team_a_current:no_comparable_cv` |
| team_a_current:EXP-022 | 2026-08-12 | rejected | 1.7523386485160688 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_core:EXP-005 | 2026-08-13 | rejected | 1.71704 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-031 | 2026-08-13 | rejected_semantic_rerun | 1.71704 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
