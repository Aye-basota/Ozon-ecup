# temporal_history_and_calendar

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 9
- Positive / negative / inconclusive: 1 / 8 / 0
- Saturation evidence: several_negative_results; evidence_is_implementation_and_protocol_specific

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7504502529083472,
      "experiment_id": "team_a_current:EXP-021",
      "n": 1
    },
    "team_b_alt:single_fold_protocol": {
      "score": 1.710617,
      "experiment_id": "team_b_alt:EXP-004",
      "n": 2
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.708532,
      "experiment_id": "team_b_core:EXP-026",
      "n": 5
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.7504502529083472,
    "team_b_alt:single_fold_protocol": 1.7112365,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.717011
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_b_alt:EXP-003 | 2026-08-11 | rejected | 1.711856 | `team_b_alt:single_fold_protocol` |
| team_b_alt:EXP-004 | 2026-08-11 | accepted_single_fold | 1.710617 | `team_b_alt:single_fold_protocol` |
| team_a_current:EXP-019 | 2026-08-12 | rejected | E10 1.751415 to 1.756366 across stress axis | `team_a_current:diagnostic_stress_not_ordinary_cv` |
| team_a_current:EXP-021 | 2026-08-12 | rejected | 1.7504502529083472 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_core:EXP-002 | 2026-08-13 | rejected | 1.717011 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-003 | 2026-08-13 | rejected | 1.717017 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-004 | 2026-08-13 | rejected | 1.716725 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-011 | 2026-08-13 | rejected | 1.717095 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-026 | 2026-08-13 | rejected_below_gate | 1.708532 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
