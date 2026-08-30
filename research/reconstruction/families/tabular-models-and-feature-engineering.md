# tabular_models_and_feature_engineering

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 14
- Positive / negative / inconclusive: 5 / 6 / 3
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_b_alt:single_fold_protocol": {
      "score": 1.710143,
      "experiment_id": "team_b_alt:EXP-006",
      "n": 3
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.708309,
      "experiment_id": "team_b_core:EXP-021",
      "n": 5
    }
  },
  "median": {
    "team_b_alt:single_fold_protocol": 1.710919,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.716802
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-001 | 2026-08-10 | baseline | mean=1.76879; OOF=1.76861; calibrated_OOF=1.76570 | `team_a_current:exp-001:comparability_unconfirmed` |
| team_a_current:EXP-005 | 2026-08-10 | accepted_with_caveat | mean=1.75988; calibrated_OOF=1.75889 | `team_a_current:exp-005:comparability_unconfirmed` |
| team_a_current:EXP-007 | 2026-08-10 | rejected | mean=1.77335; calibrated_OOF=1.77065 | `team_a_current:exp-007:comparability_unconfirmed` |
| team_b_alt:EXP-001 | 2026-08-11 | baseline | 1.711195 | `team_b_alt:single_fold_protocol` |
| team_b_alt:EXP-002 | 2026-08-11 | accepted_single_fold | 1.710919 | `team_b_alt:single_fold_protocol` |
| team_b_alt:EXP-006 | 2026-08-11 | accepted_single_fold | 1.710143 | `team_b_alt:single_fold_protocol` |
| team_a_current:EXP-017 | 2026-08-12 | accepted_development | baseline600=1.75170; best200=1.75103; production300=1.75108; OOF_cal_200=1.75801 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_core:EXP-001 | 2026-08-12 | baseline | 1.717017 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-009 | 2026-08-13 | rejected_below_gate | 1.716802 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-010 | 2026-08-13 | rejected_below_gate | 1.716961 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-020 | 2026-08-13 | rejected_below_gate | 1.708634 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-021 | 2026-08-13 | rejected_below_gate | 1.708309 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_alt:EXP-016 | 2026-08-15 | accepted_lb_report_only | baseline=1.709007; candidate=1.708883 | `team_b_alt:two_fold_temporal_rmsle` |
| team_a_current:EXP-046 | 2026-08-23 | rejected | 1.7474573027361282 | `team_a_current:exp-046:comparability_unconfirmed` |
