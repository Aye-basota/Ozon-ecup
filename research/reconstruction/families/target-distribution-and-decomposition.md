# target_distribution_and_decomposition

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 18
- Positive / negative / inconclusive: 3 / 12 / 3
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7489187507745048,
      "experiment_id": "team_a_current:RUN-S04-LGB",
      "n": 1
    },
    "team_a_current:single_fold_protocol": {
      "score": 1.7459117188603979,
      "experiment_id": "team_a_current:EXP-038",
      "n": 2
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.708564,
      "experiment_id": "team_b_core:EXP-025",
      "n": 9
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.7489187507745048,
    "team_a_current:single_fold_protocol": 1.7462506955846235,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.71691
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-014 | 2026-08-11 | accepted | mean=1.75834; calibrated_OOF=1.75744; ensemble_OOF=1.75645 | `team_a_current:exp-014:comparability_unconfirmed` |
| team_a_s2:EXP-009 | 2026-08-11 | accepted_technical | not_applicable; hybrid verify max_abs_error=0.0054901898 | `team_a_s2:simulation_or_analytic_check` |
| team_a_s2:EXP-010 | 2026-08-11 | accepted_single_fold | Poisson=1.76579436; hurdle=1.75749468 | `team_a_s2:single_fold_protocol` |
| team_a_current:RUN-S04-LGB | 2026-08-13 | MACHINE_RESULT_SUBMISSIONS_CREATED | 1.7489187507745048 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_core:EXP-006 | 2026-08-13 | rejected | 2.045762 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-007 | 2026-08-13 | rejected | 2.39132 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-008 | 2026-08-13 | rejected | 1.732425 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-012 | 2026-08-13 | rejected_below_gate | 1.71691 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-022 | 2026-08-13 | rejected | 1.71155 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-025 | 2026-08-13 | rejected_below_gate | 1.708564 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-028 | 2026-08-13 | rejected | 1.716251 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-029 | 2026-08-13 | local_rejected_lb_positive_candidate | 1.716161 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-032 | 2026-08-13 | rejected | 1.720406 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_a_current:EXP-038 | 2026-08-21 | rejected | 1.7459117188603979 | `team_a_current:single_fold_protocol` |
| team_a_current:EXP-045 | 2026-08-22 | rejected | 1.7465896723088494 | `team_a_current:single_fold_protocol` |
| teammate_review:TM-TRAIN-HURDLE-FAST12 | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-HURDLE-STABLE18 | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| independent_anniversary:EXP-058 | 2026-08-25 | rejected | REAL-base mean=-0.0011394720; REAL-shuffled=-0.0006736373; REAL-shifted=+0.0003902915 | `independent_anniversary:pseudo_production_half_split` |
