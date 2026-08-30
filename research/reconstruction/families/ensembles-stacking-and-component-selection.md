# ensembles_stacking_and_component_selection

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 21
- Positive / negative / inconclusive: 8 / 10 / 3
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": {
      "score": 1.747284906303239,
      "experiment_id": "team_a_current:EXP-040",
      "n": 1
    },
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7477422600193706,
      "experiment_id": "team_a_current:EXP-026",
      "n": 3
    },
    "team_a_current:informational_4fold_not_decision_metric": {
      "score": 1.7472719649885926,
      "experiment_id": "team_a_current:EXP-059",
      "n": 1
    },
    "team_a_current:preflight_or_auxiliary_metric": {
      "score": 1.7412785664479717,
      "experiment_id": "team_a_current:EXP-052",
      "n": 1
    },
    "team_b_alt:single_fold_protocol": {
      "score": 1.671639,
      "experiment_id": "team_b_alt:EXP-008",
      "n": 1
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.716366,
      "experiment_id": "team_b_core:EXP-013",
      "n": 2
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": 1.747284906303239,
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.7505691416172469,
    "team_a_current:informational_4fold_not_decision_metric": 1.7472719649885926,
    "team_a_current:preflight_or_auxiliary_metric": 1.7412785664479717,
    "team_b_alt:single_fold_protocol": 1.671639,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.7164395
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-006 | 2026-08-10 | accepted | mean=1.75886; calibrated_OOF=1.75716 | `team_a_current:exp-006:comparability_unconfirmed` |
| team_a_current:EXP-008 | 2026-08-10 | rejected | standalone_mean=1.85169; standalone_calibrated_OOF=1.82105; blend_OOF=1.77210 | `team_a_current:exp-008:comparability_unconfirmed` |
| team_b_alt:EXP-008 | 2026-08-11 | accepted_lb_report_only | 1.671639 | `team_b_alt:single_fold_protocol` |
| team_a_current:EXP-018 | 2026-08-12 | accepted_development | AVG3=1.75046; AVG5=1.75037 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-020 | 2026-08-12 | rejected | 1.7505691416172469 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_alt:EXP-009 | 2026-08-12 | accepted_lb_report_only | long_buy=1.671832; ensemble=1.670716 | `team_b_alt:single_fold_protocol` |
| team_b_alt:EXP-010 | 2026-08-12 | neutral_weight_sweep | baseline/best=1.670716; w0.55,s1.00=1.670717; w0.55,s1.01=1.670731 | `team_b_alt:single_fold_protocol` |
| team_a_current:EXP-024 | 2026-08-13 | rejected | 1.7523448608080778 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-026 | 2026-08-13 | accepted_technique_stale_production_conclusion | 1.7477422600193706 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_alt:EXP-011 | 2026-08-13 | accepted_lb_report_only | scale1.0=1.733432; scale1.4=1.712473 | `team_b_alt:single_fold_protocol` |
| team_b_core:EXP-013 | 2026-08-13 | rejected_below_gate | 1.716366 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-014 | 2026-08-13 | rejected_below_gate | 1.716513 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-034 | 2026-08-13 | rejected | temporal2F=1.721106; central2F=1.716842 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_alt:EXP-014 | 2026-08-14 | rejected | baseline=1.709007; t0.05=1.709007; t0.08=1.709078; t0.20=1.716274 | `team_b_alt:two_fold_temporal_rmsle` |
| team_b_alt:EXP-015 | 2026-08-14 | rejected_lb_qualitative | w0.525=1.709004; baseline w0.5=1.709007 | `team_b_alt:two_fold_temporal_rmsle` |
| team_b_alt:EXP-017 | 2026-08-15 | accepted_lb_report_only | candidate=1.708295; baseline=1.708883 | `team_b_alt:two_fold_temporal_rmsle` |
| team_a_current:EXP-040 | 2026-08-21 | rejected_below_gate | 1.747284906303239 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| teammate_review:TM-RUN-EXTRA90-20260823 | 2026-08-23 | completed_no_lb | unknown | `teammate_review:walk_forward_4fold_wcv_1_2_4_8` |
| teammate_review:TM-RUN-FIXEDSTACK-20260823 | 2026-08-23 | completed_no_lb | unknown | `teammate_review:walk_forward_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-052 | 2026-08-24 | rejected | 1.7412785664479717 | `team_a_current:preflight_or_auxiliary_metric` |
| team_a_current:EXP-059 | 2026-08-24 | prepared_not_uploaded | 1.7472719649885926 | `team_a_current:informational_4fold_not_decision_metric` |
