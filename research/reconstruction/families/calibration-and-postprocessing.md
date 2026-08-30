# calibration_and_postprocessing

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 13
- Positive / negative / inconclusive: 3 / 9 / 1
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": {
      "score": 1.747485106715659,
      "experiment_id": "team_a_current:EXP-042",
      "n": 1
    },
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7495766939331436,
      "experiment_id": "team_a_current:EXP-023",
      "n": 1
    },
    "team_a_current:single_fold_protocol": {
      "score": 1.7412947349972308,
      "experiment_id": "team_a_current:EXP-057",
      "n": 2
    },
    "team_b_alt:single_fold_protocol": {
      "score": 1.672748,
      "experiment_id": "team_b_alt:EXP-005",
      "n": 1
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.708632,
      "experiment_id": "team_b_core:EXP-017",
      "n": 5
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": 1.747485106715659,
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.7495766939331436,
    "team_a_current:single_fold_protocol": 1.744402298681533,
    "team_b_alt:single_fold_protocol": 1.672748,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.708737
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_s2:EXP-011 | 2026-08-11 | accepted_with_selection_override | K1=1.76950379; K2=1.76703708; K3=1.76581759; K5=1.76492425; K8=1.76484592; K15=1.76561190; nested_sigma_final=1.76481286 | `team_a_s2:exp-011:comparability_unconfirmed` |
| team_a_s2:EXP-012 | 2026-08-11 | accepted_standalone | mean=1.7683053375; OOF=1.76817 reported | `team_a_s2:structural_four_fold_rmsle` |
| team_b_alt:EXP-005 | 2026-08-11 | accepted_single_fold_calibration | 1.672748 | `team_b_alt:single_fold_protocol` |
| team_a_current:EXP-023 | 2026-08-12 | prepared_high_risk_not_scored | 1.7495766939331436 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_core:EXP-015 | 2026-08-13 | local_accept_lb_reject | 1.708737 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-016 | 2026-08-13 | rejected | 1.708734 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-017 | 2026-08-13 | rejected_below_gate | 1.708632 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-018 | 2026-08-13 | rejected | 1.708783 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-019 | 2026-08-13 | rejected | 1.709231 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_a_current:EXP-042 | 2026-08-21 | rejected_local_submitted_by_override | 1.747485106715659 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| team_a_current:EXP-057 | 2026-08-24 | rejected | 1.7412947349972308 | `team_a_current:single_fold_protocol` |
| team_a_current:EXP-060 | 2026-08-24 | prepared_not_uploaded | unknown | `team_a_current:no_comparable_cv` |
| team_a_current:EXP-062 | 2026-08-25 | rejected_preflight | 1.7475098623658354 | `team_a_current:single_fold_protocol` |
