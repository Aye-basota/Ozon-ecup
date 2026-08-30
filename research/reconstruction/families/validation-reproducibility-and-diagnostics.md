# validation_reproducibility_and_diagnostics

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 16
- Positive / negative / inconclusive: 3 / 8 / 5
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:single_fold_protocol": {
      "score": 1.74125604941532,
      "experiment_id": "team_a_current:EXP-058",
      "n": 2
    },
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": {
      "score": 1.707779,
      "experiment_id": "team_b_core:EXP-023",
      "n": 1
    }
  },
  "median": {
    "team_a_current:single_fold_protocol": 1.743542958612784,
    "team_b_core:two_fold_decision_mean_with_fold3_diagnostic": 1.707779
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-016 | 2026-08-11 | diagnostic_negative_lb | candidate_wCV=1.74911; baseline_wCV=1.74948 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_b_alt:EXP-007 | 2026-08-11 | validation_audit | mean=1.729343; std=0.015957 | `team_b_alt:four_fold_equal_weight_temporal` |
| independent_domain:EXP-028 | 2026-08-13 | rejected_stop | ordinary candidate wCV=1.7511635266 vs1.7510761354; weighted=1.7368100387 vs1.7367245197 | `independent_domain:exp-028:comparability_unconfirmed` |
| team_a_current:EXP-027 | 2026-08-13 | accepted_diagnostic | clip289_LOFO_delta=-0.00055 vs sent champion | `team_a_current:diagnostic_stress_not_ordinary_cv` |
| team_a_current:EXP-028 | 2026-08-13 | rejected_before_training | unknown | `team_a_current:preflight_or_auxiliary_metric` |
| team_b_alt:EXP-012 | 2026-08-13 | accepted_validation_audit | scale1.0=1.720075; scale1.2=1.709007; scale1.4=1.709215 | `team_b_alt:two_fold_temporal_rmsle` |
| team_b_alt:EXP-013 | 2026-08-13 | rejected_lb_transfer | best scale1.300=1.707984; scale1.275=1.708010; baseline1.2=1.709007 | `team_b_alt:two_fold_temporal_rmsle` |
| team_b_core:EXP-023 | 2026-08-13 | rejected_below_gate | 1.707779 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-024 | 2026-08-13 | rejected_below_gate | best XGBoost blend=1.708145; segmented rec=1.708621; w365 decile=1.708509; baseline=1.708737 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| team_b_core:EXP-030 | 2026-08-13 | lb_diagnostic_positive | incomparable_unknown | `team_b_core:no_comparable_cv` |
| team_a_current:EXP-043 | 2026-08-22 | technical_pass | 1.7458298678102477 | `team_a_current:single_fold_protocol` |
| team_a_current:EXP-048 | 2026-08-23 | technical_inconclusive | incomparable; no canonical single score | `team_a_current:invalid_mixed_fold_comparison` |
| team_a_current:EXP-049 | 2026-08-23 | rejected_for_production_support | standard3 delta=-0.000547; pseudo-matched delta=-0.000551 | `team_a_current:three_fold_protocol` |
| team_a_current:EXP-050 | 2026-08-23 | blocked | reuse_only; no new score | `team_a_current:no_comparable_cv` |
| team_a_current:EXP-058 | 2026-08-24 | rejected | 1.74125604941532 | `team_a_current:single_fold_protocol` |
| team_a_current:EXP-032-MANIFEST | unknown | duplicate_manifest | duplicated summaries only; no independent run | `team_a_current:exp-032-manifest:comparability_unconfirmed` |
