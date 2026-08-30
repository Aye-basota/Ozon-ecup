# behavioral_occurrence_and_btyd

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 15
- Positive / negative / inconclusive: 1 / 4 / 10
- Saturation evidence: several_negative_results; evidence_is_implementation_and_protocol_specific

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": {
      "score": 1.7472406781889451,
      "experiment_id": "team_a_current:EXP-047",
      "n": 3
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": 1.7472406805088336
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-013 | 2026-08-10 | open_mixed | mean=1.75893; calibrated_OOF=1.75792 | `team_a_current:exp-013:comparability_unconfirmed` |
| independent_renewal:EXP-027 | 2026-08-13 | rejected_stop | CLOCK meta calibrated wCV=1.7480763705; standalone CLOCK two-part=1.7596937417 | `independent_renewal:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-047 | 2026-08-23 | rejected_below_research_gate | 1.7472406781889451 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| team_a_current:EXP-051 | 2026-08-23 | production_pass_not_uploaded | 1.7472406805088336 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| teammate_review:TM-RUN-FINAL6H-20260823 | 2026-08-23 | completed_no_lb | unknown | `teammate_review:walk_forward_4fold_wcv_1_2_4_8` |
| teammate_review:TM-TRAIN-OCC-R10-FAST | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R12-WIDE | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R14-MULTISCALE | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R16-BAL | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R18-WIDE | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R20-SHALLOW | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R22-STABLE | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| teammate_review:TM-TRAIN-OCC-R24-MULTISCALE | 2026-08-23 | completed | unknown | `teammate_review:training_unit_no_standalone_metric` |
| team_a_current:EXP-063 | 2026-08-25 | rejected | 1.7475203601061626 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| independent_global_regime:EXP-057 | unknown | rejected | final_GLOBAL=1.7474862831; final_BASE=1.7474288101; friend=1.7475098625 | `independent_global_regime:calibrated_temporal_4fold_wcv_1_2_4_8` |
