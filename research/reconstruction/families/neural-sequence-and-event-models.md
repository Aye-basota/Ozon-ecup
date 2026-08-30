# neural_sequence_and_event_models

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 10
- Positive / negative / inconclusive: 6 / 4 / 0
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": {
      "score": 1.7477494758488472,
      "experiment_id": "team_a_current:EXP-039",
      "n": 1
    },
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7475192976531704,
      "experiment_id": "team_a_current:EXP-036",
      "n": 3
    },
    "team_a_current:group_a_half_panel_4fold": {
      "score": 1.7451442623457651,
      "experiment_id": "team_a_current:EXP-032B",
      "n": 2
    },
    "team_a_current:single_fold_protocol": {
      "score": 1.7491325780358982,
      "experiment_id": "team_a_current:EXP-029",
      "n": 2
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_nested_lofo": 1.7477494758488472,
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.7483427844623738,
    "team_a_current:group_a_half_panel_4fold": 1.7461783623656042,
    "team_a_current:single_fold_protocol": 1.7564024308251278
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_a_current:EXP-025 | 2026-08-13 | accepted_component | 1.7483427844623738 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-029 | 2026-08-14 | rejected | 1.7491325780358982 | `team_a_current:single_fold_protocol` |
| team_a_current:EXP-030 | 2026-08-18 | continued_to_multiseed | 1.7528404114727858 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-030B | 2026-08-19 | continued_to_multiseed | 1.7636722836143575 | `team_a_current:single_fold_protocol` |
| team_a_current:EXP-030C | 2026-08-19 | accepted_component_candidate | D3A wCV by seed=1.75284,1.75048,1.75163; base=1.75361,1.75148,1.75270 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-032 | 2026-08-19 | continued | 1.7472124623854435 | `team_a_current:group_a_half_panel_4fold` |
| team_a_current:EXP-032B | 2026-08-19 | gate_pass_no_ensemble_gain | 1.7451442623457651 | `team_a_current:group_a_half_panel_4fold` |
| team_a_current:EXP-036 | 2026-08-20 | continued_as_coauthor_rejected_as_replacement | 1.7475192976531704 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-039 | 2026-08-21 | rejected | 1.7477494758488472 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| team_a_current:EXP-044 | 2026-08-22 | rejected_below_gate | Fresh-VOL deltas [-0.000107510,-0.000015188,-0.000141583]; mean=-0.000088094; Fresh-base mean=+0.000052355 | `team_a_current:single_fold_protocol` |
