# production_integration_and_provenance

This page is descriptive research memory, not a recommendation for future work.

- Experiments: 7
- Positive / negative / inconclusive: 3 / 0 / 4
- Saturation evidence: mixed_or_insufficient_evidence

## Comparable score groups

```json
{
  "best": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": {
      "score": 1.7475098627238472,
      "experiment_id": "team_a_current:EXP-037",
      "n": 2
    }
  },
  "median": {
    "team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8": 1.74763881921891
  }
}
```

## Experiments

| Experiment | Date | Status | CV | Comparison class |
|---|---|---|---:|---|
| team_b_core:EXP-033 | 2026-08-13 | weak_lb_positive_vs_baseline | incomparable_unknown | `team_b_core:public_lb_only_no_local_cv` |
| team_a_current:EXP-035 | 2026-08-19 | accepted_candidate_not_sent | 1.747767775713973 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-037 | 2026-08-20 | accepted | 1.7475098627238472 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| team_a_current:EXP-065 | 2026-08-25 | accepted_final_package_not_uploaded_here | A=1.747509862; B delta=-0.000321 with 4/4; test ratio=1.1734 | `team_a_current:no_comparable_cv` |
| team_a_current:EXP-066 | 2026-08-25 | blocked_no_canonical_latest_oof | unknown | `team_a_current:no_comparable_cv` |
| team_a_current:EXP-067 | 2026-08-25 | provenance_incomplete_external_lb | unknown/incomparable | `team_a_current:exp-067:comparability_unconfirmed` |
| team_a_current:EXP-068 | 2026-08-25 | blocked_historical_replay | unknown | `team_a_current:no_comparable_cv` |
