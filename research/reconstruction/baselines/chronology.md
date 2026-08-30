# Baseline chronology

Chronology is separated by research line. Repeated textual baselines are retained because a name alone does not prove identical folds, train coverage, or prediction sources.

| First seen | Research line | Baseline or recipe | First reference | Comparison class |
|---|---|---|---|---|
| 2026-08-10 | team_a_current | EXP-001 S1-B0 | team_a_current:EXP-002 | `team_a_current:exp-002:comparability_unconfirmed` |
| 2026-08-10 | team_a_current | EXP-003 S1-E02 | team_a_current:EXP-004 | `team_a_current:exp-004:comparability_unconfirmed` |
| 2026-08-10 | team_a_current | EXP-005 S1-E10 direct; secondary EXP-006 S1-BEST | team_a_current:EXP-013 | `team_a_current:exp-013:comparability_unconfirmed` |
| 2026-08-10 | team_a_current | EXP-006 S1-BEST | team_a_current:EXP-007 | `team_a_current:exp-007:comparability_unconfirmed` |
| 2026-08-10 | team_a_current | team_a_current:EXP-001 | team_a_current:EXP-001 | `team_a_current:exp-001:comparability_unconfirmed` |
| 2026-08-11 | team_a_current | EXP-005 S1-E10; ensemble baseline EXP-006 | team_a_current:EXP-014 | `team_a_current:exp-014:comparability_unconfirmed` |
| 2026-08-11 | team_a_current | EXP-014 DIST mix | team_a_current:EXP-016 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-11 | team_a_current | EXP-014 distribution-head production recipe | team_a_current:EXP-015 | `team_a_current:no_comparable_cv` |
| 2026-08-11 | team_a_s2 | S1-BEST benchmark and prior S2 screens | team_a_s2:EXP-012 | `team_a_s2:structural_four_fold_rmsle` |
| 2026-08-11 | team_a_s2 | S2 hurdle with K=3 | team_a_s2:EXP-011 | `team_a_s2:exp-011:comparability_unconfirmed` |
| 2026-08-11 | team_a_s2 | plain/offset Poisson structural count | team_a_s2:EXP-010 | `team_a_s2:single_fold_protocol` |
| 2026-08-11 | team_a_s2 | pure Fenton-Wilkinson aggregation | team_a_s2:EXP-009 | `team_a_s2:simulation_or_analytic_check` |
| 2026-08-11 | team_b_alt | team_b_alt:EXP-001 | team_b_alt:EXP-001 | `team_b_alt:single_fold_protocol` |
| 2026-08-11 | team_b_alt | team_b_alt:EXP-001 HGBR | team_b_alt:EXP-006 | `team_b_alt:single_fold_protocol` |
| 2026-08-11 | team_b_alt | team_b_alt:EXP-001; components EXP-004/005/006 | team_b_alt:EXP-008 | `team_b_alt:single_fold_protocol` |
| 2026-08-11 | team_b_alt | team_b_alt:EXP-006 | team_b_alt:EXP-006 | `team_b_alt:single_fold_protocol` |
| 2026-08-12 | team_a_current | E10 direct at 600 rounds | team_a_current:EXP-017 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-12 | team_a_current | E10/E03a/E02/DIST at standard gap | team_a_current:EXP-019 | `team_a_current:diagnostic_stress_not_ordinary_cv` |
| 2026-08-12 | team_a_current | EXP-017 300-round model | team_a_current:EXP-022 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-12 | team_a_current | EXP-017 E10 at 300 rounds seed42 | team_a_current:EXP-018 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-12 | team_a_current | EXP-018 AVG3 | team_a_current:EXP-020 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-12 | team_b_alt | team_b_alt:EXP-008 | team_b_alt:EXP-009 | `team_b_alt:single_fold_protocol` |
| 2026-08-12 | team_b_alt | team_b_alt:EXP-009 | team_b_alt:EXP-010 | `team_b_alt:single_fold_protocol` |
| 2026-08-12 | team_b_core | naive30/naive90 rules | team_b_core:EXP-001 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-12 | team_b_core | team_b_core:EXP-001 | team_b_core:EXP-001 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | independent_calendar | DOMAIN-01 fixed-L180 real-test domain classifier | independent_calendar:EXP-029 | `independent_calendar:exp-029:comparability_unconfirmed` |
| 2026-08-13 | independent_domain | S1-ROUNDS direct and SEQ-01-MIX production ranking | independent_domain:EXP-028 | `independent_domain:exp-028:comparability_unconfirmed` |
| 2026-08-13 | independent_renewal | SEQ-01-MIX with existing b30 occurrence head | independent_renewal:EXP-027 | `independent_renewal:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-13 | team_a_current | EXP-014 DIST mix; internal MHZ base | team_a_current:EXP-024 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-13 | team_a_current | EXP-025 SEQ-01 and EXP-014 DIST mix | team_a_current:EXP-026 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-13 | team_a_current | EXP-025 SEQ-01 submission and EXP-026 SEQ-AVG3 submission | team_a_current:EXP-027 | `team_a_current:diagnostic_stress_not_ordinary_cv` |
| 2026-08-13 | team_a_current | S1-DIST-MIX | team_a_current:RUN-S04-LGB | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-13 | team_a_current | existing DIST training construction | team_a_current:EXP-028 | `team_a_current:preflight_or_auxiliary_metric` |
| 2026-08-13 | team_b_alt | team_b_alt:EXP-009 only for LB lineage; local regime differs | team_b_alt:EXP-011 | `team_b_alt:single_fold_protocol` |
| 2026-08-13 | team_b_alt | team_b_alt:EXP-011 prediction recipe | team_b_alt:EXP-012 | `team_b_alt:two_fold_temporal_rmsle` |
| 2026-08-13 | team_b_alt | team_b_alt:EXP-011 scale1.2 | team_b_alt:EXP-013 | `team_b_alt:two_fold_temporal_rmsle` |
| 2026-08-13 | team_b_core | rounded uncalibrated LB baseline and G_hurdle LB candidate | team_b_core:EXP-033 | `team_b_core:public_lb_only_no_local_cv` |
| 2026-08-13 | team_b_core | same-run central single-cutoff LightGBM | team_b_core:EXP-034 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-001 feature/model baseline; secondary team_b_core:EXP-015 | team_b_core:EXP-028 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-015 | team_b_core:EXP-016 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-016 | team_b_core:EXP-016 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-017 | team_b_core:EXP-017 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-018 | team_b_core:EXP-018 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-019 | team_b_core:EXP-019 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-028; secondary EXP-001/EXP-015 | team_b_core:EXP-029 | `team_b_core:two_fold_decision_mean_with_fold3_diagnostic` |
| 2026-08-13 | team_b_core | team_b_core:EXP-029 E_hurdle submission | team_b_core:EXP-030 | `team_b_core:no_comparable_cv` |
| 2026-08-13 | team_b_core | team_b_core:EXP-033 | team_b_core:EXP-033 | `team_b_core:public_lb_only_no_local_cv` |
| 2026-08-14 | team_a_current | fresh local sequence base on fold 2025-10-16 | team_a_current:EXP-029 | `team_a_current:single_fold_protocol` |
| 2026-08-14 | team_b_alt | team_b_alt:EXP-011 weights0.5/0.5 scale1.2 | team_b_alt:EXP-015 | `team_b_alt:two_fold_temporal_rmsle` |
| 2026-08-15 | team_b_alt | team_b_alt:EXP-011-like direct long-buy ensemble | team_b_alt:EXP-016 | `team_b_alt:two_fold_temporal_rmsle` |
| 2026-08-15 | team_b_alt | team_b_alt:EXP-016 direct post-order ensemble | team_b_alt:EXP-017 | `team_b_alt:two_fold_temporal_rmsle` |
| 2026-08-18 | team_a_current | paired sequence base seed42 | team_a_current:EXP-030 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-19 | team_a_current | paired base for seeds42-44 | team_a_current:EXP-030C | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-19 | team_a_current | paired base seed43 | team_a_current:EXP-030B | `team_a_current:single_fold_protocol` |
| 2026-08-19 | team_a_current | plain base and DIST-TAB/sequence ensemble slots on half-panel A | team_a_current:EXP-032B | `team_a_current:group_a_half_panel_4fold` |
| 2026-08-19 | team_a_current | plain sequence base on half-panel group A | team_a_current:EXP-032 | `team_a_current:group_a_half_panel_4fold` |
| 2026-08-19 | team_a_current | sent SEQ01 sequence slot | team_a_current:EXP-035 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-20 | team_a_current | D3A seed42 standalone and current sequence slot | team_a_current:EXP-036 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-20 | team_a_current | prior strongest ensemble/sequence slot | team_a_current:EXP-037 | `team_a_current:calibrated_temporal_4fold_wcv_1_2_4_8` |
| 2026-08-21 | team_a_current | EXP-037 strongest-current | team_a_current:EXP-039 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| 2026-08-21 | team_a_current | EXP-037 strongest-current exact wCV 1.747509863 | team_a_current:EXP-042 | `team_a_current:calibrated_temporal_4fold_nested_lofo` |
| 2026-08-21 | team_a_current | paired fold-2025-10-16 sequence base | team_a_current:EXP-038 | `team_a_current:single_fold_protocol` |
| 2026-08-22 | team_a_current | plain SEQ01 paired base on fold 2025-10-16 | team_a_current:EXP-044 | `team_a_current:single_fold_protocol` |
| 2026-08-22 | team_a_current | plain paired base on fold 2025-10-16 | team_a_current:EXP-045 | `team_a_current:single_fold_protocol` |
| 2026-08-22 | team_a_current | same configuration repeated | team_a_current:EXP-043 | `team_a_current:single_fold_protocol` |
| 2026-08-23 | team_a_current | BTYD05/FRESH candidates from EXP-047-049 | team_a_current:EXP-050 | `team_a_current:no_comparable_cv` |
| 2026-08-23 | team_a_current | candidate rankings from prior artifact-only searches | team_a_current:EXP-048 | `team_a_current:invalid_mixed_fold_comparison` |
| 2026-08-23 | team_a_current | standard 3-fold evaluation of BTYD05_FRESH1 | team_a_current:EXP-049 | `team_a_current:three_fold_protocol` |
| 2026-08-23 | teammate_review | STRONGEST_CURRENT + table_core | teammate_review:TM-RUN-FIXEDSTACK-20260823 | `teammate_review:walk_forward_4fold_wcv_1_2_4_8` |
| 2026-08-23 | teammate_review | TM-RUN-FINAL6H-20260823 | teammate_review:TM-RUN-EXTRA90-20260823 | `teammate_review:walk_forward_4fold_wcv_1_2_4_8` |
| 2026-08-23 | teammate_review | TM-RUN-FIXEDSTACK-20260823 | teammate_review:TM-RUN-FINAL6H-20260823 | `teammate_review:walk_forward_4fold_wcv_1_2_4_8` |
| 2026-08-24 | team_a_current | EXP-037 exact wCV 1.747509863 | team_a_current:EXP-053 | `team_a_current:exp-053:comparability_unconfirmed` |
| 2026-08-24 | team_a_current | EXP-037 strongest-current on fold 2025-10-16 | team_a_current:EXP-052 | `team_a_current:preflight_or_auxiliary_metric` |
| 2026-08-24 | team_a_current | UNC component and EXP-037 strongest-current on fold 2025-10-16 | team_a_current:EXP-058 | `team_a_current:single_fold_protocol` |
| 2026-08-24 | team_a_current | existing final prediction | team_a_current:EXP-060 | `team_a_current:no_comparable_cv` |
| 2026-08-24 | team_a_current | no-adapt/control on fold 2025-10-16 | team_a_current:EXP-056 | `team_a_current:single_fold_protocol` |
| 2026-08-25 | independent_anniversary | STRONGEST_CURRENT pseudo-production reconstruction | independent_anniversary:EXP-058 | `independent_anniversary:pseudo_production_half_split` |
| 2026-08-25 | team_a_current | external teammate pipeline | team_a_current:EXP-067 | `team_a_current:exp-067:comparability_unconfirmed` |
| 2026-08-25 | team_a_current | historical recency-ridge pipeline | team_a_current:EXP-068 | `team_a_current:no_comparable_cv` |
| 2026-08-25 | team_a_current | latest external/canonical solution sought | team_a_current:EXP-066 | `team_a_current:no_comparable_cv` |
| unknown | independent_global_regime | re-anchored table_core overlay; friend STRONGEST_CURRENT reference | independent_global_regime:EXP-057 | `independent_global_regime:calibrated_temporal_4fold_wcv_1_2_4_8` |
| unknown | team_a_current | same design as EXP-032/EXP-032B | team_a_current:EXP-032-MANIFEST | `team_a_current:exp-032-manifest:comparability_unconfirmed` |
