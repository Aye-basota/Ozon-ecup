# Forensic audit of exported branch experiment reports

## Scope and evidence boundary

This audit covers exactly the 59 exported primary experiment reports in:

- experiments/team_a_s2: 4 reports;
- experiments/team_b_core: 33 reports;
- experiments/team_b_alt: 17 reports;
- experiments/independent_renewal: 1 report;
- experiments/independent_domain: 1 report;
- experiments/independent_calendar: 1 report;
- experiments/independent_global_regime: 1 report;
- experiments/independent_anniversary: 1 report.

The corresponding machine-readable reconstruction is branch_reports_records.jsonl. It has one object per report and keeps measured facts in facts separately from the report author's interpretation in interpretation.

AGENTS, STATE, HISTORY, README, roadmap, TODO, master-summary, and other project-state summaries were not used as factual evidence. The global-regime report contains five textual references to STATE.md and the anniversary report contains one; these references are recorded as provenance contamination only. No result in this audit is derived from those statements.

Evidence precedence used here was: copied run artifacts and metric JSON/CSV, run_metrics.csv associations, experiment manifests and submission files, then the individual primary report. No claim from a report was upgraded to artifact-backed merely because it contains a precise number.

## Exact coverage counts

| Measure | Count | Meaning |
|---|---:|---|
| Report rows | 59 | All in-scope Markdown reports |
| JSONL rows | 59 | One valid JSON object per report |
| Unique report paths | 59 | No orphan or repeated report path |
| Canonical experiment units | 58 | After collapsing one exact semantic rerun |
| Exact duplicate/rerun records | 1 | team_b_core:EXP-031 duplicates team_b_core:EXP-005 |
| Retained semantic re-evaluations | 2 | team_b_alt:EXP-012 and EXP-015 revisit prior evidence under a changed purpose/protocol |
| Distinct normalized family labels | 30 | Bottom-up labels in the JSONL |
| Machine-artifact-supported records | 9 | Four team_a_s2 plus five independent reports |
| Primary-report-only records | 50 | All team_b_core and team_b_alt records |
| Records with any numeric measured outcome | 59 | Includes CV, LB-only, AUC, simulation error, or controlled contrast |
| Records without any numeric measured outcome | 0 | None |
| Records without a numeric local-CV scalar | 2 | team_b_core:EXP-030 and EXP-033; both are LB-only |
| Records carrying an LB field | 13 | Includes reused and qualitative LB evidence |
| Submission-file-backed LB records | 1 | team_a_s2:EXP-012 |
| Report-only LB records | 12 | No matching submission artifact in the specified evidence roots |
| Platform-export-verified LB records | 0 | No leaderboard export/API evidence was present |
| Named Team B CSV files missing from evidence roots | 13 | Listed below |
| All 59 reports represented | yes | Path set equality verified |

The “without numeric result” count is zero under the broad evidence definition above. If “result” is restricted to an ordinary local CV scalar, the count is two. S2-E00 is a numerical simulation audit, CALENDAR-PLACEBO-01 reports AUC diagnostics, and EXP-058 reports controlled deltas; they therefore are not result-less even though they do not expose an ordinary RMSLE CV scalar.

## Machine-evidence availability

| Namespace | Reports | Specified machine evidence | run_metrics.csv rows linked by namespace/association | Assessment |
|---|---:|---|---:|---|
| team_a_s2 | 4 | 18 artifact files, one copied report, one copied submission | 14 | Direct machine support for all four |
| team_b_core | 33 | none | 0 | Report-only |
| team_b_alt | 17 | none | 0 | Report-only |
| independent_renewal | 1 | 17 worktree artifact files | 2 | Direct machine support |
| independent_domain | 1 | 17 direct domain_01 files, 25 git-export files; 874 main and 17 renewal dependency files also copied | 486 | Direct result is supported; the large association count contains dependencies and is not 486 independent runs |
| independent_calendar | 1 | 69 worktree and 17 git-export files | 3 | Direct machine support |
| independent_global_regime | 1 | 9 worktree artifact files | 2 | Direct machine support |
| independent_anniversary | 1 | 9 artifact files plus one report and two source files | 4 | Direct machine support |

## Comparison normalization

All deltas in branch_reports_records.jsonl use candidate minus its stated baseline. Negative is better for RMSLE. When a primary report uses “gain” with the opposite sign, the normalized delta is retained and the sign mismatch is documented in conflicts.

The protocols below are not pooled:

- team_b_core normally makes decisions from folds 1 and 2; fold 3 is diagnostic. Its reported two-fold decision mean was preserved rather than silently recomputing a three-fold mean.
- team_b_alt EXP-001 through EXP-006 and EXP-008 through EXP-010 are principally single-cutoff studies at 2026-01-15. EXP-007 is a four-fold validation audit. EXP-011 uses a different clean cutoff. EXP-012 through EXP-017 use two-fold variants. Scores crossing these regimes are marked incomparable unless the report performs a paired comparison.
- team_a_s2 EXP-009 is a simulation-accuracy audit, EXP-010 a one-fold screen, EXP-011 a two-fold calibration study, and EXP-012 a four-fold standalone run. Their raw numbers are not treated as one leaderboard.
- independent renewal, domain, calendar, global-regime, and anniversary reports use different endpoints: weighted temporal CV, ordinary/weighted domain CV, classification AUC, re-anchored weighted CV, and paired controlled contrasts respectively.
- Local CV, OOF, public LB, simulation error, AUC, and auxiliary diagnostics remain distinct fields or explicitly labeled strings.

## Report-by-report factual index

The “fact” column below is a compact index. Full folds, baselines, seeds, artifacts, confounders, and interpretation are in the JSONL.

### team_a_s2

| Experiment | Compact factual result | Interpretation/status recorded separately | Evidence |
|---|---|---|---|
| EXP-009 | FW max error 0.1002666; hybrid MC-200k max error 0.00549019; QN11/QN21 difference 1.2888e-11 | Hybrid aggregator accepted as technical prerequisite | Machine + report |
| EXP-010 | Hurdle 1.75749468 vs Poisson 1.76579436; delta -0.00829968 on one fold | Accepted one-fold screen | Machine + report |
| EXP-011 | K5 1.76492425; numerical grid minimum K8 1.76484592; nested-sigma final 1.76481286 | K5 deliberately retained downstream despite K8 numerical minimum | Machine + report |
| EXP-012 | Four folds 1.78113175, 1.77350915, 1.76246388, 1.75611656; mean 1.76830534; LB 1.6619324597771563 | Accepted standalone, rejected near-neutral S1 blend | Machine + report + copied submission |

### team_b_core

| Experiment | Normalized factual endpoint | Report interpretation/status | Evidence |
|---|---|---|---|
| EXP-001 | Two-fold baseline 1.717017 | Baseline | Report-only |
| EXP-002 | 1.717011; delta -0.000006 | Rejected below practical gate | Report-only |
| EXP-003 | 1.717017; delta 0 | Rejected | Report-only |
| EXP-004 | 1.716725; delta -0.000292 | Rejected below gate | Report-only |
| EXP-005 | 1.717040; delta +0.000023 | Rejected | Report-only; later exact rerun |
| EXP-006 | 2.045762; delta +0.328745 | Rejected | Report-only |
| EXP-007 | 2.391320; delta +0.674303 | Rejected | Report-only |
| EXP-008 | 1.732425; delta +0.015408 | Rejected | Report-only |
| EXP-009 | 1.716802; delta -0.000215 | Rejected below gate | Report-only |
| EXP-010 | 1.716961; delta -0.000056 | Rejected below gate | Report-only |
| EXP-011 | 1.717095; delta +0.000078 | Rejected | Report-only |
| EXP-012 | 1.716910; delta -0.000107 | Rejected below gate | Report-only |
| EXP-013 | 1.716366; delta -0.000651 | Rejected below gate | Report-only |
| EXP-014 | 1.716513; delta -0.000504 | Rejected below gate | Report-only |
| EXP-015 | Local 1.708737 vs 1.717017, delta -0.008280; rounded LB about 1.6700 vs 1.6615 | Local accept, LB reject | Report-only |
| EXP-016 | 1.708734; delta -0.000003 vs calibrated baseline | Rejected | Report-only |
| EXP-017 | 1.708632; delta -0.000105 | Rejected below gate | Report-only |
| EXP-018 | 1.708783; delta +0.000046 | Rejected | Report-only |
| EXP-019 | 1.709231; delta +0.000494 | Rejected | Report-only |
| EXP-020 | 1.708634; delta -0.000103 | Rejected below gate | Report-only |
| EXP-021 | 1.708309; delta -0.000428 | Rejected below gate | Report-only |
| EXP-022 | 1.711550; delta +0.002813 | Rejected | Report-only |
| EXP-023 | 1.707779; delta -0.000958 | Rejected below gate | Report-only |
| EXP-024 | Best arm 1.708145 vs 1.708737; delta -0.000592 | Rejected below gate | Report-only |
| EXP-025 | 1.708564; delta -0.000173 | Rejected below gate | Report-only |
| EXP-026 | 1.708532; delta -0.000205 | Rejected below gate | Report-only |
| EXP-028 | 1.716251; -0.000766 vs EXP-001 but +0.007514 vs EXP-015 | Rejected against correct current baseline | Report-only |
| EXP-029 | 1.716161; +0.007424 vs EXP-015; LB 1.65841662470559 | Local reject, report claims useful LB candidate | Report-only |
| EXP-030 | No local CV; LB F 1.6566618522758063 and G 1.6561975700155196 | Positive LB diagnostic only | Report-only |
| EXP-031 | 1.717040; delta +0.000023, exactly repeats EXP-005 | Rejected semantic rerun | Report-only |
| EXP-032 | 1.720406; delta +0.003389; one fold value absent | Rejected | Report-only |
| EXP-033 | No comparable local spring CV; rounded LB 1.6599 | Weak LB improvement claim | Report-only |
| EXP-034 | Temporal 1.721106 vs paired central 1.716842; delta +0.004264 | Rejected | Report-only |

### team_b_alt

| Experiment | Normalized factual endpoint | Report interpretation/status | Evidence |
|---|---|---|---|
| EXP-001 | Single-cutoff HGBR baseline 1.711195 | Baseline | Report-only |
| EXP-002 | 1.710919; delta -0.000276 | Accepted single-fold feature change | Report-only |
| EXP-003 | 1.711856; delta +0.000661 | Rejected | Report-only |
| EXP-004 | 1.710617; delta -0.000578 | Accepted single-fold feature change | Report-only |
| EXP-005 | Scale sweep 1.672748; delta -0.038447 | Accepted single-fold calibration | Report-only |
| EXP-006 | LightGBM 1.710143; delta -0.001052 | Accepted single-fold model replacement | Report-only |
| EXP-007 | Four-fold audit mean 1.729343, std 0.015957 | Validation audit, not candidate improvement | Report-only |
| EXP-008 | Local 1.671639; reported rounded LB 1.657 | Accepted in report | Report-only |
| EXP-009 | Ensemble 1.670716 vs component 1.671832; LB 1.6568530995317488 | Accepted in report | Report-only |
| EXP-010 | Best equals baseline 1.670716; tested w0.55 variants 1.670717/1.670731 | Neutral weight sweep | Report-only |
| EXP-011 | Clean-cutoff local scale1.4 1.712473; LB best scale1.2 1.6549097093483665 | Accepted LB setting despite local rank reversal | Report-only |
| EXP-012 | Two-fold scale scores 1.720075, 1.709007, 1.709215; reuses EXP-011 LB values | Validation-alignment audit, no new submission | Report-only |
| EXP-013 | Local best scale1.300 1.707984; LB scale1.300 1.656279619331512 vs scale1.2 1.6549097093483665 | Rejected transfer to LB | Report-only |
| EXP-014 | Best threshold arm equals baseline 1.709007; nontrivial gates worse | Rejected | Report-only |
| EXP-015 | Two-fold w0.525 1.709004 vs w0.5 1.709007; exact candidate LB absent, described as slightly worse | Rejected qualitative LB result | Report-only |
| EXP-016 | 1.708883 vs 1.709007; LB 1.6547788658437297 | Accepted in report | Report-only |
| EXP-017 | 1.708295 vs 1.708883; LB 1.6546318191 | Accepted in report | Report-only |

### independent reports

| Experiment | Machine-backed fact | Report interpretation/status | Comparability |
|---|---|---|---|
| RENEWAL-01 / EXP-027 | Base wCV 1.7483427844; CLOCK meta 1.7480763705; replacement LOFO -0.0004159051, 4/4; incremental meta vs self -0.0001110765 | STOP; no submission | Its weighted four-fold protocol is not pooled with Team B |
| DOMAIN-01 / EXP-028 | Ordinary 1.7511635266 vs 1.7510761354, delta +0.0000873912; weighted delta +0.0000855190; production LOFO +0.0000097718, 2/4 | STOP; no submission | Domain folds and weighted endpoint are distinct |
| CALENDAR-PLACEBO-01 / EXP-029 | Real-120d AUC 0.64434668; residual to gap-curve expectation -0.00115 | STOP-CALENDAR; no submission | Classification/placebo diagnostic, not RMSLE |
| GLOBAL-REGIME-OCC-RANK / EXP-057 | Re-anchored final_GLOBAL 1.7474862831 vs final_BASE 1.7474288101; delta +0.0000574730, 0/4 | Rejected | Exact original endpoint unavailable because X3/B OOF was missing |
| Exact anniversary / EXP-058 | REAL-base -0.001139472; REAL-shuffled -0.000673637; REAL-shifted +0.000390291, wrong sign in both recipient halves | Rejected | Controlled contrast; absolute ordinary CV unavailable |

## Family distribution

The 30 exact family labels and counts present in the JSONL are:

| Family | Count | Family | Count |
|---|---:|---|---:|
| anniversary_features | 1 | calendar_and_domain_shift | 1 |
| calendar_features | 1 | calibration | 3 |
| calibration_and_shrinkage | 1 | count_and_zero_modeling | 1 |
| domain_shift_and_reweighting | 1 | ensemble_weight_sweeps | 2 |
| ensembles | 4 | feature_engineering | 6 |
| global_regime_occurrence | 1 | hyperparameter_tuning | 6 |
| integrated_pipeline | 1 | objective_functions | 2 |
| postprocessing | 1 | postprocessing_and_ensembles | 1 |
| renewal_and_occurrence_models | 1 | seasonal_features | 1 |
| seed_averaging | 1 | structural_pipeline | 1 |
| structural_target_aggregation | 1 | tabular_baseline | 2 |
| tabular_models | 1 | target_decomposition | 1 |
| target_distribution | 1 | target_processing | 1 |
| temporal_features | 4 | train_example_construction | 4 |
| validation_audit | 2 | zero_modeling | 5 |

These labels describe the tested change, not an assertion that all members share a comparable validation protocol.

## Deduplication and identity

One record is collapsed for canonical counting:

- team_b_core:EXP-031 is an exact semantic rerun of team_b_core:EXP-005. Both give decision mean 1.717040 and delta +0.000023 under the same fold convention. EXP-031 remains in the registry with duplicate_of set to team_b_core:EXP-005.

Two related experiments are retained as distinct canonical units:

- team_b_alt:EXP-012 reuses the LB observations from EXP-011, but its measured object is a two-fold validation-alignment audit. It is not a new submission and not a duplicate of the original training run.
- team_b_alt:EXP-015 revisits the ensemble-weight question from EXP-010 under a two-fold protocol and a fixed scale. It is a semantic continuation, not an exact rerun.

Bare numeric IDs are unsafe identifiers in this slice. Nineteen numeric IDs collide across namespaces, covering 42 of 59 records: EXP-001 through EXP-017, plus EXP-028 and EXP-029. Namespaced experiment_id values are unique.

## LB and submission evidence

Thirteen records contain an LB field. Only team_a_s2:EXP-012 has a copied submission artifact. The file submission_strategy_2.csv has 250,000 rows and SHA256 cab04e8cc94066819ebe1c548624267768c380b1c98b183331c7d13625b01668; copied machine checks cover schema/order. Its reported LB 1.6619324597771563 is manifest/report-backed, but no independent platform export was available.

The other 12 LB-bearing records are report-only. One of them, team_b_alt:EXP-012, merely reuses earlier LB values; team_b_alt:EXP-015 gives only a qualitative candidate outcome. Precise decimals in the remaining reports do not substitute for the absent CSV or leaderboard export.

The following 13 Team B CSV names are mentioned by primary reports but absent from both specified evidence roots:

- E_hurdle_variance_exp029.csv
- F_hurdle_variance_exp029_season_yoy_1_168.csv
- G_hurdle_variance_exp029_season_1_12.csv
- exp_011_dense8_logens.csv
- submissions/exp_002_conversions_hgbr.csv
- submissions/exp_008_recency_lightgbm_scale064.csv
- submissions/exp_009_recency_long_buy_lgbm_logens.csv
- submissions/exp_010_logens_wrec055_scale100.csv
- submissions/exp_010_logens_wrec055_scale101.csv
- submissions/exp_015_dense8_logens_wrec0525_scale120.csv
- submissions/exp_016_post_order_wrec050_scale120.csv
- submissions/exp_017_dist_post_order_wrec050_scale120.csv
- submissions/H_multicutoff_spring_exp033.csv

No independent experiment in this slice produced a submission.

## Conflicts, selection divergences, and confounders

1. team_a_s2:EXP-011: the machine grid has a marginally lower numerical score at K8 than K5 by about 0.00007832, while the report and downstream S2 final manifest deliberately select K5. This is an explicit selection override, not silently rewritten as K8.
2. team_b_core:EXP-031 and EXP-032: their prose “gain” sign is opposite the normalized candidate-minus-baseline convention. The JSONL records +0.000023 and +0.003389 respectively, both worse.
3. team_b_core:EXP-015: local calibration improves 1.717017 to 1.708737, while the rounded reported LB worsens from about 1.6615 to about 1.6700. These are retained as separate facts.
4. team_b_alt:EXP-011: the clean local cutoff prefers scale 1.4, while the reported LB prefers 1.2. team_b_alt:EXP-013 again prefers a larger scale locally but loses to 1.2 on LB. team_b_alt:EXP-015 has a tiny local improvement and a qualitatively worse LB. None is treated as a universal scale result.
5. team_b_core:EXP-032 lacks one fold-level score even though an aggregate decision score is reported.
6. independent_global_regime:EXP-057 could not reproduce the exact planned X3/B endpoint because the required OOF was absent. Its machine result is a clearly labeled re-anchored comparison.
7. independent_anniversary:EXP-058 passes a retrain-parity tolerance but is not bitwise identical. The reported approximately 8,101 seconds is retrain plus experiment runtime; the machine summary's 6.9 seconds is only a finalization stage.
8. Five STATE.md mentions in EXP-057 and one in EXP-058 are provenance contamination. They were not used to establish baselines, prior results, verdicts, or lineage.

## Reproducibility limits and completeness

- All 59 in-scope report files resolve from their JSONL report_path, and there are no missing or extra paths.
- All 59 JSON lines parse, use the same 30-field schema as primary_reports_records.jsonl, and have unique namespaced IDs.
- Machine evidence supports nine records. The remaining 50 records have no matching metric, prediction, OOF, or submission artifact in the specified evidence roots and are therefore report-only even when a report gives exact fold values.
- Runtime is unknown for 55 of 59 records. For the four remaining records, at least part of the value is a stage estimate rather than a complete wall-clock measurement.
- Date is unknown for independent_global_regime:EXP-057. Seed is unknown for team_b_core:EXP-031, EXP-032, and EXP-033.
- Public-LB claims cannot be independently verified against a platform export. One submission CSV is available and checksum-backed; 13 named Team B CSVs are missing.
- No cross-protocol aggregate “best CV” was computed. Any later aggregation must filter by validation_protocol, baseline, train coverage, endpoint, and evidence strength.

Within the stated branch-report scope, reconstruction coverage is complete at the report-row level (59/59) and partial at the artifact-verification level (9/59).
