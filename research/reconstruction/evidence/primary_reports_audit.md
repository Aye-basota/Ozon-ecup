# Forensic audit of primary experiment reports

## Scope and evidence boundary

This audit covers every case-insensitive `experiments/exp_*.md` file in the source repository and the direct metrics, code, logs, predictions, models, and submission artifacts referenced by those reports. The source repository was treated as read-only.

No factual assertion in this audit was sourced from `AGENTS.md`, `AGENT.md`, `STATE.md`, `HISTORY.md`, `README.md`, roadmaps, TODO files, instruction/prompt files, or master/executive summaries. When a primary report itself cited such a secondary document, that citation was excluded from the factual record. Report-local measured values were retained; author judgments were stored separately as `interpretation`.

Machine-readable detail is in `primary_reports_records.jsonl`. It has one JSON object per report and uses the literal string `unknown` when the primary evidence does not establish a value.

## Exact census

- Primary Markdown report files found: **65**.
- JSONL records written: **65**; unique `report_path` values: **65**.
- Canonical experiment/audit units after deduplication: **64**.
- Non-independent duplicate/manifest records: **1** (`EXP_032_S04_conditional_fresh_seq.md`, linked to canonical `EXP-032`).
- Explicit rerun/extension reports, retained as separate evidence: **2** (`EXP-030B`, `EXP-030C`).
- Reopened older component under a materially newer baseline, not treated as a duplicate: **1** (`EXP-063` revisiting E11).
- Fine-grained bottom-up family labels in JSONL: **28**, including the one `experiment_manifest` label; **27** non-manifest labels.
- Source `submissions/` CSV files present: **23**.
- Exact existing submission paths directly associated by these reports: **19** unique paths (**22** report occurrences because later integration/diagnostic reports reuse files).
- Distinct numeric public-LB values recoverable from primary reports: **15**. Of these, **12** are associated at report level with a specific existing submission filename, **1** is ambiguously attached to one of two files (`EXP-015`), and **2** are external/historical values without an exact file-hash binding (`EXP-067`, `EXP-068`). None was independently queried from the competition platform in this audit.
- Mechanical normalized-status prefixes in the 65 records: `accepted*` **13**, `rejected*` **31**, `continued*` **4**, `blocked*` **3**, `prepared*` **3**, `diagnostic*` **2**, `technical*` **2**, `production*` **1**. The remaining **6** are one each: `baseline`, `duplicate_manifest`, `gate_pass_no_ensemble_gain`, `mixed_negative`, `open_mixed`, and `provenance_incomplete_external_lb`.

These status-prefix counts are an inventory, not a recommendation and not a claim that every `accepted*` record is a stronger standalone model. Several are accepted diagnostics, components, or packaging steps.

## Direct-reference verification

The 65 reports contained 234 raw backticked path-like mentions. After removing prose fragments and normalizing literal path references, 204 references were checked against the source tree:

- **199/204** resolve to existing files.
- **2/204** are code-symbol references rather than paths: `src/config.py::cutoff_grid()` and `src/validation.py:wcv`.
- **3/204** are deliberately absent artifacts whose absence is itself a primary fact: `submissions/BLOCK4_SAF_submission.csv`, `submissions/submission_FRESH_CONTRAST_MOE.csv`, and `artifacts/model_SEQ-D3A-BASE-S42-TEST.pt`.
- Unexpected missing positive artifact claims: **0** among these normalized literal references.

The intentionally absent model is part of the production blocker documented by `EXP-050`; the two absent submissions are explicit stop outcomes of `EXP-039` and `EXP-040`. Large artifacts were not copied or modified.

Four source submission CSVs are not directly associated by an exact path in the scoped Markdown reports: `submission_s04_a.csv`, `submission_s04_b.csv`, `submission_s04_blend.csv`, and `submission_tier_a_checkpoint.csv`. The last is listed in a direct submissions manifest, while the three S04 files require lineage evidence outside the scoped report-path associations. They should remain orphan/needs-association candidates rather than being guessed into an experiment.

## Comparison normalization

The following score classes are kept distinct in the JSONL and table below:

1. `EXP-001` through the early S1 reports primarily use a four-fold equal-weight mean and/or calibrated concatenated OOF.
2. From `EXP-016`, the main project metric is usually fold-calibrated `wCV` with temporal weights `1:2:4:8`. It is not numerically interchangeable with the earlier equal-weight mean.
3. LOFO ensemble deltas measure component contribution under an ensemble protocol; they are not standalone model scores.
4. `EXP-032` and `EXP-032B` use half-panel group-A `wCV(A)` and are explicitly incomparable with project-wide wCV.
5. Single-fold gates and preflights (`EXP-029`, `EXP-030B`, `EXP-038`, `EXP-043`-`045`, `EXP-052`, `EXP-054`-`058`, `EXP-061`, `EXP-062`, `EXP-064`) cannot be promoted to four-fold conclusions.
6. Three-fold pseudo-matched or same-fold analyses (`EXP-048`, `EXP-049`) use different support from standard four-fold CV.
7. Production-only, artifact-only, and provenance audits may have no new CV at all. Their valid result is an artifact, reproducibility, or blocker fact.
8. Public LB is separate from every local endpoint. A report plus an existing CSV is still weaker evidence than a platform export or score-to-hash manifest, neither of which is available for most scores here.

## Report-by-report normalized index

| Report | Protocol / primary measured fact | Normalized status | Key limitation or conflict |
|---|---|---|---|
| EXP-001 | 4F mean 1.76879; calibrated OOF 1.76570 | baseline | raw mean is not later wCV |
| EXP-002 | 4F 1.76981; delta +0.00102; 0/4 | rejected | exact panel implementation only |
| EXP-003 | 4F 1.76182; delta -0.00697; 4/4 | accepted | raw mean convention |
| EXP-004 | E03a 1.76787; E03b 1.76999; E03c 1.79931; E04 two-fold 1.75705 | mixed_negative | multi-arm/mixed protocol; E03d result missing |
| EXP-005 | 4F 1.75988; calibrated OOF 1.75889; delta -0.00194 | accepted_with_caveat | adversarial AUC reported as 1.0 |
| EXP-006 | ensemble 1.75886; calibrated OOF 1.75716; 3 scored CSVs | accepted | LB is report-level evidence |
| EXP-007 | minimal model 1.77335; LB 1.667424590457357 | rejected | standalone replacement |
| EXP-008 | standalone 1.85169; blend OOF 1.77210; LB 1.6682180280505314 | rejected | standalone and blend endpoints differ |
| EXP-013 | two-part 1.75893; calibrated OOF 1.75792 | open_mixed | better than E10, worse than S1-BEST |
| EXP-014 | DIST head 1.75834; ensemble OOF 1.75645; LB 1.6507774106 | accepted | post-hoc report extension present |
| EXP-015 | production-only, 24 cutoffs; best-of-two LB 1.6512012383165489 | rejected | LB cannot be mapped to either CSV |
| EXP-016 | MIX-E11 wCV 1.74911 vs 1.74948; LB 1.6510029 | diagnostic_negative_lb | local improvement did not transfer |
| EXP-017 | 600 rounds 1.75170; 200 1.75103; chosen 300 1.75108 | accepted_development | best and production choice differ |
| EXP-018 | AVG3 1.75046; AVG5 1.75037; 4/4 | accepted_development | seed ensemble, not new representation |
| EXP-019 | gap stress: E10 1.751415 to 1.756366 | rejected | stress axis incomparable with standard CV |
| EXP-020 | train-blocks=0 1.750569; delta +0.000113 | rejected | small exact-construction result |
| EXP-021 | real features delta -0.000006; shuffled -0.000028 | rejected | real did not beat shuffled control |
| EXP-022 | dense grid 1.752339; delta +0.001263; 0/4 | rejected | equal-volume subsampled design |
| EXP-023 | correction 1.749577 vs 1.749484; 0/4; CSV exists | prepared_high_risk_not_scored | positive diagnostic but negative target score |
| EXP-024 | FULL 1.75234; delta +0.00286 vs DIST mix; 0/4 | rejected | multiple internal/external baselines |
| EXP-025 | TCN standalone 1.75270; blend LOFO -0.00106; LB 1.650176372731295 | accepted_component | weak standalone, useful blend component |
| EXP-026 | AVG3 1.74963; blend LOFO -0.00169 | accepted_technique_stale_production_conclusion | later LB/depth audit overturns production interpretation |
| EXP-027 | SEQAVG3 LB 1.6553135958569027; clip289 LOFO -0.00055 | accepted_diagnostic | causal attribution remains diagnostic |
| EXP-028 | leakage found before training; proposed rows +51.4% | rejected_before_training | no model score by design |
| EXP-029 | one-fold base 1.74808; variants +0.00178/+0.00105 | rejected | one fold only |
| EXP-030 | seed42 D3A wCV 1.75284 vs 1.75361; 2/4 | continued_to_multiseed | later reports stronger evidence |
| EXP-030B | one paired fold delta -0.00035 at seed43 | continued_to_multiseed | rerun/extension, one fold |
| EXP-030C | 3-seed mean delta -0.00095; 3/4 folds | accepted_component_candidate | no ensemble/LB endpoint |
| EXP-032 | half-panel fresh-clean delta -0.00128; 4/4 | continued | wCV(A), one encoder seed |
| EXP-032-MANIFEST | preregistration plus duplicated result appendices | duplicate_manifest | duplicate ID; no independent run |
| EXP-032B | fresh gate positive; sequence-slot delta +0.00082; 0/4 | gate_pass_no_ensemble_gain | endpoint-dependent signs, half panel |
| EXP-035 | SEQAVG3 LOFO -0.00055; D3A-AVG3 -0.00061; CSV not sent | accepted_candidate_not_sent | close correlated components |
| EXP-036 | ETX 1.74953; coauthor LOFO -0.00091; 4/4 | continued_as_coauthor_rejected_as_replacement | role-dependent verdict; test anomaly |
| EXP-037 | final wCV 1.74751; LB 1.6496571 | accepted | report-level LB/file/hash evidence |
| EXP-038 | one-fold funnel/control results change sign by lambda | rejected | 10.3 GPU h; run-floor noise |
| EXP-039 | full wCV 1.747749; delta +0.000240; 0/4 | rejected | submission deliberately absent |
| EXP-040 | full wCV 1.747285; delta -0.000225; 4/4 | rejected_below_gate | positive score but below decision magnitude |
| EXP-041 | Ridge blends +0.000278/+0.000207; LB 1.6502464747481933 | rejected_local_submitted_by_override | verdict/action divergence |
| EXP-042 | delta -0.000024756; 2/4; LB 1.649547109893236 | rejected_local_submitted_by_override | p0 replay max difference 0.73336 |
| EXP-043 | two identical runs 1.745829867810; variance 0 | technical_pass | reproducibility control only |
| EXP-044 | one-fold 3-seed Fresh-VOL mean -0.000088094 | rejected_below_gate | one fold; worse than plain base on mean |
| EXP-045 | one-fold true-shuffled mean +0.000436113; 1/3 seeds | rejected | shuffled control wins |
| EXP-046 | primary refresh delta -0.000002447; 3/4 | rejected | 24 correlated trajectories; best factor post hoc |
| EXP-047 | BG/NBD residual nested delta -0.000269184; 4/4 | rejected_below_research_gate | later production gate differs |
| EXP-048 | pseudo-matched/standard selection audit | technical_inconclusive | incompatible support and estimands |
| EXP-049 | corrected same-3F deltas -0.000547/-0.000551 | rejected_for_production_support | exact production helpers missing |
| EXP-050 | FRESH encoder missing; BTYD MLE unstable | blocked | no new CV; later optimizer resolution |
| EXP-051 | fixed BTYD05 delta -0.000320983; 4/4; CSV exists | production_pass_not_uploaded | different gate from EXP-047 |
| EXP-052 | one-fold real-shuffled +0.001265695; selected scale 0 | rejected | real worse than shuffled |
| EXP-053 | late gate delta -0.000006419 | diagnostic_closed | no full LOFO/test |
| EXP-054 | burst-gap scale 0; delta 0 | rejected_preflight | one-fold CPU preflight |
| EXP-055 | landmark-memory scale 0; delta 0 | rejected_preflight | one-fold CPU preflight |
| EXP-056 | late-control +0.00003163; slot +0.00000723 | rejected | one fold; wrong sign in both halves |
| EXP-057 | matched-shuffled +0.000002324 | rejected | one fold; halves opposite |
| EXP-058 | standalone real-perm -0.000129771; ensemble +0.000071048 | rejected | endpoint signs differ; one fold |
| EXP-059 | fixed SEQ65 wCV 1.747272; delta -0.000238; 4/4 | prepared_not_uploaded | informational fixed-weight result |
| EXP-060 | level-shift CSV exists; no CV/LB | prepared_not_uploaded | incomparable, artifact only |
| EXP-061 | open-funnel scales 0; delta 0 | rejected_preflight | one-fold preflight |
| EXP-062 | detrend scales 0; delta 0 | rejected_preflight | one-fold preflight |
| EXP-063 | E11 revisit nested +0.0000105; fixed -0.0000098 | rejected | reopened idea, not duplicate run |
| EXP-064 | event-order scales 0; delta 0 | rejected_preflight | one-fold preflight |
| EXP-065 | packages strongest and BTYD05 exact files/hashes | accepted_final_package_not_uploaded_here | inherited metrics, no new model |
| EXP-066 | searched 627 NPZ and 83 parquet; canonical latest OOF absent | blocked_no_canonical_latest_oof | artifact absence, not model score |
| EXP-067 | exact external test recipe; LB 1.64921756224069 | provenance_incomplete_external_lb | missing two OOF components; LB unbound |
| EXP-068 | historical CSV/recipe; LB text 1.6492897556391737 | blocked_historical_replay | 32 OOF + 6 test helpers absent; LB unbound |

## Deduplication and ancestry notes

### Exact/non-independent duplicate

`EXP_032_S04_conditional_fresh_seq.md` is a preregistration/manifest that was later amended with summaries of the lowercase `EXP-032` pilot and `EXP-032B` production-conditioned follow-up. It is preserved as a record but has `duplicate_of: EXP-032` and is excluded from the canonical-unit count.

### Reruns and extensions

- `EXP-030B` is a seed43/single-fold paired replay of the D3A idea from `EXP-030`.
- `EXP-030C` is the 3-seed × 4-fold confirmation of the same D3A branch.
- `EXP-063` reuses the older E11 occurrence prediction family under the much later `EXP-037` ensemble. Because the parent baseline and integration question changed, it is a reopened idea rather than a duplicate of `EXP-013`.

### Strong-solution lineage visible in primary reports

- Early tabular line: `EXP-001` → `EXP-003`/`EXP-005` → `EXP-006` → `EXP-014` → `EXP-016`.
- Sequence line: `EXP-025` → `EXP-026` → failure diagnosis `EXP-027` → D3A `EXP-030/030B/030C` → sequence-slot selection `EXP-035`.
- Sparse-event line: `EXP-036` → seed-averaged/coauthor integration `EXP-037`.
- BG/NBD residual line: `EXP-047` → selection/support audits `EXP-048/049` → production blocker `EXP-050` → stable production `EXP-051` → package `EXP-065`.
- External/historical provenance lines remain parallel and incomplete in `EXP-066`-`068`; they must not be merged into the locally reproducible `EXP-037/051/065` ancestry without missing OOF/helper evidence.

## Contradiction registry for this evidence slice

### Hard identity or score-binding conflicts: 3 clusters

1. **Duplicate ID 032.** Uppercase manifest and lowercase primary pilot both claim the `EXP-032` identity; only the lowercase pilot is canonical result evidence.
2. **EXP-015 ambiguous LB binding.** The report states one `best of two` LB value but lists two sent CSVs; assigning the score to either filename would be fabrication.
3. **EXP-026 versus EXP-027 stale production record.** `EXP-026` says the full-depth SEQAVG3 file was not sent and treats full depth as safe. `EXP-027` records that it was sent, gives LB 1.6553135958569027, and identifies full-depth test support as a failure mode. The later report supersedes submission state and interpretation, while the local OOF measurements in `EXP-026` remain valid.

### Verdict/action divergences: 2 clusters

- `EXP-041` and `EXP-042` each retain a local `REJECT` verdict but later have submission files and reported LB values because of explicit owner overrides. This is not evidence that the local verdict was numerically wrong; action and research verdict are separate fields.

### Protocol corrections or apparent conflicts resolved by normalization: 3 clusters

- `EXP-048` used incompatible pseudo-matched/standard support; `EXP-049` corrects the comparison to the same three folds.
- `EXP-047` rejects a nested delta under a research magnitude gate; `EXP-051` passes a fixed-weight candidate under a production gate after optimizer stabilization. Different estimator and gate, so the decisions are not directly contradictory.
- `EXP-050` reports unstable production MLE; `EXP-051` changes the optimizer and records stable production. This is a resolved implementation lineage, not two claims about the same fit.

### Unbound leaderboard provenance: 2 clusters

- `EXP-067`: external LB 1.64921756224069 is not bound to the reconstructed CSV by a score manifest/hash and canonical OOF is missing for two components.
- `EXP-068`: historical LB 1.6492897556391737 is present as recipe/manifest text but not bound to the surviving CSV hash; exact replay is impossible with 38 missing helper artifacts.

## Completeness and residual gaps

- Orphan primary Markdown reports within the scoped filename family: **0**; every one of the 65 files has a JSONL record.
- Duplicate `report_path` values: **0**.
- Duplicate/non-independent result records: **1** manifest, explicitly linked.
- Primary reports with no comparable new model score are retained, not dropped. These include leakage stops, reproducibility controls, production blockers, artifact-only packaging, and provenance audits.
- The largest reconstruction gaps are deliberately recorded as `unknown`, `blocked`, or `incomparable`: E03d's missing numeric result; the EXP-015 score-to-file mapping; exact FRESH production encoder; canonical latest/external OOF; and the 38 recency-ridge replay helpers.
- Runtime is absent from many early reports and is not inferred from file timestamps.
- Seeds are marked `inherited` or `unknown` when a report reuses predictions without proving the original training seed.
- LB scores are author-/report-recorded unless explicitly labeled external/historical; no score is elevated to platform-verified evidence.
- The audit did not run training, create predictions, tune weights, or create a submission.

## Source preservation

All writes for this task were confined to the sibling repository files `research_clean/evidence/primary_reports_audit.md` and `research_clean/evidence/primary_reports_records.jsonl`. No source-repository file was edited.
