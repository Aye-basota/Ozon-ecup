# Strategy results forensic audit: EXP019–032b, unnumbered S04, and EXP035–068

## Scope and evidence policy

This audit covers machine-evidenced canonical units in EXP019–EXP032b, the unnumbered S04 LightGBM run, and the contiguous EXP035–EXP068 window. It examines `research/strategies/results/**`, matching artifacts under `artifacts/**`, concrete scripts under `src/**` or the strategy directory, and produced submissions. No independent run evidence was found for EXP031, EXP033, or EXP034, so no rows were fabricated for those IDs.

Evidence priority was machine output first: NPZ/NPY/CSV/JSON manifests and checksums, then concrete experiment code, then a primary experiment report only when a machine artifact did not contain the needed field. AGENTS/AGENT, STATE, HISTORY, README files, strategy indexes, roadmaps, TODOs, master/executive summaries, and recommendation text were not used as evidence. EXP066 itself embeds a claim sourced from a forbidden README; that claim is explicitly excluded. Repository text suggesting future work was ignored.

No source file was modified and no training, inference, tuning, or submission generation was run.

This file deliberately overlaps the report-oriented registry for canonical experiment IDs. It is an independent machine-evidence verification layer, not a second set of experiments: downstream integration should join on normalized canonical ID/name, prefer the machine score, protocol, artifact parity, and checksum recorded here, and retain report prose only as interpretation. The unnumbered S04 run is the one additional canonical unit because machine reports, OOF, test predictions, and three submissions prove that it actually ran. Arms, seeds, controls, and weight sweeps remain nested inside their canonical unit.

## Coverage counts

- Canonical experiment/run units: **51**: 50 numbered/suffixed IDs and one unnumbered S04 machine run.
- Early scope: **16** canonical numbered/suffixed units in EXP019–EXP034 plus S04. EXP031, EXP033, and EXP034 are absent; uppercase EXP_032 is a specification, not another run.
- Normalized bottom-up families in this file: **16**.
- Units with a four-fold measured primary result: **27**. Two are same-split half-panel results, one uses paired per-seed deltas, EXP059 is machine-labelled informational, and EXP051 numerically replays EXP047 validation; those qualifiers prevent cross-protocol comparison.
- Three-fold-only corrected comparison: **1** (EXP049).
- One-fold, preflight, reproducibility, diagnostic, or nonstandard stress units: **16**.
- Package, production-only, invalid, or blocked units without a new comparable CV: **7**.
- Negative/rejected/no-go units: **30**.
- Pass/progression, reproducibility pass, or artifact-package/submission units: **16**.
- Blocked/inconclusive/provenance-only units: **5**.
- Confirmed created submission files in this scope: **13**.
- SHA-bound public leaderboard results found in this evidence zone: **0**.
- Dedup/rerun/output-identity cases recorded: **9 clusters/cases**.

The JSONL file is the canonical machine-readable output. Arms, seed variants, weight sweeps, and controls are nested inside their canonical experiment instead of being counted as separate hypotheses.

## Metric protocols

The main comparable protocol is calibrated RMSLE on expanding temporal folds 2025-09-04, 2025-09-18, 2025-10-02, and 2025-10-16, aggregated with weights 1:2:4:8. A fixed recipe, LOFO-selected recipe, standalone score, one-fold pilot, public LB, gap stress curve, paired per-seed delta, and half-panel user split are different objects and are never merged here.

The early S1 component-bank baseline was independently reconstructed from aligned OOF arrays as 0.15 E10 + 0.30 E02 + 0.10 E03a + 0.45 DIST, with folds 1.7691336442 / 1.7625660982 / 1.7507266459 / 1.7431352014 and wCV 1.7494836024. EXP025 changes this to SEQ-01-MIX at wCV 1.7483427845, and EXP026 changes it again to SEQ-AVG3-MIX at wCV 1.7477422600. These are distinct baselines.

The recurring exact baseline is EXP037 STRONGEST_CURRENT:

`0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 ETX-AVG3 + 0.225 SEQ-AVG3`

Later exact manifests replay calibrated fold scores 1.7668833568 / 1.7605095768 / 1.7486292240 / 1.7412785664 and wCV 1.7475098625201952. EXP035–037 also compare against an earlier SEQ-01-MIX reference at wCV 1.7483427845330886; deltas against that reference must not be presented as deltas against STRONGEST_CURRENT.

## Chronology and normalized result

### EXP019–032b and unnumbered S04

| ID | Normalized result | Protocol | Status |
|---|---:|---|---|
| EXP019 | E10 G30 1.751414965; G120 1.756366450; stress Δ +0.004951485 | four-fold gap stress, not canonical CV | reject diagnostic |
| EXP020 | 1.750569142, Δ +0.000112739 vs capacity-matched TB1 | four-fold avg3; 2/4 | reject |
| EXP021 | 1.750450253, Δ -0.000006150; shuffled control 1.750428441 | four-fold avg3 | reject mechanism |
| EXP022 | 1.752338649, Δ +0.001262513 | four-fold capacity-matched; 0/4 | fail |
| EXP023 | 1.749576694, Δ +0.000093097 | four-fold fixed; 0/4 | reject; submission created |
| S04 | final blend 1.748918751, Δ -0.000564852 vs S1-DIST-MIX | four-fold fixed; 4/4 | three submissions created |
| EXP024 | FULL 1.752344861, Δ -0.000002329 vs SELF control | four-fold causal contrast | reject |
| EXP025 | ensemble 1.748342784, Δ -0.001140818 vs S1-DIST-MIX | four-fold fixed; 4/4 | continue; submission created |
| EXP026 | ensemble 1.747742260, Δ -0.000600524 vs EXP025 | four-fold fixed; 4/4 | technique accepted; submission created |
| EXP027 | no canonical candidate CV | cross-depth and avail diagnostics | diagnostic accepted |
| EXP028 | no run metric | leakage preflight; training stopped | reject preflight |
| EXP029 | A25 1.749132578, Δ +0.001052819 vs fresh BASE | one fold | reject |
| EXP030 | 1.752840411, Δ -0.000770503 vs paired seed-42 BASE | four-fold single seed; 2/4 | continue |
| EXP030b | 1.763672284, Δ -0.000350527 on 09-18 | one-fold seed-43 rerun | scale diagnostic |
| EXP030c | paired weighted Δ -0.000947340; averaged-OOF Δ -0.000703801 | 3 seeds x 4 folds; two valid aggregations | keep; no submission |
| EXP032 | FRESH 1.747212462, Δ -0.001278994 vs CLEAN | same group-A half-panel; 4/4 | continue mechanism |
| EXP032b | DISTxFRESH 1.745144262, Δ -0.001009788 vs DISTxCLEAN | same group-A half-panel; 4/4 | mechanism pass, component reject |

### EXP035–068

| ID | Normalized result | Protocol | Status |
|---|---:|---|---|
| EXP035 | 1.747767776, Δ -0.000575009 vs SEQ-01-MIX | 4-fold fixed; LOFO also positive | submission created |
| EXP036 | 1.747519298, Δ -0.000823487 for 50/50 ETX-S42+SEQ slot | 4-fold fixed vs SEQ-01-MIX | component partial pass |
| EXP037 | 1.747509863, Δ -0.000832922 vs SEQ-01-MIX | 4-fold fixed; 4/4 | promoted STRONGEST_CURRENT |
| EXP038 | best direct 1.745911719, but FUNNEL-vs-control near zero/worse | one fold | reject |
| EXP039 | 1.747749476, Δ +0.000239613 | 4-fold honest LOFO; 0/4 | reject |
| EXP040 | 1.747284906, Δ -0.000224956 | 4-fold nested; 4/4 | reject by magnitude gate |
| EXP041 | 1.747716581, Δ +0.000206719 | 4-fold blend; 0/4 | reject; submission forced |
| EXP042 | 1.747485107, Δ -0.000024756 | 4-fold nested; 2/4 | reject; submission forced |
| EXP043 | repeated score 1.745829868, bitwise identical | one-fold deterministic rerun | reproducibility pass |
| EXP044 | mean paired Δ -0.000088094 | one fold, 3 seeds | reject |
| EXP045 | true-vs-shuffle mean Δ +0.000436113 | one fold, 3 seeds | fail |
| EXP046 | 1.747457303, Δ -0.000052560 | 4-fold fixed; 4/4 | reject by magnitude |
| EXP047 | nested 1.747240678, Δ -0.000269184 | 4-fold nested; 4/4 | reject by gate |
| EXP048 | no valid normalized number | mismatched fold protocols | technical inconclusive |
| EXP049 | Δ -0.000546668 standard / -0.000551306 matched | same 3 folds only | production unsupported |
| EXP050 | no new CV | production stability/parity audit | blocked |
| EXP051 | nested 1.747240681, Δ -0.000269182 | 4-fold numerical replay | reject, yet submission created |
| EXP052 | actual pilot Δ 0; oracle is non-realizable | one-fold pilot after oracle preflight | reject |
| EXP053 | late-fold Δ -0.000006419 | one fold; full LOFO not run | no promotion |
| EXP054 | selected scale 0 in both directions | one-fold preflight | no-go |
| EXP055 | selected scale 0 in both directions | one-fold preflight | no-go |
| EXP056 | slot Δ +0.000007228 | one-fold paired | reject |
| EXP057 | matched slot Δ +0.000016169 | one-fold matched/shuffle | reject |
| EXP058 | real slot Δ -0.000022517; real worse than perm control by +0.000071048 | one fold | reject |
| EXP059 | 1.747271965, Δ -0.000237898 | 4-fold, explicitly informational | submission created |
| EXP060 | no local CV | production-only log shift | submission created |
| EXP061 | Δ 0, selected correction 0 | 4-fold crossfit | reject |
| EXP062 | Δ 0, selected correction 0 | 4-fold crossfit | reject |
| EXP063 | nested 1.747520360, Δ +0.000010498 | 4-fold nested | reject |
| EXP064 | Δ 0, selected correction 0 | 4-fold crossfit | reject |
| EXP065 | no new CV; two existing files verified | integration/package audit | accept package |
| EXP066 | no run metric | prerequisite inventory | blocked: latest OOF absent |
| EXP067 | test blend reconstructs within tolerance; no canonical OOF | provenance/test audit | partial reproducibility |
| EXP068 | historical metric is incomparable to latest; replay not run | blocked audit | blocked |

## Ancestry

The reconstructed early ancestry begins from the aligned S1 component bank. EXP019 is a validation-stress branch; EXP020 and EXP022 change train construction; EXP021 and EXP023 change temporal representation/postprocessing; unnumbered S04 and EXP024 are parallel auxiliary-supervision branches. EXP028 is a stopped preflight branch with no trained candidate.

The sequence lineage is:

`S1-DIST-MIX → EXP025 SEQ-01-MIX → EXP026 SEQ-AVG3/full-depth → EXP027 transfer diagnostic → EXP029 fresh paired implementation → EXP030 seed42 D3A → EXP030b seed43 one-fold → EXP030c multiseed paired`

EXP032 branches from the frozen seed-42 SEQ/D3A checkpoints into conditional-intensity heads; EXP032b reuses EXP032 conditional predictions bitwise and changes only the extensive activity component. Its group-A half-panel scores do not enter the full-panel baseline chronology.

The later principal lineage is:

`SEQ-01-MIX → EXP035 SEQ-AVG3 slot → EXP036 ETX-S42 mixture → EXP037 STRONGEST_CURRENT`

EXP037 then becomes the exact shared parent for parallel branches:

- tabular/structural: EXP039, EXP041, EXP046, EXP057, EXP058;
- conditional/auxiliary: EXP038, EXP040, EXP043–045, EXP052;
- probabilistic BTYD: EXP047 → EXP049 → EXP050 → EXP051;
- postprocessing and fixed ensemble probes: EXP042, EXP059, EXP060, EXP063;
- temporal/behavioral representation preflights: EXP053 → EXP054/055, plus EXP056 and EXP061/062/064;
- integration/provenance: EXP065 → EXP066 → EXP067 → EXP068.

EXP065 packages EXP037 and EXP051 without creating a new model. EXP067 proves that its `friend` component is byte-identical to EXP037. The teammate `latest` test artifact then adds `occ_meta_B` and `occ_raw_X3`, but both late components lack canonical row-level OOF and share downstream Ridge/greedy ancestry. EXP068 cannot replay an older Ridge stack because the exact member banks and raw meta matrices are absent.

## Deduplication and reruns

1. EXP020 baseline aliases SAMPLE-BASELINE-B-AVG3 and SAMPLE-TB1-AVG3 are array-identical after schema-aligned sorting; NPZ byte hashes differ only because of packaging.
2. EXP029 V1016 BASE and EXP030 V1016 BASE are array-identical aliases under different filenames; they are one baseline artifact, not two runs.
3. EXP030b's seed-43 09-18 pair is embedded/repeated in EXP030c under a different A10/compile execution. Deltas agree to about 1e-5, but the model runs are not byte-identical.
4. EXP032 and EXP032b reuse all four conditional mu/SEQ-composite predictions bitwise. EXP032b is a nested extensive-head composition, not fresh conditional-model training.
5. EXP038 contains BASE-R2, a genuine stochastic rerun of BASE; its +0.000333 difference is an observed execution-noise control, not a new hypothesis.
6. EXP043 run2 is an exact duplicate of run1: predictions, model/optimizer/RNG snapshots, and SHA256 all match.
7. EXP051 OOF is a numerical replay of EXP047 (nested delta differs by about 2e-9); EXP051 remains a distinct canonical unit because it changes optimizer stability and creates production/test artifacts.
8. EXP067 V1 and V2 are partial duplicate directories. V2 is canonical; component_manifest, level_audit, and REPORT are hash-identical, while OOF/reconstruction/summary differ.
9. EXP061, EXP062, and EXP064 test distinct hypotheses but all choose correction scale zero, so their final candidate predictions are baseline-identical. EXP052/054/055 similarly terminate at zero correction on their one-fold/preflight scopes.

EXP048→049 and EXP050→051 are follow-ups, not duplicates: EXP049 corrects an invalid fold comparison; EXP051 changes the production optimizer and succeeds where EXP050 fails.

## Material contradictions and confounders

- EXP020/021 OOF row order differs between files; positional comparison appears inconsistent, but sorting by (cutoff, user_id) proves exact key/target parity.
- EXP023 produces a physical submission despite worsening all four folds.
- S04's original final-blend recipe manifest is absent. The unique 0.05-grid recipe 0.30 E02 + 0.10 E03a + 0.15 DIST + 0.45 S04-B reproduces all 250,000 CSV rows exactly; this is recovered provenance, not an original logged manifest.
- EXP024's causal contrast is FULL-vs-SELF (-0.000002329), not FULL-vs-production (+0.002861); using the latter as the hypothesis delta would be misleading.
- EXP026 changes seed averaging, ensemble weights, and test depth together, so its submission cannot isolate any one change.
- EXP030c's mean paired per-seed delta (-0.000947340) and the delta from scoring averaged OOF (-0.000703801) are both valid but different metrics.
- EXP032/032b absolute scores are group-A half-panel results and incomparable to full-panel project CV. EXP032b passes the fresh-supervision mechanism but rejects the assembled component in the fixed slot.
- EXP035: a D3A average is numerically better in the validation table, but the produced submission uses SEQ-AVG3; machine evidence does not explain the choice.
- EXP038: direct auxiliary-vs-BASE gains are confounded by a +0.000333 BASE rerun shift. The causal FUNNEL-vs-BUYCTRL contrast is much smaller or adverse.
- EXP041 and EXP042: submissions were produced after validation REJECT. EXP042 additionally failed exact DIST test-reference reconstruction.
- EXP048: its original selection penalty compares different fold sets and is invalid. EXP049 is the corrected three-fold analysis.
- EXP051: `summary_oof.json` says REJECT and `PROMOTE_TO_PRODUCTION_EXPERIMENT=NO`; production support says PASS and a submission exists.
- EXP052: the -0.4829 oracle delta is not a realizable model result; the actual pilot selects zero correction.
- EXP058: real-vs-permutation changes sign between standalone and final-slot metrics. These are distinct metrics, not evidence of a consistent mechanism.
- EXP059: a submission exists, but the OOF result is explicitly labelled informational and no final gate is recorded.
- EXP065: ACCEPT means package integrity, not a new validated model improvement.
- EXP067: numerical reconstruction passes tolerance but is not byte-identical, and the supplied SHA manifest has one mismatch.
- EXP067/068 LB values are externally reported or not SHA-bound to the exact CSV; they remain non-confirmed.
- EXP068's historical wCV is a different walk-forward baseline/protocol and is incomparable to canonical outer LOFO against latest.

## Confirmed created submissions in this scope

| Source experiment | File | SHA256 | Validation relation |
|---|---|---|---|
| EXP023 | submission_HOLIDAY-YOY.csv | 41c551a62a663d29382d3d82274f075f223ffe8e0989ecef4c71efd53e9456aa | worse 4/4; exact formula reconstruction |
| S04 | submission_s04_a.csv | a4bf8e347548538d263f87f2d54973ced727862ad76214cadee8ad20cb48c013 | component A |
| S04 | submission_s04_b.csv | 8bf2499f9632dab79b32f0ab7b9183f2fa5519c300b0f950547248a1d6013083 | component B |
| S04 | submission_s04_blend.csv | b515bc150e7b522d2d857fd49dde271e5b9a405e95d163e21d5436314255e0b8 | recovered exact fixed blend |
| EXP025 | submission_SEQ01_mix.csv | ce2f535561a3673c29726833b96fa4444e3b3dc51912c58799ea55a41ef67964 | positive four-fold mix |
| EXP026 | submission_SEQAVG3_mix.csv | 25c1cc5edc559de46c6f2950be78054dd6ece3446c901a2295fb4cbf3f66227b | positive but three changes confounded |
| EXP035 | submission_SEQAVG3_clip289_mix.csv | da644a35ca247f6ef1a11bbf601d17515ddebf63674650f441fb9dada3389b7c | positive vs earlier SEQ-01-MIX |
| EXP037 | submission_STRONGEST_CURRENT.csv | abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda | promoted four-fold recipe |
| EXP041 | submission_RIDGE15.csv | 4f2f4c65ebc1568be78ffaea011996919e98da4ba5beffd0ec83efb9bcfe9397 | forced after REJECT |
| EXP042 | submission_ZERO2D_SHRINK.csv | 9f1cf32671fb18291659b61da232244d370f7c9af2e0cf9d8aebf9eba406d461 | forced after REJECT; lineage mismatch |
| EXP051 | submission_BTYD05.csv | c3cfb4d90f50ceff8f5d8f8aaca072664966fb91018eb0a3fa01195dc38c2932 | fixed recipe favorable; nested verdict REJECT |
| EXP059 | submission_SEQ65_TEMPORAL_HEAVY.csv | 33c6a4e70dbc0d061508c8179e3b820ffa00829d134d802fc0767ac3f4b69248 | OOF marked informational |
| EXP060 | submission_LEVEL_MINUS_006.csv | 1b40f67d119d0dcc4798a4da5612707b8d44f1dfe3fa20b28c28b836c2c8c0f1 | no local CV |

All 13 files exist physically and were checksum-verified. The early Holiday, SEQ01, SEQAVG3, and S04 A/B/final CSV formulas reproduce saved test arrays to six decimals. None has a SHA-bound public-LB event in this evidence zone. EXP065 only copies/packages EXP037 and EXP051, EXP067 creates an audit reconstruction, and EXP068 references a historical teammate CSV; none of those is counted as a newly created submission here.

## Missing evidence

- No SHA-bound LB event for any of the 13 created submissions above in this strategy-results evidence set.
- EXP019 has no completed selected DIST G90 arm.
- EXP028 has no machine JSON/CSV/log/OOF; its preflight row is necessarily report-level and explicitly weak.
- EXP026 has saved full-depth test predictions but not the exact seed-43/44 full-depth checkpoints.
- EXP027 is diagnostic-only and has no canonical candidate OOF/CV.
- EXP032/032b have no full-panel or LOFO validation and no submission.
- No independent run evidence was found for EXP031, EXP033, or EXP034.
- STRATEGY_05 contains description-only variants without raw run artifacts; the root tier_a_fixed_mix_oof.csv is a derived analysis without a manifest/lineage. Neither was promoted to a canonical experiment.
- EXP036/037 total training runtimes are incomplete; only isolated run costs are available.
- EXP038 has no multi-fold validation.
- EXP040 exact production TCN checkpoint and saved conditional-head weights are absent.
- EXP047/051 have a validation/production policy disagreement that no run manifest resolves.
- EXP052–058 mostly stop at one fold or preflight and therefore lack four-fold OOF.
- EXP060 has no local-CV evaluation.
- EXP066–068 lack canonical row-level OOF for teammate late components.
- EXP067 cannot reduce total effective CAP ancestry to a proven scalar.
- EXP068 lacks the historical OOF member checkpoints, test prediction bank, exact raw meta matrix, and exact LB-to-CSV binding.

## Completeness checks

- The JSONL has **51 lines and 51 unique IDs**, each with all required top-level fields; 16 normalized families are represented.
- EXP035–068 is contiguous at 34 IDs. EXP019–034 contributes 16 numbered/suffixed run units; EXP031/033/034 are absent by evidence, uppercase EXP_032 is a specification, and unnumbered S04 is added from machine artifacts.
- Each canonical record except explicitly weak EXP028 has at least one machine artifact or concrete run artifact in evidence.
- Orphan prediction families relevant to this window were associated to their canonical experiment where manifests/hashes allowed it.
- The results-tree pass covered 484 files (120 JSON, 239 CSV, 2 NPZ, 41 Markdown, and concrete scripts/other files); directory names alone were never treated as runs.
- Thirteen physical created submission files were checksum-verified; none has a verified SHA-bound LB event in this audit.
- EXP061/062/064 scripts and feature caches are associated despite producing baseline-identical zero corrections.
- Blocked EXP066–068 are retained as negative provenance evidence rather than dropped.
- Forbidden STATE/AGENT/HISTORY/README/index/roadmap/TODO/master-summary claims were not used to populate facts.
