# exp_048 — SELECTION-MISMATCH / SELECTION-MATCHED CV

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_048_selection_mismatch_cv`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_048_selection_mismatch_cv`
- **Original source:** `experiments/exp_048_selection_mismatch_cv.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** sequence model, BTYD
- **Features:** freshness/conditional features, Search/Catalog decomposition
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** `1.766883357 / 1.760509577 / 1.748629224 / 1.741278566`, wCV
- **Known score:** `1.766883357 / 1.760509577 / 1.748629224 / 1.741278566`, wCV
- **Seed:** Shuffle uses 100 fixed seed-42 permutations inside
- **Postprocessing:** shape comparisons rather than global-level wins.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_048 — SELECTION-MISMATCH / SELECTION-MATCHED CV

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** a28a71f
- **Prefix:** `SELMATCH_EXP048`
- **SELECTION-MISMATCH VERDICT:** **TECHNICAL_INCONCLUSIVE**
- **PROMOTE_TO_PRODUCTION_AUDIT:** **NO**

## Exact mechanism and audit status

`STRONGEST_CURRENT` reconstructed exactly from raw OOF: fold scores
`1.766883357 / 1.760509577 / 1.748629224 / 1.741278566`, wCV
`1.747509863`. Row keys, target, component predictions, fold sizes, paths,
hashes, order, and fold calibration are in `audit_manifest.json`.

The competition universe is the intersection of any-row activity in G1
`2025-11-16..12-15`, G2 `2025-12-16..2026-01-14`, and G3
`2026-01-15..02-13`: 250,000 users in every block. The test panel at
`2026-02-13` is exactly the same three-past-block rule. Standard validation
panels are additionally conditioned on this globally future selection because
the supplied daily log contains only those selected 250k users.

For each V, target dates are `(V,V+30]`; selection variables use only
`(V+30,V+120]`, split into three 30-day blocks. The date sets are disjoint.
`future_min_events` is the minimum across F1/F2/F3 of daily
`searches+cat+to_cart+to_ord`; an inactive block makes it zero. Full source
dates are preserved in `audit_manifest.json`.

## Support and prevalence

|fold|pi(k=0)|pi(k=1)|pi(k=2)|pi(k=3)|overlap with G|
|---|---|---|---|---|---|
|2025-09-04|0.0000|0.0036|0.0348|0.9616|48/90|
|2025-09-18|0.0000|0.0004|0.0244|0.9752|62/90|
|2025-10-02|0.0000|0.0007|0.0304|0.9689|76/90|
|2025-10-16|0.0000|0.0000|0.0000|1.0000|90/90|

The 10-16 F1/F2/F3 windows match G1/G2/G3 exactly and every validation row has
`k=3`; it is therefore excluded from pseudo-matched aggregation. Per-fold max
weights and ESS are in `matched_support.csv`; minimum ESS fraction is
**0.570**.

However, `pi_ref(k=0) > 0` while **none** of the three eligible folds contains
`k=0`. All three folds are therefore unsupported under the registered rule;
the reported k>0 reweighting is a conditional sensitivity diagnostic, not the
requested full-reference matched estimate. This forces `TECHNICAL_INCONCLUSIVE`.

## Natural-continuation reference

Sixteen equally weighted weekly landmarks `2025-04-03..2025-07-17` use the
same three-past-block panel rule and all satisfy `V+120 <= 2025-11-15`.

|k|pi_ref|cluster-bootstrap 95%|
|---|---|---|
|0|0.00495|[0.00478,0.00512]|
|1|0.01689|[0.01658,0.01721]|
|2|0.06304|[0.06237,0.06365]|
|3|0.91512|[0.91430,0.91601]|

Limitation: the reference is still built inside the globally selected 250k
universe and is not an unbiased estimate of the full platform population.

## Standard / survivor / pseudo-matched rankings

|family|scheme|candidate|rank|delta to STRONGEST|folds correct sign|
|---|---|---|---|---|---|
|standalone|A_STANDARD_4F|ETX-AVG3|1|+0.001100|0|
|standalone|A_STANDARD_4F|SEQ-AVG3|2|+0.002124|0|
|standalone|A_STANDARD_4F|DIST|3|+0.003106|0|
|standalone|A_STANDARD_4F|CAP|4|+0.013132|0|
|standalone|A_STANDARD_4F|BTYD|5|+0.053191|0|
|reference|A_STANDARD_4F|UNC|1|+0.003996|0|
|incremental|A_STANDARD_4F|BTYD05_FRESH1|1|-0.000467|4|
|incremental|A_STANDARD_4F|BTYD05|2|-0.000321|4|
|incremental|A_STANDARD_4F|FRESH|3|-0.000225|4|
|incremental|A_STANDARD_4F|ZERO2D|4|-0.000025|2|
|incremental|A_STANDARD_4F|SEQ_SLOT_50|5|+0.000000|0|
|incremental|A_STANDARD_4F|SEQ_SLOT_25|6|+0.000053|2|
|incremental|A_STANDARD_4F|SEQ_SLOT_75|7|+0.000100|0|
|standalone|B_SURVIVOR_K3|ETX-AVG3|1|+0.001110|0|
|standalone|B_SURVIVOR_K3|SEQ-AVG3|2|+0.002128|0|
|standalone|B_SURVIVOR_K3|DIST|3|+0.003131|0|
|standalone|B_SURVIVOR_K3|CAP|4|+0.013163|0|
|standalone|B_SURVIVOR_K3|BTYD|5|+0.053083|0|
|reference|B_SURVIVOR_K3|UNC|1|+0.003950|0|
|incremental|B_SURVIVOR_K3|BTYD05_FRESH1|1|-0.000477|4|
|incremental|B_SURVIVOR_K3|BTYD05|2|-0.000337|4|
|incremental|B_SURVIVOR_K3|FRESH|3|-0.000220|4|
|incremental|B_SURVIVOR_K3|ZERO2D|4|-0.000016|2|
|incremental|B_SURVIVOR_K3|SEQ_SLOT_50|5|+0.000000|0|
|incremental|B_SURVIVOR_K3|SEQ_SLOT_25|6|+0.000052|2|
|incremental|B_SURVIVOR_K3|SEQ_SLOT_75|7|+0.000101|0|
|standalone|C_PSEUDO_MATCHED_3F|ETX-AVG3|1|+0.001654|0|
|standalone|C_PSEUDO_MATCHED_3F|SEQ-AVG3|2|+0.001883|0|
|standalone|C_PSEUDO_MATCHED_3F|DIST|3|+0.002992|0|
|standalone|C_PSEUDO_MATCHED_3F|CAP|4|+0.011743|0|
|standalone|C_PSEUDO_MATCHED_3F|BTYD|5|+0.051767|0|
|reference|C_PSEUDO_MATCHED_3F|UNC|1|+0.004369|0|
|incremental|C_PSEUDO_MATCHED_3F|BTYD05_FRESH1|1|-0.000551|3|
|incremental|C_PSEUDO_MATCHED_3F|BTYD05|2|-0.000340|3|
|incremental|C_PSEUDO_MATCHED_3F|FRESH|3|-0.000252|3|
|incremental|C_PSEUDO_MATCHED_3F|ZERO2D|4|-0.000056|3|
|incremental|C_PSEUDO_MATCHED_3F|SEQ_SLOT_25|5|-0.000040|2|
|incremental|C_PSEUDO_MATCHED_3F|SEQ_SLOT_50|6|+0.000000|0|
|incremental|C_PSEUDO_MATCHED_3F|SEQ_SLOT_75|7|+0.000196|0|

Full pairwise deltas and bootstrap rank
intervals are in `rankings.csv`, `pairwise_deltas.csv`,
`rank_correlations.csv`, and `bootstrap.csv`. Differences below 0.0002 or whose
bootstrap interval spans zero are treated as unresolved, not rank changes.

|family|scheme A|scheme B|Spearman|Kendall|
|---|---|---|---|---|
|standalone|A_STANDARD_4F|B_SURVIVOR_K3|1.000|1.000|
|standalone|A_STANDARD_4F|C_PSEUDO_MATCHED_3F|1.000|1.000|
|standalone|B_SURVIVOR_K3|C_PSEUDO_MATCHED_3F|1.000|1.000|
|incremental|A_STANDARD_4F|B_SURVIVOR_K3|1.000|1.000|
|incremental|A_STANDARD_4F|C_PSEUDO_MATCHED_3F|0.964|0.905|
|incremental|B_SURVIVOR_K3|C_PSEUDO_MATCHED_3F|0.964|0.905|

Raw order-changing pairs: **incremental: SEQ_SLOT_25 vs SEQ_SLOT_50 (A +0.000053, C -0.000040)**.
Order changes surviving the 0.0002 materiality rule: **none**.

## Bootstrap and shuffle controls

500 cluster-bootstrap replicates resample `user_id`; the point-estimate weighted
offsets are held fixed inside bootstrap (explicit diagnostic approximation).
The point matched scores themselves use exact weighted optimal offsets.

|candidate|boot mean|10/90|2.5/97.5|P(delta<0)|shuffle central 90%|outside|
|---|---|---|---|---|---|---|
|BTYD05|-0.000334|[-0.000392,-0.000278]|[-0.000432,-0.000241]|1.000|[-0.000041,+0.000031]|True|
|FRESH|-0.000250|[-0.000293,-0.000207]|[-0.000315,-0.000188]|1.000|[-0.000023,+0.000016]|True|
|ZERO2D|-0.000057|[-0.000073,-0.000041]|[-0.000080,-0.000035]|1.000|[-0.000011,+0.000015]|True|
|SEQ_SLOT_25|-0.000037|[-0.000078,+0.000011]|[-0.000106,+0.000031]|0.848|[-0.000029,+0.000024]|False|
|SEQ_SLOT_50|+0.000000|[+0.000000,+0.000000]|[+0.000000,+0.000000]|0.000|[-0.000000,+0.000000]|False|
|SEQ_SLOT_75|+0.000193|[+0.000144,+0.000234]|[+0.000125,+0.000261]|0.000|[-0.000023,+0.000030]|False|
|BTYD05_FRESH1|-0.000544|[-0.000625,-0.000466]|[-0.000674,-0.000421]|1.000|[-0.000046,+0.000035]|False|

Shuffle uses 100 fixed seed-42 permutations inside
`fold × rec_buy_bin × w180_days_buy_bin` and preserves every stratum size.

## Selection penalty and magnitude

`selection_penalty = delta_survivor_conditioned - delta_pseudo_matched`.

|comparison|standard|survivor k=3|pseudo-matched|selection penalty|
|---|---|---|---|---|
|CAP - STRONGEST|+0.013132|+0.013163|+0.011743|+0.001420|
|DIST - STRONGEST|+0.003106|+0.003131|+0.002992|+0.000139|
|ETX-AVG3 - STRONGEST|+0.001100|+0.001110|+0.001654|-0.000544|
|SEQ-AVG3 - STRONGEST|+0.002124|+0.002128|+0.001883|+0.000246|
|BTYD - STRONGEST|+0.053191|+0.053083|+0.051767|+0.001316|
|BTYD05 - STRONGEST|-0.000321|-0.000337|-0.000340|+0.000003|
|FRESH - STRONGEST|-0.000225|-0.000220|-0.000252|+0.000032|
|ZERO2D - STRONGEST|-0.000025|-0.000016|-0.000056|+0.000040|
|SEQ_SLOT_25 - STRONGEST|+0.000053|+0.000052|-0.000040|+0.000091|
|SEQ_SLOT_50 - STRONGEST|+0.000000|+0.000000|+0.000000|+0.000000|
|SEQ_SLOT_75 - STRONGEST|+0.000100|+0.000101|+0.000196|-0.000094|
|BTYD05_FRESH1 - STRONGEST|-0.000467|-0.000477|-0.000551|+0.000074|
|DIST - CAP|-0.010026|-0.010032|-0.008752|-0.001280|
|DIST - UNC|-0.000890|-0.000820|-0.001377|+0.000558|
|ETX-AVG3 - SEQ-AVG3|-0.001025|-0.001019|-0.000229|-0.000790|

Largest absolute measured penalty is **0.001420**. Evidence that mismatch
alone explains the systematic 0.0004–0.0006 floor: **NO for incremental candidates; k=0 support is absent in every eligible fold**;
largest incremental penalty is **0.000094**.
Evidence of an incremental candidate at scale >=0.001: **NO**.

|candidate|survivor fixed-fold delta|survivor shape-only delta|level component|
|---|---|---|---|
|BTYD05|-0.000337|-0.000337|+0.000000|
|FRESH|-0.000220|-0.000220|-0.000000|
|ZERO2D|-0.000016|-0.000016|-0.000001|
|SEQ_SLOT_25|+0.000052|+0.000052|+0.000000|
|SEQ_SLOT_50|+0.000000|-0.000000|+0.000000|
|SEQ_SLOT_75|+0.000101|+0.000101|-0.000000|
|BTYD05_FRESH1|-0.000477|-0.000478|+0.000000|

Fixed-versus-shape decomposition and residual-correction correlations in
`slice_diagnostics.csv` separate level shift from residual alignment. The
matched endpoint always refits one weighted fold offset, so its deltas are
shape comparisons rather than global-level wins.

## What can and cannot be concluded

This is a pseudo-selection-matched sensitivity analysis, not an unbiased test
estimate. It identifies how ranking changes under a fixed k-only continuation
reference within the selected universe. It cannot recover users absent from the
250k data, establish platform-population performance, or justify new gates,
weights, test inference, or a submission.

## Verdict

**TECHNICAL_INCONCLUSIVE**. No model training, test inference, public-LB use, or
submission was performed. If non-actionable, the expected upside is below that
of a structurally new `Search/Catalog future-GMV target decomposition`, which
would attack target structure rather than this small/unstable selection axis;
that decomposition is not implemented here.
