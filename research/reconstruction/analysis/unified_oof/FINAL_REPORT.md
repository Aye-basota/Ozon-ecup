# Unified OOF final report

## 1. Cache status

The exact teammate occurrence checkpoint bank is **missing**. Exact TEST components and native validation summaries were recovered; canonical row-level OOF for `occ_meta_B/final6h_B`, `occ_raw_X3/extra90_3`, and `latest` was not. Full replay was not started because the missing core cache makes it a 20–30 CPU-hour reconstruction. The one permitted cheap fallback was run in 35.9s.

## 2. Alignment

Canonical `(cutoff,user_id)` alignment is complete for EXP-037 primitives, BTYD and fallback: **770,616/770,616**, zero missing/extras/duplicates, target equality PASS. TEST `user_id` alignment is 250,000/250,000. EXP-037 replay error is `4.530e-07`; latest recipe replay error is `8.882e-16`.

## 3. Occurrence offline performance

Exact teammate sources remain unscorable on canonical folds. Native summary gains (`occ_meta_B` about −0.001767; `occ_raw_X3` about −0.001625) are against teammate base wCV 1.749804 and are not substituted for canonical OOF. The locked fallback occurrence-only LightGBM overlay gives fixed-scale wCV **1.746947315**, delta **-0.000562547**, 4/4.

## 4. Incremental utility

Nested inner-fold scalar selection gives:

| baseline | delta wCV | folds | last fold | mean lambda |
|---|---:|---:|---:|---:|
| A_EXP037 | -0.000457508 | 4/4 | -0.000441084 | 0.8079 |
| B_EXP037_SEQ65 | -0.000416141 | 4/4 | -0.000400849 | 0.7756 |
| C_EXP037_BTYD05 | -0.000379807 | 4/4 | -0.000376829 | 0.7403 |
| D_EXP037_SEQ65_BTYD05 | -0.000343759 | 4/4 | -0.000342167 | 0.7097 |


The decisive row is occurrence after the canonical `SEQ65+BTYD05` compound: delta `-0.000343759`, 4/4, last fold `-0.000342167`. This is secondary evidence below the −0.0005 incremental production floor, while total candidate delta versus EXP-037 is `-0.000906458`.

## 5. Signal overlap

OOF correction overlap is quantified in `CORRECTION_OVERLAP.csv`. The fallback is a real residual direction, but it is a newly cross-fitted mechanism probe—not an exact reconstruction of the teammate models. TEST centered correlation of fallback correction with `latest−STRONGEST` is **0.256079**; compound versus latest is **0.251486**.

## 6. Best honest combination

Best measured canonical recipe is `0.95·SEQ65 + 0.05·BTYD + λ·fallback_occ_delta`, with the occurrence model/features fixed in `run_occurrence_fallback.py`. Nested OOF uses outer-specific lambdas; full-OOF production scalar would be **λ=0.912822**. Nested wCV is **1.746603405**, total delta versus EXP-037 **-0.000906458**, 4/4. Its incremental delta over compound is **-0.000343759**, secondary rather than an automatic production gain. This is not declared an exact teammate+canonical union because teammate OOF is missing.

## 7. Test-space sanity

Fallback TEST/OOF correction variance ratio is **0.968208**. Fallback-vs-latest centered correction correlation is **0.256079**, projection of latest on fallback is **0.233708**. The full canonical compound+fallback TEST candidate has centered correction correlation **0.348921** with `latest−STRONGEST`. `compound+latest` remains blocked: without canonical `z_latest`, no honest λ can be selected; no public-LB weight optimization was performed.

## 8. Recommendation

**RUN ONE CHEAP FOLLOW-UP**

Decision code: `YES_MECHANISM_EXACT_TEAMMATE_RECIPE_PENDING_CACHE`. The fallback demonstrates a non-absorbed occurrence mechanism, but exact teammate+canonical OOF remains unmeasured. Retrieve only the compact teammate fold/TEST NPZ and rerun this audit. Do not rebuild the full 9.7 GB feature cache and do not submit `latest + λ·compound` before exact latest OOF exists.
