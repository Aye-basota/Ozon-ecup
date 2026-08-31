# EXP086 — Occurrence Shock Residual

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp086_occurrence_shock`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP086_OCCURRENCE_SHOCK`
- **Original source:** `research/new_directions/EXP086_OCCURRENCE_SHOCK`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** LightGBM, BTYD, Ridge, two-part / hurdle, ensemble, blend
- **Features:** recency, funnel features, occurrence features, gap/burst features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | 8 raw occurrence heads | frozen `OCC_QUEUE` / `train_occ_child` | `1[GMV30 > 0]` | six 227-column and two 202-column normalized-long cutoff-safe tables | last 10–24 eligible weekly cutoffs, always `train cutoff + 30 <= validation cutoff` | exact source TEST child; not run after NO_GO | YES |
- **Known score:** nested `Delta MSE = +0.000240322` и `Delta RMSLE = +0.00006833`, то есть correction ухудшает результат;
- **Seed:** Bootstrap выполнен как 2,000-replicate Poisson cluster bootstrap по `user_id` сразу через все transitions, seed `20260829`. CI целиком находится на стороне вреда. Expected robust `Delta RMSLE = +0.00006833`, а не improvement.
- **Postprocessing:** Incremental semantics здесь конкретна: “насколько восемь cutoff-capped incidence estimators и их meta learner изменяют вероятность следующего 30-дневного occurrence относительно `p_base` данного пользователя”. Это не global activity level и не повтор прямого GMV30 learner.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 60 files, 2 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP086 — Occurrence Shock Residual

## Verdict

**NO_GO.** Teammate occurrence family действительно была восстановлена на уровне первичных incidence-outputs: восемь raw probability heads, базовый hurdle-state (`p_base`, `mu`), all-eight walk-forward meta probability и error-risk trust gate. На четырёх clean cutoffs получена новая fully-purged lineage без late survivorship windows.

Но user-specific post-production signal не подтвердился. Для deployable joint raw/meta direction:

- weighted purged post-span `rho = -0.005541`;
- latest `rho = -0.005481`;
- положительный знак только на `1/3` transitions;
- nested `Delta MSE = +0.000240322` и `Delta RMSLE = +0.00006833`, то есть correction ухудшает результат;
- bootstrap 95% CI для `Delta MSE`: `[+0.000135585, +0.000347981]`, `P(gain) = 0.000`.

Даже отдельные естественные raw/meta directions далеко ниже gate: weighted post-span `rho = +0.000913` для raw core и `+0.000039` для meta core. Поэтому focused refinement, TEST inference и submission candidate запрещены заранее заданными gates.

## Occurrence pipeline reconstruction

### Provenance result

Исходный mechanism найден в полном teammate repository, а не аппроксимирован похожей моделью. Главный source — `continue_best_bas_final6h.py`, SHA256 `2b0fb5b26ef4fad10a1604c7abf13a9db23b9cf7051ed61f44e54d95c7a7e041`. Дополнительные materialization/final-blend sources имеют SHA256 `1d7d436ecc95edcbf5adf489695e5380c4b2174c29e9f7fcbf19ebf8e8655d83` и `f106bfc95ee6696f6b162aa59723eae218a68f05cf67d9db2c16416f776167e1`.

| artifact | pipeline | target | features | training dates | TEST path | reproducible |
| --- | --- | --- | --- | --- | --- | --- |
| 8 raw occurrence heads | frozen `OCC_QUEUE` / `train_occ_child` | `1[GMV30 > 0]` | six 227-column and two 202-column normalized-long cutoff-safe tables | last 10–24 eligible weekly cutoffs, always `train cutoff + 30 <= validation cutoff` | exact source TEST child; not run after NO_GO | YES |
| base hurdle `p_base`, `mu` | `recent_hurdle`, two-part | incidence + positive-case `log1p(GMV30)` | 227 columns | all eligible weekly cutoffs | exact source TEST path; not run after NO_GO | YES |
| `meta_raw` | deterministic state extraction | none | first 72 recent aggregate columns | state at validation cutoff | exact source TEST path; not run after NO_GO | YES |
| all8 meta occurrence | `walk_meta_occ(all8,power=1.7,leaves=31)` | `1[GMV30 > 0]` | 72 meta-raw + `p_base/mu/table_core` + 8 raw `p/logit/delta` pairs | only earlier fully known folds | `final_meta_occ`, seed 7900; not run after NO_GO | YES |
| risk trust gate | `walk_risk_gate(all8,power=1.7)` | false-one and severe-over indicators | meta matrix + three hurdle-error columns | only earlier fully known folds | `final_risk_gate`, seeds 8901/8902; not run after NO_GO | YES |
| EXP086 intermediates | primary replay before downstream overlay | all incidence/reliability outputs | `p_base`, `mu`, 8 raw `p`, meta `p`, risk and shock directions | all four requested cutoffs | `occurrence_predictions.parquet` | YES |
| `occ_raw_X3.csv` | `xraw_occ_r10_fast_adapt__...` | downstream log-GMV candidate | `occ_r10_fast` adaptive overlay + frozen Ridge/greedy anchor | old folds 09-04, 09-18, 10-02, 10-16 | original component SHA256 `0ac3f2...27356` | TEST YES; full old downstream OOF NO |
| `occ_meta_B.csv` | `metaocc_l31_risk__...` | downstream log-GMV candidate | all8 meta + risk gate + same anchor | old folds 09-04, 09-18, 10-02, 10-16 | original component SHA256 `8d90a0...e7989d8` | TEST YES; full old downstream OOF NO |
| `latest.csv` | log-space `0.12 friend + 0.16 meta_B + 0.72 raw_X3`, then `z >= 0` | final submission vector | three TEST vectors | no new fit at blend step | original SHA256 `7ef5b2...ba8e722` | YES |

Полная machine-readable таблица находится в `occurrence_pipeline_reconstruction.csv`; source hashes, manifests и exact artifact relation — в `provenance.json`.

### Raw-head configuration

Общие параметры raw heads: LightGBM binary logloss, learning rate `0.035`, bagging fraction `0.90`, L2 `14`, L1 `1`, max bin `127`, seed `42`. Различаются только замороженные параметры из teammate `OCC_QUEUE`:

| head | cutoffs | tau | rounds | leaves | min leaf | feature mode | feature fraction |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `occ_r10_fast` | 10 | 55 | 380 | 31 | 520 | all/227 | 0.82 |
| `occ_r12_wide` | 12 | 70 | 430 | 79 | 380 | all/227 | 0.72 |
| `occ_r14_multiscale` | 14 | 85 | 430 | 47 | 430 | multiscale/202 | 0.90 |
| `occ_r16_bal` | 16 | 100 | 440 | 47 | 430 | all/227 | 0.84 |
| `occ_r18_wide` | 18 | 125 | 470 | 63 | 420 | all/227 | 0.76 |
| `occ_r20_shallow` | 20 | 155 | 500 | 23 | 760 | all/227 | 0.90 |
| `occ_r22_stable` | 22 | 180 | 500 | 31 | 650 | all/227 | 0.80 |
| `occ_r24_multiscale` | 24 | 220 | 520 | 31 | 700 | multiscale/202 | 0.88 |

Base hurdle повторяет teammate `recent_hurdle`: 420 rounds, learning rate `0.035`, 63 leaves, min leaf 260, feature fraction `0.78`, bagging `0.88`, L2 `14`, L1 `1.5`, max bin `63`, seed `42`. Meta classifier: 420 trees, learning rate `0.03`, 31 leaves, min child 450, feature fraction `0.78`, seeds `7100+i`. Два risk classifiers: 320 trees, 23 leaves, min child 550, seeds `8100+i` и `8200+i`.

### What is and is not exact

Первичный occurrence mechanism и его TEST code path восстановлены однозначно. Оригинальные full TEST vectors `occ_raw_X3`, `occ_meta_B` и `latest` также найдены и hash-verified.

Не найден только старый полный historical OOF bank downstream Ridge/greedy anchor, поверх которого materialized `occ_raw_X3` и `occ_meta_B`. Поэтому EXP086 **не заменяет** этот missing bank похожей моделью и не выдаёт reconstructed downstream submissions за exact OOF. Purged evidence использует максимально первичные exact outputs occurrence pipeline, до этого anchor: raw/meta probabilities, hurdle state и risk reliability.

## Novelty vs previous experiments

Mechanism не эквивалентен закрытым branches, поэтому `STOP_DUPLICATE` не применяется.

| previous family | что было раньше | новая информация teammate occurrence |
| --- | --- | --- |
| EXP063 occurrence revisit | один S1–E11 two-part GMV member | восемь incidence-only heads и disagreement с собственным hurdle-state |
| EVENT-ORDER | ordered daily funnel transitions | aggregate-state temporal incidence estimators, не event tokens |
| OPEN-FUNNEL | unresolved Search/Cart after last purchase | broad incidence ensemble без handcrafted unresolved-funnel state |
| BURST/GAP | fixed episode/gap summaries | recency/capacity disagreement между независимо обученными incidence heads |
| BTYD | параметрический repeat-purchase process | nonparametric cutoff-capped LightGBM occurrence probability |
| MHZ hazard | multi-horizon hazard/count targets | один exact 30d incidence target; novelty — disagreement с own state, не новый horizon |
| generic count/value | count bins и conditional value | future count/value не предсказываются; `mu` только переводит probability shock в log-GMV units |
| EXP075 trajectory | position-specific daily/weekly path | частично тот же aggregate feature universe, но другой supervised object: relative next-30d incidence shock |

Incremental semantics здесь конкретна: “насколько восемь cutoff-capped incidence estimators и их meta learner изменяют вероятность следующего 30-дневного occurrence относительно `p_base` данного пользователя”. Это не global activity level и не повтор прямого GMV30 learner.

## Purged folds

| validation cutoff | target window | max allowed prior target end | role |
| --- | --- | --- | --- |
| 2025-07-03 | 2025-07-04 — 2025-08-02 | — | warm-start calibration only |
| 2025-08-07 | 2025-08-08 — 2025-09-06 | 2025-08-02 | transition 1 |
| 2025-09-11 | 2025-09-12 — 2025-10-11 | 2025-09-06 | transition 2 |
| 2025-10-16 | 2025-10-17 — 2025-11-15 | 2025-10-11 | transition 3 |

Spacing ровно 35 дней. В каждом transition предшествующий 30-day target полностью известен за пять дней до следующего cutoff. Последний target заканчивается 2025-11-15. Late survivorship-conditioned windows не использованы.

Для каждого cutoff сохранены 8 raw `p`, `p_base`, `mu`, `p_meta`, risk output и все raw/perpendicular candidate vectors. Всего materialized 40 component NPZ и 40 metadata JSON; metadata фиксируют feature names, exact train cutoffs, maximum train-target end, seeds, configs, source SHA и output SHA.

## Candidate construction

Из outputs pipeline заранее зафиксированы четыре небольших направления:

```text
raw core       = (p_occ_r10_fast - p_base) * mu
meta core      = (p_meta_all8_l31 - p_base) * mu
disagreement   = (p_meta_all8_l31 - p_occ_r10_fast) * mu
risk-weighted  = exact teammate trust gate applied to meta core
```

Для каждого fold и каждого direction выполнено `d <- d - mean(d)`, затем два раза least-squares projection из span `[1, cap, unc, dist, seq, etx, z_production_like]`. Production-like baseline — exact EXP082 recipe `0.10 cap + 0.20 unc + 0.25 dist + 0.225 seq + 0.225 etx`. Primary residual всегда `target_log - z_production_like`; base-hurdle residual используется только как weak-overlap diagnostic.

После проекции mean каждого deployable correction не превосходит численный шум порядка `1e-20`. Perpendicular RMS fraction на transitions составляет `0.966–0.989` для raw и `0.904–0.986` для meta: отрицательный результат не вызван уничтожением почти всей геометрии самой проекцией.

## Raw/meta occurrence results

В таблице `raw rho` и `meta rho` — отдельные post-span directions; `joint rho` — same-fold oracle headroom и не deployable evidence; `post-span rho` и `Delta RMSLE` — joint coefficients, fit только на полностью известных прошлых folds.

| Fold | raw rho | meta rho | joint rho | post-span rho | Delta RMSLE |
| ---- | ------: | -------: | --------: | ------------: | ----------: |
| 2025-08-07 | -0.000031 | +0.003635 | +0.003793 | +0.000031 | +0.00001152 |
| 2025-09-11 | -0.005209 | -0.011498 | +0.011565 | -0.008448 | +0.00006251 |
| 2025-10-16 | +0.004210 | +0.004908 | +0.005507 | -0.005481 | +0.00008603 |
| **weighted** | **+0.000913** | **+0.000039** | — | **-0.005541** | **+0.00006833** |

Mandatory scalar audit:

| fold | direction | rho | b | G | oracle amplitude | oracle MSE gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 08-07 | raw | -0.000031 | -0.000003069 | 0.002992550 | -0.001026 | -0.000000003 |
| 08-07 | meta | +0.003635 | +0.000643331 | 0.009630805 | +0.066799 | -0.000042974 |
| 09-11 | raw | -0.005209 | -0.000564110 | 0.003769344 | -0.149657 | -0.000084423 |
| 09-11 | meta | -0.011498 | -0.001665229 | 0.006741973 | -0.246994 | -0.000411302 |
| 10-16 | raw | +0.004210 | +0.000452468 | 0.003807647 | +0.118831 | -0.000053767 |
| 10-16 | meta | +0.004908 | +0.000666030 | 0.006071352 | +0.109701 | -0.000073064 |

Oracle sign и scale меняются резко между folds; поэтому same-fold gains нельзя переносить в deployment.

## Production overlap

Weighted correlation ladder для естественных, ещё не переориентированных past-fit коэффициентами directions:

| candidate | rho vs weak hurdle residual | rho vs strong residual | rho after full-span projection |
| --- | ---: | ---: | ---: |
| raw core | +0.018154 | +0.002997 | +0.000913 |
| meta core | +0.017908 | +0.002129 | +0.000039 |
| raw/meta disagreement | +0.005082 | -0.000039 | -0.000720 |
| risk-weighted meta | +0.015432 | +0.000602 | -0.001201 |

Это именно сценарий “occurrence переоткрывает production information”. У raw/meta имеется около `0.018` correlation против weak hurdle residual, но после strong production baseline остаётся только `0.002–0.003`, а после удаления полного production span — практически ноль. Никакого устойчивого residual channel сверх production не обнаружено.

## Joint covariance

Pooled same-fold covariance на трёх transitions:

```text
G = [[0.003680261, 0.001788379],
     [0.001788379, 0.006771451]]
b = [0.000096940, -0.000003286]
a_oracle = [0.030489, -0.008538]
condition number = 2.652
```

Pooled same-fold joint oracle gain всего `0.000002984` MSE. Conditional raw contribution given meta — `0.000002982`; conditional meta contribution given raw — `0.000000430`. Иными словами, meta даёт маленькую условную добавку к raw в same-fold oracle geometry, но не deployable stability.

Past-only coefficients подчёркивают instability:

| applied fold | raw coefficient | meta coefficient |
| --- | ---: | ---: |
| 2025-08-07 | -0.118915 | 0.000000 |
| 2025-09-11 | -0.064174 | +0.076748 |
| 2025-10-16 | -0.065558 | -0.101972 |

На latest fold оба deployed signs расходятся с latest same-fold oracle signs (`+0.076513`, `+0.086122`), что и превращает маленький mathematical headroom в фактический вред.

## Segment diagnostics

Segments использованы только для объяснения, не для tuning. Joint nested correction имеет mean около нуля на каждом fold и в большинстве сегментов ухудшает MSE.

- По baseline deciles улучшение появляется только в decile 7 (`Delta MSE -0.0000836`) и едва заметно в decile 9 (`-0.00000836`); остальные deciles ухудшаются.
- По recency ни один bucket не даёт устойчивого weighted gain; минимальный вред у `>90/never`, но знак всё равно положительный.
- По purchase frequency все buckets ухудшаются; локального стабильного frequency mechanism нет.
- По recent activity единственный отрицательный bucket — 26–30 active days (`Delta MSE -0.0000237`), тогда как крупнейший bucket 6–15 days ухудшается на `+0.0003549`.

Локальные отрицательные cells малы, множественны и не согласованы между независимыми axes; использовать их для rescue-тюнинга противоречило бы NO_GO gate.

## Mathematical headroom

| quantity | value |
| --- | ---: |
| required gap MSE | 0.004909274 |
| occurrence nested gain | **-0.000240322** |
| fraction of gap, signed | **-4.8953%** |
| fraction of gap that can robustly be closed | **0.0000%** |
| pooled same-fold oracle gain | 0.000002984 |
| pooled same-fold oracle fraction of gap | 0.0608% |

Negative “gain” означает observed worsening. Даже nondeployable pooled oracle ceiling закрывает лишь `0.0608%` required MSE gap; это на два порядка ниже practically meaningful scale.

Bootstrap выполнен как 2,000-replicate Poisson cluster bootstrap по `user_id` сразу через все transitions, seed `20260829`. CI целиком находится на стороне вреда. Expected robust `Delta RMSLE = +0.00006833`, а не improvement.

## TEST geometry

Не рассчитывалась. `STRONG_GO` не пройден, поэтому final occurrence training, TEST zero-centering/projection, `perp_fraction`, correlations с ORTH/EXP075 и amplitude application не выполнялись. Это предотвращает превращение диагностического отрицательного branch в submission candidate.

## Output

Submission candidate **не создан** и SHA candidate отсутствует, как требует NO_GO protocol. Основные reproducibility artifacts:

- `occurrence_predictions.parquet` — все первичные fold outputs и projected signals;
- `candidate_signal_audit.csv`, `raw_meta_joint_metrics.csv`, `production_overlap.csv` — полный mathematical audit;
- `bootstrap.json`, `joint_covariance.json`, `mathematical_headroom.json`, `segment_diagnostics.csv`;
- `occurrence_components/` — 40 NPZ + 40 metadata JSON;
- `artifact_manifest.csv` и `SHA256SUMS.txt` — artifact integrity.

Leaderboard information нигде не использовалась.

## Final conclusion

1. **Teammate occurrence mechanism:** да, exact primary mechanism и original TEST artifact relation восстановлены. Старый downstream historical OOF overlay нельзя назвать fully reproducible из-за отсутствующего frozen Ridge/greedy bank; он не подменялся.
2. **Отличие от старых occurrence experiments:** восемь incidence-only cutoff-capped heads, их disagreement с own hurdle state, walk-forward all8 meta learner и error-risk trust gate образуют новый supervised representation, а не generic GMV/count/hazard replay.
3. **Реальный purged post-production signal:** нет. Weak signal почти полностью поглощается strong production baseline и full-span projection.
4. **Weighted/latest rho:** deployable joint `-0.005541 / -0.005481`; raw core `+0.000913 / +0.004210`; meta core `+0.000039 / +0.004908`.
5. **Conditional raw/meta contributions:** same-fold raw-given-meta `0.000002982` MSE, meta-given-raw `0.000000430`; coefficients не переносятся стабильно между folds.
6. **Expected robust Delta RMSLE:** `+0.00006833` — worsening; bootstrap `P(gain)=0.000`.
7. **Доля gap:** deployable signed `-4.8953%`, robust closeable `0%`; даже same-fold oracle только `0.0608%`.
8. **Decision:** **NO_GO**. Gates не менялись после результата; refinement запрещён.
9. **Candidate path/SHA:** отсутствуют, потому что GO/STRONG_GO не достигнуты.
