# E-CUP 2026, track 3 — состояние проекта после Phase 11

Дата аудита: 2026-08-24. Аудит выполнен по существующим скриптам, CSV/JSON/MD/TXT,
NPZ/NPY/PT, manifests, review bundles и submission-файлам. Тяжёлое обучение и
инференс на полном датасете не запускались. `src/validation.py`, `src/config.py`,
существующие checkpoints и `best_bas/submission_STRONGEST_CURRENT/` не изменялись.

## 1. Сводка состояния

- Задача: прогноз `GMV` пользователя на следующие 30 дней; рабочее пространство
  предсказаний и метрика — `z = log1p(GMV)`, RMSLE.
- Лучший полностью документированный исходный пакет:
  `best_bas/submission_STRONGEST_CURRENT/`, wCV **1.74751**, public LB
  **1.6496571**, SHA256 submission
  `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.
- Первый подтверждённый public-прорыв поверх него: Ridge/fixed-stack,
  **1.6492897556391737**. Значение записано в manifests Final6h; наиболее вероятный
  соответствующий файл — `best_bas/_best_bas_combo_10h/submissions/
  submission_combo10h_candidate_1_ridge_drop_recent_hurdle_stable18_s075.csv`,
  но явной строки «этот SHA получил этот LB» в репозитории нет.
- Лучший public LB согласно переданному внешнему журналу: **1.64921756224069**
  (малый LB-calibrated blend). Само значение и однозначный файл этого последнего
  blend в репозитории **не сохранены**. Лучший присутствующий поздний submission
  с известным из внешнего журнала LB — `extra90_3`, **1.6492260257794873**.
- Основной научный вывод Phase 12–14: существенная часть ошибки определяется
  occurrence (`GMV>0`), но выигрыш на нулях легко оплачивается ухудшением
  positive magnitude. Грубый routing больших сегментов не переносится.
- Основной вывод `best_bas`: безопаснее сохранять сильные SEQ/ETX magnitude и
  менять только табличный слот/occurrence малой поправкой. Даже локальные дельты
  `−0.0015…−0.0021` у сильно коррелированных кандидатов дали на public лишь
  десятки `1e-6…1e-4` после первого Ridge-прорыва.

## 2. Две разные схемы валидации — не смешивать числа

### Phase 12–14

- Primary folds: `2025-11-17`, `2025-12-15`, `2026-01-14`.
- `user_mod=3`, seed 42; seed и folds заданы в runner/manifests.
- База: Phase 11 `ridge_base` из
  `artifacts/phase11_model_selection_base_oof_mod3/`.
- Базовые RMSLE: **1.733944 / 1.737624 / 1.691539**, mean **1.721036**,
  recent-weighted **1.711725**.
- Для обучаемых temporal-моделей selection/calibration использует только
  доступные прошлые anchors; текущий fold не участвует в выборе. Для deployable
  routers используются только предыдущие primary folds. Oracle и all-fold
  specialist tables — диагностика, не честная OOF-подмена.
- В summaries одновременно есть `mean_rmsle`, recent-weighted, worst, January и
  wins против Ridge. Из-за выраженного январского режима одного среднего мало.

### `STRONGEST_CURRENT` и исследования `best_bas`

- Clean folds: `2025-09-04`, `2025-09-18`, `2025-10-02`, `2025-10-16`.
- Train cutoff удовлетворяет `T+30<=V`; train-панель 1-блочная, validation-панель
  3-блочная; per-fold calibration; wCV-веса **1:2:4:8**.
- `STRONGEST_CURRENT` wCV **1.74751**. Табличная proxy-база поздних continuation
  имеет wCV около **1.749774–1.749804**: это не противоречие, а другая
  реконструированная база/слот.
- CV Phase 12–14 (ноябрь–январь, mod3) и wCV `best_bas` (сентябрь–октябрь,
  1:2:4:8) количественно **несопоставимы**.

## 3. Хронология

| Этап | Фактический артефакт/время | Что сделано | Итог |
|---|---|---|---|
| Phase 12 | `phase12_localprob_tcn_long_v1_20260819_022327`, 8.360 ч | p-band occurrence/effective-q, decorrelated linear residual, classic TCN52, ModernTCN52, forced epoch/EMA/SWA | p-band modestly positive; standalone sequence-модели отрицательны/нестабильны |
| Phase 13 | `phase13_specialized_routing_long_v1_20260820_020252`, 9.452 ч | fine p-band, multiscale GBDT, long MoE, basic TCN, temporal segment/loss routing | routers и multiscale дали `−0.0013…−0.0023`; basic TCN выиграл только январь |
| Phase 14 | `phase14_error_specialists_ensemble_long_v1_20260820_134000`, 9.128 ч | банк 39 итоговых экспертов, first/second-level classes, error-detectors, pairwise competence, occurrence-only TCN/ModernTCN, safe composite | mean **1.716709**; причина выигрыша почти целиком January/occurrence |
| Исходный teammate bundle | submission отправлен 2026-08-20 | CAP/UNC/DIST + SEQ-AVG3 + ETX-AVG3 DCW | wCV **1.74751**, LB **1.6496571** |
| `run_best_bas_research_23h.py` | 2026-08-21 02:06–03:38 | начаты tabular/class1 эксперименты | остановка на повреждённом parquet из-за `os error 112` (место на диске); progress/manifests остались stale |
| `continue_best_bas_12h_v2.py` | 2026-08-21 12:31–16:28, 3.945 ч | cache repair, CAP/UNC/DIST/hurdle OOF, class1 specialists, LambdaRank | все кандидаты хуже proxy; LB `ranker_safe` **1.6541336855**, `class1_occ` **1.6880685734** |
| `run_best_bas_truehybrid_long_v2.py` | старт 2026-08-21 20:01 | true-hybrid/occ seeds | только `RUN_MANIFEST_START.json`; завершённых результатов нет |
| `run_best_bas_fixedstack_14h_v2.py` | старт 2026-08-22 13:30 | фиксированный SEQ/ETX, Ridge/meta/p-band/effective-q и новые table experts | сохранены validation tables, но нет финального manifest/report/submissions: запуск формально частичный |
| `continue_fixedstack_combo_10h.py` | 2026-08-23 04:18–05:46, 1.470 ч | завершены recent hurdle и Ridge subset/combo, собраны 4 submission | первый реально успешный public Ridge-stack: **1.6492897556** |
| `continue_best_bas_final6h.py` | 2026-08-23 16:18–20:48, 4.497 ч | 8 occurrence-only LightGBM, meta-occurrence, risk gate, adaptive blend | local `−0.00165…−0.00177`, 3/3 recent; meta-occ B public **1.6492612567** по внешнему журналу |
| `materialize_final6h_extra90m.py` | завершён 2026-08-23 22:25, 31.1 мин | cached diversification без retrain | `extra90_2` **1.6493195368**, `extra90_3` **1.6492260258** по внешнему журналу |
| Последний ручной blend | после Extra90 | малый LB-calibrated blend | **1.6492175622**, но файл/рецепт/score log отсутствует в репозитории |

Названия прежних runners `continue_best_bas_12h.py` и
`run_best_bas_fixedstack_14h.py` в текущем дереве отсутствуют; сохранены именно
`continue_best_bas_12h_v2.py` и `run_best_bas_fixedstack_14h_v2.py`.

## 4. Phase 12 — локальная вероятность и TCN52

| Модель | mean | Nov / Dec / Jan | Δ mean к Ridge | Вердикт |
|---|---:|---|---:|---|
| `pband_local_soft` | **1.719636** | 1.735080 / 1.737015 / 1.686814 | **−0.001399**, 2/3 | полезна, но основной gain в январе; Nov хуже `+0.001136` |
| diagnostic `0.75*pband+0.25*ridge` | **1.719601** | см. summary | −0.001435, 2/3 | лучший mean Phase 12, но post-hoc diagnostic, не отдельный temporal deployable selection |
| diagnostic `0.5*pband+0.5*ridge` | 1.719822 | — | −0.001213, 3/3 | стабильнее и меньше gain |
| `modern_tcn52_depthwise` | 1.728467 | 1.738249 / 1.738648 / 1.708504 | **+0.007431**, 0/3 | standalone reject; 25% blend с Ridge = 1.720404 и 3/3, источник слабой комплементарности |
| `classic_tcn52_residual` | 1.739451 | 1.741166 / 1.739637 / 1.737549 | **+0.018415**, 0/3 | reject, особенно January `+0.046011` |
| `linear_decorrelated_residual` | 1.728595 | 1.744985 / 1.739771 / 1.701031 | **+0.007560**, 0/3 | низкая корреляция без качества не помогла |
| `modern_tcn52_aux_occ` | incomplete | только Dec 1.738736 | — | недоисследовано: 1/3 prediction |

Локальные модели строили `p_eff` и эффективную вероятность
`q*=clip(z/mu,0,1)` внутри диапазонов `p`. Они подтвердили неоднородность ошибки
по p-band, но оптимальные family/blend менялись между folds. Forced epochs,
EMA/SWA не спасли classic TCN. Positive magnitude отдельно не улучшилась.

## 5. Phase 13 — specialized routing

| Модель | mean | recent-w | Jan | Δ mean / wins | Состояние |
|---|---:|---:|---:|---:|---|
| `router_p5risk_previous_only` | **1.718718** | 1.707927 | 1.684691 | **−0.002318**, 2/3 | лучший mean; Nov остаётся Ridge из-за отсутствия истории |
| `router_p10_previous_only` | 1.719069 | 1.708409 | 1.685353 | −0.001966, 1/3 | temporal, но Dec слегка хуже |
| `multiscale_basic_lgbm` | 1.719490 | 1.709276 | 1.687313 | −0.001545, 2/3 | простой и относительно устойчивый; Nov почти neutral |
| `router_loss_lgbm_latest_only` | 1.719711 | 1.709763 | 1.688459 | −0.001325, 2/3 | небольшой выигрыш, не выбирает oracle reserve |
| `fine_pband_stable` | 1.720395 | 1.710826 | 1.690207 | −0.000641, **3/3** | самый стабильный знак, малая величина |
| `basic_tcn_direct` | 1.720720 | **1.707354** | **1.676373** | −0.000315, 1/3 | сильнейший January `−0.015166`, но Dec `+0.012248`: режимно нестабилен |
| `long_moe_regime_direct` | 1.723825 | 1.710677 | 1.685549 | **+0.002789**, 1/3 | reject standalone; January полезен |
| `long_moe_midp_focus` | incomplete | — | отсутствует | 2 folds | не завершён |
| `basic_modern_tcn_direct` | incomplete | — | отсутствует | 2 folds, оба хуже | не завершён; ранние данные отрицательны |

Oracle по 15 экспертам: RMSLE **1.5779 / 1.5747 / 1.4607**, то есть reserve
`−0.156 / −0.163 / −0.231` к Ridge. Это не достижимая модель, но подтверждает
комплементарные user-level ошибки. В win-share заметны Phase 11
`moe_behavior_dcn_gate`, `basic_tcn_direct`, multiscale, decorrelated linear и
classic/Modern TCN; temporal routers извлекли только малую часть резерва.

## 6. Phase 14 — occurrence, error detectors и competence routing

### Итоговые модели

| Модель | mean | Nov / Dec / Jan | Δ mean / wins |
|---|---:|---|---:|
| `phase14_safe_composite` | **1.716709** | 1.733944 / 1.737727 / **1.678457** | **−0.004326**, 1/3 (Nov tie) |
| `modern_tcn_occurrence` | 1.716810 | 1.735141 / 1.736710 / **1.678578** | −0.004226, 2/3 |
| `router_hier_final_latest` | 1.717215 | 1.733944 / 1.737373 / 1.680326 | −0.003821, 2/3 |
| `router_firstlevel_final_latest` | 1.717897 | 1.733944 / 1.737816 / 1.681930 | −0.003139, 1/3 |
| `router_pairwise_competence_final` | 1.718468 | 1.733944 / 1.737075 / 1.684384 | −0.002568, 2/3 |
| `temporal_occurrence_calibration` | 1.718733 | 1.738031 / 1.737009 / 1.681158 | −0.002303, 2/3 |
| `error_detector_corrector` | 1.721330 | 1.735406 / 1.738243 / 1.690340 | **+0.000294**, 1/3 |
| `ridge_base` | 1.721036 | 1.733944 / 1.737624 / 1.691539 | baseline |

`phase14_safe_composite` — walk-forward pair search: на первом fold Ridge, затем
pair/blend выбирается только по предыдущим folds. Его mean-лидерство почти
полностью создаёт January; Dec на `+0.000103` хуже Ridge. Поэтому это сильный
локальный результат, но не универсально стабильная архитектура.

### Что именно установлено

1. **Occurrence — главный bottleneck.** У `modern_tcn_occurrence` zero RMSLE
   относительно Ridge изменился на `−0.0498 / −0.0056 / −0.1572`, а positive
   RMSLE одновременно на `+0.0356 / +0.0027 / +0.1125`. Общий выигрыш — баланс
   этих противоположных эффектов, а не улучшение magnitude.
2. Error-detectors обучались для `false_zero`, `false_one`, `severe_under`,
   `severe_over`, `catastrophic`. AUC по folds:
   - false-zero **0.8785 / 0.8833 / 0.8931**;
   - false-one **0.8895 / 0.8860 / 0.8789**;
   - severe-over **0.8455 / 0.8404 / 0.8304**;
   - severe-under **0.6481 / 0.6435 / 0.6614**;
   - catastrophic **0.7047 / 0.7015 / 0.7019**.
3. Прямая magnitude correction по detector score закрыта: `error_detector_corrector`
   хуже mean и выигрывает только January на `−0.00120`.
4. Detectors полезнее как признаки model competence. First-level,
   hierarchical second-level и pairwise routers положительны, но их gain мал
   относительно oracle.
5. `classifier=1` занимает в среднем **58.19%** строк и около **64.7%** squared
   error. Сегмент raw-score `false_one=1 & over_risk=1` занимает **17.67%** строк
   и около **26.7%** ошибки; даже лучшие модели в нём имеют RMSLE около 2.10.
6. Oracle по 39 моделям: **1.571642 / 1.565430 / 1.432355**, reserve к Ridge
   **−0.1623 / −0.1722 / −0.2592**. Pairwise final извлёк только
   `−0.00257` mean.
7. Temporal occurrence calibration сама нестабильна: Nov хуже `+0.00409`, Dec
   лучше `−0.00062`, Jan лучше `−0.01038`; forecast logit shifts
   `−0.1185 / −0.0487 / −0.1700`.
8. `tcn_hurdle_twohead` и occurrence-errorfocus снова сильны только January и
   ухудшают ранние folds. Стандартный `tcn_occurrence` не завершён (2/3).

## 7. Исходный `STRONGEST_CURRENT`

Финальная смесь в `log1p`:

```text
0.10 * S1-CAP
+ 0.20 * S1-UNC
+ 0.25 * S1-DIST
+ 0.225 * ETX-AVG3 @ DCW
+ 0.225 * SEQ-AVG3 @ depth-clip 289
```

Это эквивалентно табличной части весом 0.55 и sequence-слоту весом 0.45, внутри
которого `0.5*ETX-AVG3 + 0.5*SEQ-AVG3`.

- SEQ: causal dilated TCN, 17 каналов, hidden 64, 8 blocks, kernel 3,
  dropout 0.1, 4 epochs. Seeds 42/43/44. На TEST все предсказания используют
  `depth-clip 289`; checkpoint seed 42 исторически не сохранён, но его TEST
  prediction сохранён.
- `SEQ-D3A`: depth augmentation реальных каналов, локально полезна, но в slot
  уступила/сыграла вничью с SEQ-AVG3; TEST checkpoints/predictions для D3A в
  исходном пакете отсутствуют.
- ETX: sparse event transformer, `d_model=128`, 5 blocks, 8 heads,
  `head_dim=16`, FFN 384, 192 event tokens, 4 epochs; seeds 42/43/44.
- DCW исправляет train→test режим ETX: `depth_clip=289`, static depth cap 289 и
  сдвиг cutoff DOW на четверг (`dow_shift=-1`). Без этого pair divergence на
  TEST была 3.22× OOF; после исправления 0.78×.
- ETX не превосходит SEQ по occurrence AUC, но дополняет его: честный LOFO
  sequence-слота **−0.00092, 4/4**. `Var(z_ETX-z_SEQ)` значительно больше
  межсидовой дисперсии TCN; ценна ошибка относительно партнёра, а не solo rank.
- Файл: 250 000 строк, `mean(log1p(pred))=2.329321`, без NaN/inf/negative,
  reconstruction max log error `4.97e-7`, public LB **1.6496571**.

Пакет самодостаточен для **точной пересборки submission из сохранённых TEST
predictions**, но не для полного retrain/OOF-воспроизведения: отсутствуют raw
data, booster weights CAP/UNC/DIST, SEQ-S42 TEST checkpoint и все исходные OOF
arrays. PyTorch weights: 3 ETX TEST + 2 SEQ TEST.

## 8. Эксперименты поверх `STRONGEST_CURRENT`

### Закрытая class1/routing ветка

- Proxy wCV **1.749774**.
- Все class1/ranker-кандидаты проиграли на каждом измеримом fold:
  `ranker_safe` 1.756028, `ranker_risk` 1.770563, `class1_occ` 1.787717,
  `class1_direct` 1.794574, `class1_recent` 1.795105,
  `class1_over_guard` 2.045687.
- Error-detector AUC здесь ниже Phase 14: false-zero около 0.824–0.827,
  false-one 0.764–0.766, over 0.710, under 0.735.
- Public LB подтвердил reject: `ranker_safe` **1.654133685532829**,
  `class1_occ` **1.688068573391526**.
- Вывод закрыт: нельзя грубо заменять большой `classifier=1` сегмент и нельзя
  переносить силу Phase14-detectors на другую base-модель без повторной проверки.

### Fixed stack / Ridge

- SEQ/ETX teammate не переобучались; менялся табличный слот.
- Проверены Ridge по predictions, Ridge+raw/meta, residual Ridge/LGBM, temporal
  weighting, greedy, simplex, p-band, effective-q, local bias, recent
  direct/dist/hurdle, multiscale и slot strength.
- Сильные локальные семейства:
  - `ridge_meta_a700_s075`: wCV 1.748316, Δ `−0.001488`, 3/3 recent;
  - `ridge_drop_recent_hurdle_stable18_s075`: wCV 1.748257, Δ
    `−0.001547`, latest `−0.001662`, 3/3 recent;
  - `ridge_core_plus_recent_dist_s075`: Δ `−0.001461`, 3/3 recent.
- Отрицательные: direct LGBM residual `+0.008…+0.027`, multiscale direct
  `+0.01024`, recent direct `+0.00113`, strong effective-q `+0.00306`.
- Первый Ridge-stack дал public **1.6492897556**, улучшение к friend
  `−0.00036734`: локальный gain перенёсся примерно с коэффициентом 0.24.

### Final6h/Extra90 occurrence

- Полный fold-grid и TEST predictions есть для `occ_r10_fast`, `occ_r16_bal`,
  `occ_r22_stable`, `occ_r14_multiscale`, `occ_r18_wide`,
  `occ_r24_multiscale`, `occ_r12_wide`, `occ_r20_shallow`.
- Occurrence применяется как overlay/meta-feature, а magnitude/residual сильного
  fixed stack сохраняется.
- Branch A adaptive blend: local Δ `−0.001651`, latest `−0.001811`.
- Branch B risk meta-occ: local Δ `−0.001767`, latest `−0.002027`;
  public **1.6492612567** по внешнему журналу, всего `−0.0000285` к Ridge public.
- Extra90 meta (`xmeta_div4...`): local Δ `−0.001821`, public
  **1.6493195368** — хуже Ridge, несмотря на лучший local.
- Extra90 raw `occ_r10_fast` overlay: local Δ `−0.001625`, public
  **1.6492260258**, то есть лишь `−0.0000637` к Ridge.
- Финальный LB-calibrated blend: **1.6492175622**, ещё `−0.00000846` к raw
  overlay. Рецепт/файл в репозитории не найден.

Все поздние кандидаты имеют corr с friend **0.99962–0.99972**. Это объясняет
малый public reserve: оптимизируются почти те же ошибки, а local selector
чувствителен к temporal regime и калибровке.

## 9. Артефакты и покрытие предсказаний

### Phase 12–14

- Phase 12: 13 NPZ predictions. Complete 3-fold: p-band, linear, classic TCN,
  ModernTCN depthwise. Incomplete: `modern_tcn52_aux_occ` 1/3. Ridge берётся из
  Phase 11 и отдельно в Phase12 predictions не дублируется.
- Phase 13: 37 NPZ. Complete: Ridge, fine p-band, multiscale, long MoE direct,
  basic TCN direct и 6 routers. Incomplete: `long_moe_midp_focus` 2/3,
  `basic_modern_tcn_direct` 2/3.
- Phase 14: 59 NPZ. Complete для 19 семейств, включая все итоговые routers,
  safe composite, modern occurrence, error corrector и hurdle/errorfocus.
  Incomplete: `tcn_occurrence` 2/3.
- Все эти NPZ — user_id/target/base/prediction и иногда probability/head arrays;
  PyTorch weights Phase 12–14 не сохранены.
- Review bundles существуют отдельно и содержат reports/manifests, но обычно не
  prediction NPZ; полные prediction NPZ лежат в artifact directories.

### Shared cache `best_bas/_best_bas_research`

- 80 fold NPZ = **20 семейств × 4 clean folds**.
- 15 TEST NPZ: hurdle/meta/multiscale, 8 occurrence, recent direct/dist и два
  recent hurdle.
- NPZ здесь — **prediction caches, не веса LightGBM/моделей**. Их можно сразу
  использовать для stack/diagnostics без retrain, но нельзя из них восстановить
  model object.
- Feature cache: 121 parquet, основная часть объёма ветки (~9.7 GB); производные,
  не исходные данные. Один повреждённый parquet был удалён и перестроен
  continuation; текущие перечисленные файлы читаемы в завершённых runs.
- `results/progress.json` и начальный `RUN_MANIFEST.json` stale и отражают только
  прерванный первый запуск; состояние следует определять по последующим manifests
  и фактическим 80+15 NPZ.

Подробная карта: `best_bas/ARTIFACT_INDEX_AFTER_PHASE11.md`. Машинный индекс:
`best_bas/artifact_inventory_after_phase11.json`.

## 10. PyTorch checkpoint-grid teammate

Все **52** новых файлов имеют словарь `state/cfg/val`; все строго (`strict=True`)
загрузились исходными `pipeline/src/seq.py::build_model` или
`pipeline/src/etx.py::build_model`. Ни один не требует изменения архитектуры.

| Family | Validation coverage | TEST | Пробелы/статус |
|---|---|---|---|
| ETX-01 | S42: 4/4; S43: 0/4; S44: 3/4 | S42/S43/S44 | нет S43 всех folds и S44/V1016; все 3 TEST — byte-identical duplicate original |
| SEQ-01 | S42: только `SEQ-01C/V0904`; S43: 3/4; S44: 4/4 | S43/S44 | нет S42 V0918/V1002/V1016 и TEST, S43/V1016; 2 TEST — byte-identical duplicate original |
| SEQ-D3A depth_aug=0.5 | S42 4/4; S43/G1 4/4; S44/G2 4/4 | нет | полный 3-seed validation bank, пригоден для OOF/ошибок; нет ни одного TEST |
| SEQ-D3A controls depth_aug=0 | S42 4/4; S43/G1 4/4; S44/G2 4/4 | нет | полный control validation bank; есть дополнительный plain S43/V0918 |
| legacy `SEQ-03A` | только S42/V1016: `avail_drop A25`, `avail_bnd B`, base | нет | диагностические единичные варианты; `03A-BASE` — exact duplicate `D3A-BASE-S42-V1016` |
| DETSEQ01 | S42/S43/S44 только V1016 | нет | полезный свежий 3-seed diagnostic; отсутствуют 9 ранних validation и все 3 TEST |

Полезность новых весов:

- наиболее ценный готовый ресурс — два полных D3A validation banks: можно без
  retrain реконструировать 3-seed OOF, сравнить error covariance и проверить
  diversity против SEQ/ETX/table;
- ETX S42 full и SEQ S43/S44 почти полны — полезны для частичного OOF/seed
  stability, но не дают честный полный 3-seed ETX/SEQ ensemble;
- DETSEQ01 даёт независимый 3-seed взгляд на самый свежий fold, но по одному
  fold нельзя выбирать production policy;
- новые weights сами по себе не создают новый submission: D3A/DETSEQ TEST
  checkpoints отсутствуют.

## 11. Закрытые, нестабильные и открытые вопросы

### Закрыто/не повторять без нового сигнала

- standalone decorrelated model только ради низкой corr;
- classic/Modern TCN direct/residual как полная замена Ridge;
- грубая class1 подмена, LambdaRank expert routing и risk replacement;
- direct magnitude correction по error-detector score;
- сильный effective-q и direct LGBM residual в fixed stack;
- выбор по одному latest fold;
- полная TEST depth 365 для SEQ/ETX; production policy — clip 289, ETX DCW;
- post-hoc настройка очень коррелированных весов как путь к `<1.64`.

### Нестабильно

- basic/occurrence TCN: огромный January gain и ухудшение ранних folds;
- p-band family и routing choices меняются по режимам;
- temporal prevalence/logit calibration;
- local meta-occ rankings: лучший local кандидат может ухудшать public;
- public mapping последних submission не сохранён рядом с артефактами.

### Реально недоисследовано

- полный 3-seed OOF-анализ D3A из новых checkpoints;
- error/diversity-анализ DETSEQ01 на V1016 без экстраполяции на TEST;
- недостающие ETX/SEQ validation cells и SEQ-S42 TEST weight;
- отдельный структурный источник signal, слабо коррелированный именно по
  ошибкам, а не по прогнозам;
- nested temporal selection, явно штрафующий regime sensitivity и размер
  изменения относительно LB-подтверждённой базы;
- сохранение централизованного `submission SHA -> public LB` registry.

## 12. Источники истины и известные несоответствия

1. Public LB до Ridge включительно записаны в repo manifests/docs. Четыре поздних
   public LB и финальный best взяты из переданного пользователем журнала; в repo
   соответствующего score registry нет.
2. Финальный `1.64921756224069` не привязан к сохранённому submission/рецепту.
3. Fixedstack directory не имеет финального manifest и submissions, хотя
   `candidate_validation_final.csv` заполнен; завершение произошло continuation
   `combo_10h`.
4. `truehybrid_long_v2` имеет только start manifest — результатом считать нельзя.
5. Initial `_best_bas_research/progress.json` устарел и занижает текущее
   checkpoint coverage.
6. Новая папка называется `ETX-01_weights`, `TCN_SEQ-01_weights`,
   `TCN_SEQ-D3A_weights`, `TCN_DETSEQ01_weights`, а не короткими именами из
   предварительного описания.
7. D3A — не один 29-файловый однородный grid: это два полных 12-cell banks
   (`depth_aug=0.5` и control `0`), дополнительные/legacy runs и один exact alias
   duplicate.

## 13. Контрольные пути

- Phase 12: `artifacts/phase12_localprob_tcn_long_v1_20260819_022327/`
- Phase 13: `artifacts/phase13_specialized_routing_long_v1_20260820_020252/`
- Phase 14: `artifacts/phase14_error_specialists_ensemble_long_v1_20260820_134000/`
- Review bundles: соседние `*_REVIEW_BUNDLE.zip` в `src/DL/`
- Original: `best_bas/submission_STRONGEST_CURRENT/`
- Shared cache: `best_bas/_best_bas_research/`
- Class1 continuation: `best_bas/_best_bas_continue_12h/`
- Ridge combo: `best_bas/_best_bas_combo_10h/`
- Final occurrence: `best_bas/_best_bas_final6h/`
- Extra90: `best_bas/_best_bas_extra90m/`
- Teammate weights: `best_bas/teammate_extra_weights/`
