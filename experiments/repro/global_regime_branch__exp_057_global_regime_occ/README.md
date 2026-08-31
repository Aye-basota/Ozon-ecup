# EXP-057 — GLOBAL-REGIME-OCC-RANK

## Catalogue metadata

- **Catalogue ID:** `global_regime_branch__exp_057_global_regime_occ`
- **Namespace:** `global_regime_branch`
- **Experiment ID:** `exp_057_global_regime_occ`
- **Original source:** `git:4003b6874f39:experiments/exp_057_global_regime_occ.md`
- **Source ref:** `4003b6874f397fe48577b26118ae1d560a703419`
- **Source commit:** `4003b6874f397fe48577b26118ae1d560a703419`
- **Kind:** git-history experiment card
- **Model:** LightGBM, sequence model, Ridge, two-part / hurdle, ensemble, blend, calibration diagnostic
- **Features:** recency, occurrence features, gap/burst features, dataset/user fingerprint, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | `friend` OOF, 4 фолда | `0.10·S1-E03a + 0.20·S1-E02 + 0.25·S1-DIST + 0.225·ETX-AVG3 + 0.225·SEQ-AVG3` | wCV_cal **1.7475098627** (`STATE.md`: 1.74751) |
- **Known score:** | `friend` OOF, 4 фолда | `0.10·S1-E03a + 0.20·S1-E02 + 0.25·S1-DIST + 0.225·ETX-AVG3 + 0.225·SEQ-AVG3` | wCV_cal **1.7475098627** (`STATE.md`: 1.74751) |
- **Seed:** seeds, overlay, downstream, калибровка.
- **Postprocessing:** на 10-16). Placebo эту корреляцию держит на уровне BASE. То есть реальные global-признаки
- **Submission:** детерминированный повтор (SHA256 блока совпадает при повторном построении), схему сабмита,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP-057 — GLOBAL-REGIME-OCC-RANK

**ID collision:** номер `057` уже занят в `STATE.md` за `STATE-REWEIGHT`
(`experiments/exp_057_production_state_reweight.md`, REJECT 2026-08-24). Пути этого
эксперимента (`exp_057_global_regime_occ.md`, `GLOBAL_REGIME_OCC_EXP057`) были свободны,
поэтому работа не остановлена, но при внесении в `STATE.md` строку надо переименовать
(предлагается `EXP-061 / GLOBAL-REGIME-OCC`), иначе две разные ветки будут делить один ID.

**Вопрос.** Добавляет ли cutoff-safe global monetization regime и движение пользователя
относительно population новый occurrence-сигнал поверх уже LB-полезного `occ_r10_fast`?

---

## BASELINE RECONSTRUCTION

### Что воспроизведено точно

| Объект | Проверка | Результат |
|---|---|---|
| `friend` = `STRONGEST_CURRENT` | SHA256 `latest/components/friend.csv` против `submissions/submission_STRONGEST_CURRENT.csv` | `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`, идентичны |
| `X3` = extra90 candidate 3 | SHA256 `occ_raw_X3.csv` против `submission_extra90_3_xraw_occ_r10_fast_adapt__…` | `0ac3f241685d6562c1f5b54993065475845464a0b217a56dbb5d79b42dd27356`, идентичны |
| `B` = final6h Branch B | SHA256 `occ_meta_B.csv` против `submission_final6h_B_metaocc_l31_risk__…` | `8d90a0bb1afdfe48cb6181cf1869eb30095da729e36dbea81e2051730e7989d8`, идентичны |
| `latest = .12·friend + .16·B + .72·X3` | max\|Δz\| по 250 000 строк | **8.88e-16** — точная реконструкция |
| `friend` OOF, 4 фолда | `0.10·S1-E03a + 0.20·S1-E02 + 0.25·S1-DIST + 0.225·ETX-AVG3 + 0.225·SEQ-AVG3` | wCV_cal **1.7475098627** (`STATE.md`: 1.74751) |
| pipeline-код | diff `friend_original/…/pipeline/src/*` против `HEAD:src/*` | `config/data/models/train/validation/predict/features` — **идентичны** |

Расшифрован и весь downstream-механизм (`run_best_bas_fixedstack_14h_v2.py`,
`continue_best_bas_final6h.py`):

```
TABLE_WEIGHT = 0.55
table_core   = (0.10·CAP + 0.20·UNC + 0.25·DIST) / 0.55
final        = level(friend + 0.55·(candidate_table − table_core), LEVEL = 2.3293)

p_apply(base, p_base, mu, p_new, down, up, shift, th):
    pp       = σ(logit(p_new) + shift)
    delta    = pp − p_base
    strength = down если delta < 0 иначе up          ← асимметрия
    strength = 0 где |delta| < th
    return clip(base + strength · delta · mu, 0, 20)

параметры (shift, down, up, th) подбираются walk-forward
только по фолдам строго ДО оцениваемого; фолд 1 — fixed (−.08, .75, .12, .025)
```

Асимметрия подтверждена на TEST по `X3 − A`: mean ровно 0, 66.7 % положительных /
33.3 % отрицательных, min −0.1586 против max +0.0392 — отрицательный хвост вчетверо длиннее.

`occ_r10_fast` восстановлен по точной спецификации, без приближений:
`OccCfg("occ_r10_fast", maxcuts=10, tau=55.0, rounds=380, leaves=31, min_leaf=520,
feature_mode="all", feature_fraction=.82)` поверх spec'а `recent_hurdle`
(`Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1, model="two_part",
rounds=520, norm_long=True)`), LightGBM `binary/binary_logloss`, `lr=.035`,
`bagging .90/1`, `λ2=14`, `λ1=1`, `max_bin=127`, `seed=42`. Обучающие cutoff'ы — последние
10 eligible clean с обязательным разрывом `min_gap=30`; веса `exp(−(V−T)/55)`.

### БЛОКЕР: точный endpoint ТЗ недостижим

`X3` на фолдах строится как
`walk_occ_candidate(bank, p_occ_r10_fast, base_oof = predpool["blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85"])`.
Это требует OOF-банк `_best_bas_research/checkpoints`: `{cap,unc,dist,hurdle}__<fold>.npz`
(с матрицей `meta_raw` внутри `cap`), `{multiscale_direct,recent_direct,recent_dist,
recent_hurdle}` + `MEM_VARIANTS`, 8 occurrence-семейств и ~9.7 GB feature-кэша.

Проверен весь диск (включая папку `пайплайн сокомандника` — она побайтово совпадает с
`latest_pipeline_bundle`, 147 файлов, те же относительные пути): **ни одного per-user
OOF-предсказания стека Ridge/greedy/occurrence нет**, только TEST-сабмиты и сводные CSV.
`README_PIPELINE_RU.md` §7 это признаёт прямо. Пересчёт с нуля — цепочка
`23h → 14h → 10h → 6h` runner'ов (~53 ч) со своим layout'ом `src/DL/best_bas/`, причём
внутри них есть time-guard'ы и отбор кандидатов, так что *exact* reconstruction не
гарантирован даже полным прогоном.

Следствия:
* `X3_BASE / X3_GLOBAL / X3_PLACEBO` на фолдах построить нельзя;
* `latest_GLOBAL` при exact parity `.12/.16/.72` собрать нельзя (OOF Branch B тоже нет);
* пункт §7 (production + submission) не может быть выполнен в заявленной форме.

### Re-anchored endpoint

Вместо anchor'а `A` использован `table_core` — это **штатная конфигурация того же кода**
(`walk_occ_candidate(..., base_oof=None)` по умолчанию берёт `r["table_core"]`), а не
самодельная замена. Она восстановима точно из OOF репозитория. Endpoint:

```
final_arm = friend_OOF + 0.55 · (p_apply(table_core, p, mu, p_arm, …) − table_core)
wCV       = взвешенная 1:2:4:8 калиброванная RMSLE по 4 фолдам
```

`p` и `mu` — из `recent_hurdle`, переобученного по точному spec'у товарища на всех
`train_cutoffs(V)` (18/20/22/24 cutoff'а), одинаково для всех arms.

Что это сохраняет: реальный производственный механизм (55 % табличный слот,
асимметричный overlay, walk-forward подбор параметров, фиксированный downstream).
Что теряет: дельты измеряются относительно `table_core`, а не Ridge/greedy-анкера, и
собрать `latest_GLOBAL` нельзя. Поэтому числовые гейты §6 к `X3/latest` **напрямую
не применимы**; они применяются к re-anchored endpoint'у и это отмечено в вердикте.

---

## FEATURE MANIFEST

Зафиксирован в `artifacts/GLOBAL_REGIME_OCC_EXP057/feature_manifest.json` **до** чтения
любых результатов. Всего **140** новых колонок поверх 227 существующих.

### §3.1 Global platform state — 115

Универсум: все пользователи с хотя бы одной сырой строкой в окне.
Окна `7/14/30/60/90`, по 17 метрик = 85:
`users_row, users_active, users_buy, users_search, users_cart, searches, cat, carts,
orders, gmv, buyer_rate, order_rate, conv_search_cart, conv_cart_order, gmv_per_buyer,
gmv_per_order, orders_per_buyer`.

Динамика — только заранее фиксированная, `log1p`-разности соседних блоков
`(7,7) / (14,14) / (30,30)` по 10 метрикам = 30. Ratio-форма сознательно не добавлена:
она почти монотонная функция той же разности и удвоила бы блок без информации.

Нет: cutoff-индекса, даты, идентификатора фолда, любых событий после cutoff'а.

### §3.2 User-relative state — 18

Перцентиль внутри **той же cross-section, которую модель скорит** на этом cutoff'е
(train-строки — `panel_users(T, 1)`, валидация — `panel_users(V, 3)`), по 6 метрикам
`gmv, orders, buy_days, carts, searches, active_days` для текущего и предыдущего
30-дневного блока плюс delta. Ранг — average-rank со стабильным argsort: огромная
нулевая масса получает одно общее значение, повторный прогон побитово совпадает.

`u_pct_cur_*` — control/context, `u_pct_delta_*` — основная гипотеза.

### §3.3 User/global interactions — 7

```
x_user_gmv_over_platform_per_active   log1p(u_gmv30) − log1p(GMV30 / active users)
x_user_order_rate_over_platform       log1p(u_orders30/active days) − log1p(orders30/active users)
x_user_conv_over_platform             log1p(u_carts30/u_searches30) − log1p(search→cart)
x_pctdelta_gmv_X_rec_buy              Δpercentile(GMV) × log1p(rec_buy)
x_pctdelta_gmv_X_buy_days             Δpercentile(GMV) × buy days 30d
x_global_decline_X_rec_any            dlog30(GMV) × log1p(rec_any)
x_global_decline_X_user_trend         dlog30(GMV) × (log1p(gmv_cur30) − log1p(gmv_prev30))
```

---

## PLACEBO CONTRACT

Architecture-matched negative control, тот же feature count, те же маргиналы,
та же missingness.

* **Global:** циклический сдвиг на один eligible cutoff назад. Каждый реальный вектор
  используется ровно один раз, поэтому масштаб и support идентичны, но каждому cutoff'у
  сообщается чужой режим.
* **Percentile:** перестановка **целых строк** блока внутри страт
  `cutoff × дециль recent GMV × recency-bucket`. Строки двигаются вместе, поэтому
  внутристратные маргиналы и совместное распределение сохраняются точно, рвётся только
  связь с личностью пользователя и, значит, с меткой.
* **Interactions** пересчитываются из placebo-входов, а не переставляются отдельно.

Не меняется ничего: train rows, labels, cutoff weights, LightGBM-параметры, rounds,
seeds, overlay, downstream, калибровка.

---

## PRODUCTION SUPPORT

Посчитано **до** обучения arms — и это оказалось решающим.

### Универсум датасета

```
строк 30 631 006, различных user_id — ровно 250 000
множество user_id датасета == множество user_id sample_submit  (True, 0 расхождений)
panel_users(2025-07-10, 3) = 181 704
panel_users(2025-09-04, 3) = 188 518
panel_users(2025-10-16, 3) = 197 379
panel_users(2026-02-13, 3) = 250 000
```

«Платформы» за пределами 250 000 отобранных пользователей не существует. Эти 250 000
отобраны правилом организатора **на тестовом cutoff'е** (активен в каждом из трёх
последних 30-дневных блоков до 2026-02-13). Поэтому «число активных пользователей»
не является измерением платформы: это доля фиксированной, отобранной вперёд когорты,
которая обязана расти к 2026-02-13. Половина заявленного дрейфа (`active ×1.032`)
— артефакт отбора, тот же класс, что уже записан в `STATE.md` для `T ≥ 2025-10-17`
(`e08`, `exp_028`: P(активность 30д) = 1.0 на 11-15/12-15/01-14 против 0.89–0.93 на чистых).

### Дрейф лежит вне чистого коридора

```
блок                 GMV ratio   active-user-days
2025-09-16 → 10-16     1.0252         1.0677
2025-10-16 → 11-15     1.0065         1.0794
2025-12-15 → 01-14     0.9936         0.9309
2026-01-14 → 02-13     0.8285         1.0323   ← шок
```

Все 4 чистых фолда живут в режиме роста. `g_d30_dlog_gmv` по 29 чистым cutoff'ам:
диапазон **[−0.0092, +0.1077]**. На тесте — **−0.1881**.

### Support-аудит 115 global-признаков на тесте

```
вне диапазона всех 29 чистых cutoff'ов          81 / 115  (70.4 %)
вне диапазона последних 10 (окно обучения occ)  89 / 115  (77.4 %)
```

Худшие — ровно те, на которых держится гипотеза:

| признак | train | test | z |
|---|---|---|---|
| `g_d30_dlog_gmv_per_buyer` | [−0.0179, 0.0393] | −0.1466 | −8.86 |
| `g_w30_gmv_per_order` | [33.76, 36.07] | 30.01 | −6.84 |
| `g_d30_dlog_gmv` | [−0.0092, 0.1077] | −0.1881 | −6.68 |
| `g_w30_users_row` | [197 912, 225 431] | 250 000 | +5.14 |
| `g_w90_searches` | [17.7M, 22.3M] | 27.4M | +6.48 |

LightGBM экстраполирует биннингом, то есть на тесте все decline-признаки прижались бы
к обучающему минимуму, а `x_global_decline_X_*` умножались бы на **зажатую константу**.
Механизм, ради которого ставилась гипотеза, на тесте физически не срабатывает.

### Global-признаки как cutoff-fingerprint

Spearman(порядок cutoff'а, значение) по 29 чистым cutoff'ам:

```
|ρ| ≥ 0.99   31 / 115  (27.0 %)   ← ровно ±1.0000
|ρ| ≥ 0.95   60 / 115  (52.2 %)
|ρ| ≥ 0.90   68 / 115  (59.1 %)
медиана |ρ| = 0.958
```

`g_w30_users_row`, `g_w{30,60,90}_users_{active,buy,search,cart}` и др. имеют ρ **ровно
+1.0000**: это строго монотонная биекция с порядком cutoff'а, то есть функционально тот
самый «raw cutoff index», который §3.1 запрещает. Запрет был сформулирован по имени,
а нарушается по существу.

Честная часть §3.1 — динамика: `g_d7_dlog_gmv` ρ=+0.030, `g_d30_dlog_gmv_per_buyer`
ρ=+0.041, `g_d30_dlog_conv_cart_order` ρ=−0.071. Но именно она сильнее всего вне support'а
на тесте. То есть §3.1 распадается на две половины, и **ни одна не может дать переносимый
выигрыш**: level-блок — fingerprint, dynamics-блок — вне диапазона.

`u_pct_*` (§3.2) от этого свободны: это ранги внутри cross-section, ограниченные
`(0,1]` и `[−1,1]`, безразмерные по построению.

---

## FOLD RESULTS

Все четыре чистых фолда, `min_gap=30`, веса wCV 1:2:4:8. Downstream не оптимизировался.

### Standalone occurrence

```
             AUC(1[y30>0]) по фолдам                     wAUC       logloss(w)
BASE     0.841404  0.843749  0.846380  0.847520        0.846305     0.468890
GLOBAL   0.841377  0.843740  0.846312  0.847467        0.846256     0.469280
PLACEBO  0.841336  0.843756  0.846451  0.847538        0.846330     0.469200

GLOBAL − BASE      -0.000027 -0.000010 -0.000068 -0.000054   w=-0.000050   wins 0/4
GLOBAL − PLACEBO   +0.000041 -0.000017 -0.000140 -0.000071   w=-0.000075   wins 1/4
```

### Full-ensemble (re-anchored endpoint)

```
friend (STRONGEST_CURRENT)  wCV = 1.7475098625

final_BASE     1.7474288101    к friend -0.000081
final_GLOBAL   1.7474862831    к friend -0.000024
final_PLACEBO  1.7474215524    к friend -0.000088
```

```
контраст            ΔwCV       wins   10-16       пофолдово
GLOBAL − BASE     +0.000057     0/4   +0.000060   +0.000038 +0.000001 +0.000086 +0.000060
GLOBAL − PLACEBO  +0.000065     0/4   +0.000051   +0.000081 +0.000047 +0.000098 +0.000051
PLACEBO − BASE    -0.000007     3/4   +0.000009   -0.000042 -0.000045 -0.000012 +0.000009
```

Overlay-параметры walk-forward получились **одинаковыми во всех трёх arms**
(`(-0.08,.75,.12,.025)` на фолде 1, затем `(-0.22,.65,.05,None)`, `(-0.22,.45,.05,None)` ×2),
поэтому сравнение arms не спутано с подбором overlay.

---

## FULL-ENSEMBLE EFFECT

| гейт §6 | требуется | факт | итог |
|---|---|---|---|
| ΔwCV STRONG PASS | ≤ −0.0007 | **+0.000057** | провал |
| лучше на ≥3/4 фолдах | ≥3/4 | **0/4** | провал |
| обязательно лучше 2025-10-16 | да | **+0.000060** (хуже) | провал |
| GLOBAL − PLACEBO | ≤ −0.0004 | **+0.000065** (хуже placebo) | провал |
| REJECT: ΔwCV > −0.0003 | — | выполнено | **REJECT** |
| REJECT: GLOBAL ≈ PLACEBO | — | выполнено, GLOBAL **хуже** placebo 4/4 | **REJECT** |

Знак устойчив: GLOBAL хуже BASE на всех четырёх фолдах и хуже PLACEBO на всех четырёх.

Про величину: `BASE` overlay на re-anchored endpoint'е стоит всего −0.000081, тогда как у
товарища `X3` относительно `table_core` даёт −0.00162. Разница ожидаема — их база
Ridge/greedy-анкер, а не `table_core`, абсолютные величины несопоставимы. Но **контраст
между arms** от выбора базы по знаку не зависит: на standalone-оси, где базы вообще нет,
вывод тот же.

---

## MECHANISM DIAGNOSTICS

### Признаки выучены — «модель их проигнорировала» не объяснение

```
fold       pair              corr      sd(Δlogit)   mean|Δp|   max|Δp|
20250904   GLOBAL-BASE     0.999374     0.05815     0.00756    0.07731
20250904   PLACEBO-BASE    0.999361     0.05767     0.00796    0.09981
20250918   GLOBAL-BASE     0.999397     0.05769     0.00708    0.08099
20250918   PLACEBO-BASE    0.999461     0.05588     0.00659    0.08268
20251002   GLOBAL-BASE     0.999047     0.07197     0.01146    0.11111
20251002   PLACEBO-BASE    0.999013     0.07127     0.00970    0.09320
20251016   GLOBAL-BASE     0.999300     0.06317     0.00811    0.09385
20251016   PLACEBO-BASE    0.999152     0.06732     0.01028    0.09471
```

Реальные признаки двигают модель ровно настолько же, насколько matched placebo
(sd(Δlogit) 0.058/0.058, 0.072/0.071, 0.063/0.067). Модель их использует — и получает из
них то же, что из шума.

### Выигрыш не является global offset

```
GLOBAL − BASE   w(raw) = +0.000072    w(cal) = +0.000057
```

Знак и порядок одинаковы до и после калибровки — это не артефакт уровня.

### Zero/positive decomposition согласуется, но не в пользу GLOBAL

```
              09-04      09-18      10-02      10-16      wavg
rmsle_pos    -0.000617  -0.000370  -0.001000  -0.000403  -0.000572
rmsle_zero   +0.000941  +0.000526  +0.001636  +0.000726  +0.000956
```

GLOBAL лучше на покупателях и хуже на нулях. Механизм виден в `correction_mean`: GLOBAL
корректирует вниз слабее (−0.0150 против −0.0209 на 09-04; −0.0555 против −0.0622 на 10-16).
Это перераспределение массы, а не новая различающая способность — что согласуется с плоским
AUC. Нули дают ~39 % строк при RMSLE 1.84–1.87 и перевешивают.

### Correction geometry ухудшается именно у REAL

```
                             BASE                          GLOBAL                        PLACEBO
correction_var         .002072 .003028 .001472 .001505  |  .001608 .002940 .001242 .001424  |  .001486 .003233 .001375 .001282
extreme |corr|>0.10    .0976  .4030  .1637  .1700       |  .0696  .3519  .0809  .1307       |  .0630  .4177  .1156  .0934
corr(correction, resid).00542 .01011 .01272 .00965      |  .00261 .00997 .00841 .00670      |  .00603 .01180 .01335 .00901
```

Ключевое: корреляция коррекции с фактическим остатком падает **именно у GLOBAL**
(0.00261 против 0.00542 у BASE и 0.00603 у PLACEBO на 09-04; 0.00670 против 0.00965 / 0.00901
на 10-16). Placebo эту корреляцию держит на уровне BASE. То есть реальные global-признаки
активно **разъюстируют** коррекцию относительно остатка, который она должна чинить. Это
механизменное объяснение проигрыша, а не просто «нет сигнала».

### Улучшение не сосредоточено в сегменте

`GLOBAL − BASE`, калиброванная RMSLE, взвешенно по фолдам:

```
rec_buy_0_7          +0.000033      never_bought         +0.000091
rec_buy_8_14         +0.000019      tenure_low           +0.000076
rec_buy_15_60        +0.000109      tenure_high          +0.000061
rec_buy_61_plus      +0.000018      hist_support_0       +0.000081
                                    hist_support_1_3     +0.000033
                                    hist_support_4_plus  +0.000063
```

Хуже во всех десяти когортах. Канонический пробник проекта `rec_buy 15–60` — **+0.000109**,
худший результат из всех сегментов. Спасать сегментным гейтом нечего.

---

## SUBMISSION STATUS

**Сабмит не создан.** STRONG PASS не достигнут, §7 не запускался.
`submission_GLOBAL_REGIME_OCC.csv` отсутствует намеренно.

Независимо от гейтов production-ветка была бы заблокирована support-аудитом: 70.4 %
global-признаков вне обучающего диапазона на тесте (77.4 % относительно тех 10 cutoff'ов,
на которых реально учится occurrence), 31 из 115 имеют Spearman с порядком cutoff'а ровно
±1.0000.

---

## VERDICT

**REJECT.** Cutoff-safe global monetization regime и cross-sectional trajectory
пользователя **не добавляют** occurrence-сигнала поверх `occ_r10_fast`.

Три независимые оси дают один знак:

1. **Standalone.** wAUC −0.000050 к BASE, 0/4; logloss хуже; **хуже matched placebo** (−0.000075).
2. **Full-ensemble.** ΔwCV **+0.000057**, 0/4, 10-16 хуже; `GLOBAL − PLACEBO` **+0.000065**, 0/4.
3. **Механизм.** Признаки выучены (sd(Δlogit) 0.06, до 0.11 по вероятности), но двигают
   модель ровно как placebo, и корреляция коррекции с остатком у REAL падает вдвое.

Дополнительно — два структурных факта, из-за которых ветку не спасти настройкой:

* **Универсума платформы не существует.** В датасете ровно 250 000 пользователей, и это тот
  же набор, что в `sample_submit`. Они отобраны правилом организатора на тестовом cutoff'е,
  поэтому «рост активных ×1.032» — наполовину артефакт отбора. Тот же класс, что уже записан
  в `STATE.md` для `T ≥ 2025-10-17` (`e08`, `exp_028`).
* **Шок лежит вне коридора.** `g_d30_dlog_gmv` ∈ [−0.0092, +0.1077] на 29 чистых cutoff'ах
  против **−0.1881** на тесте. LightGBM экстраполирует биннингом, поэтому на тесте
  decline-признаки прижались бы к обучающему минимуму, а `x_global_decline_X_*` умножались
  бы на зажатую константу — механизм гипотезы на тесте физически не срабатывает. Плюс
  level-блок это cutoff-index под другим именем (31 признак с |ρ| = 1.0000).

**Не повторять** в этой ветке: sweep окон `7/14/30/60/90`, другие формы динамики (ratio
вместо log1p-разности), расширение набора перцентильных метрик, другие страты placebo,
tau/rounds/leaves/feature_fraction, сегментные гейты, а также попытку «починить» support
нормировкой global-признаков — level-блок при этом остаётся монотонной функцией cutoff'а,
а dynamics-блок остаётся вне диапазона.

**Что осталось не закрыто.** Проверялся re-anchored endpoint (`table_core`), а не
`X3`/`latest`, потому что OOF-банка товарища нет. Если банк `_best_bas_research/checkpoints`
появится, точный прогон стоит **~25 мин** (arms уже обучены, нужен только replay overlay на
анкере `A`). Вердикт он, скорее всего, не изменит: standalone-ось, где базы нет вообще,
даёт тот же знак.

---

## ВОСПРОИЗВЕДЕНИЕ

```bash
python src/global_regime_occ.py --stage global-state      # ~4 мин
python src/global_regime_occ.py --stage support-audit     # секунды
python src/global_regime_occ.py --stage hurdle            # ~60 мин (4 фолда x 2 LightGBM)
python src/global_regime_occ.py --stage occ --arm BASE     # ~8 мин
python src/global_regime_occ.py --stage occ --arm GLOBAL   # ~20 мин
python src/global_regime_occ.py --stage occ --arm PLACEBO  # ~20 мин
python src/global_regime_occ.py --stage evaluate          # ~3 мин
pytest src/test_global_regime_occ.py                      # 41 тест, 11 с
```

41 тест покрывает: cutoff-safety (усечение будущего — no-op на всём векторе признаков),
корректность агрегации (ручной счёт на синтетическом логе), детерминизм перцентиля
(перестановочная инвариантность + побитовый повтор), отсутствие доступа к target-окну,
**BASE exact reproduction** (`assemble_augmented(arm=None)` побитово равен `train.assemble`),
placebo сохраняет маргиналы точно и двигает >50 % строк, выравнивание строк/пользователей,
детерминированный повтор (SHA256 блока совпадает при повторном построении), схему сабмита,
асимметрию `p_apply`, walk-forward overlay без подглядывания в оцениваемый фолд, сверку
сетки overlay с рецептом товарища.

Артефакты: `artifacts/GLOBAL_REGIME_OCC_EXP057/` — `baseline_reconstruction.json`,
`strongest_oof_replay.json`, `feature_manifest.json`, `global_state.parquet/csv`,
`global_support_audit.json`, `global_cutoff_monotonicity.json`, `fold_results.json`,
`hurdle_*.npz`, `occ_{BASE,GLOBAL,PLACEBO}_*.npz`, логи прогонов.
