# HISTORY — архив экспериментов

Сюда переносятся старые строки таблицы из STATE.md, когда их становится больше 10.
Новые сверху. Провалившиеся гипотезы при архивации обязаны остаться
строкой в «Не повторять» в STATE.md.

С `exp_016` главная метрика проекта — **wCV** (веса фолдов 1:2:4:8, пофолдовые
RMSLE после оптимального лог-сдвига). Старые строки ниже приведены в CV, новые —
в wCV; колонка помечена. Пересчёт любой строки: `python -m src.report` через
`from_oof(<exp_id>)`, значения уже дозаполнены в `experiments/log.csv`.

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| LEVEL-MINUS-006 | 2026-08-24 | A1 | EXP-060: fixed production-level diagnostic, единственное изменение `z_STRONGEST−0.06` | CV/training NONE; mean z **2.329321→2.269499** | **PREPARED; submission создан, LB не отправляли** (`exp_060`) |
| SEQ65-TEMPORAL | 2026-08-24 | A1 | EXP-059: fixed production-regime probe, sequence 0.45→0.65 при level 2.3293 | **−0.000238 к STRONGEST, 4/4** | **READY; submission создан, LB не отправляли** (`exp_059`) |
| RECENCY-RIDGE-REPLAY | 2026-08-25 | A1 | EXP-068: exact teammate Ridge-stack replay + redundancy audit к `latest` | formula replay **BLOCKED**; TEST corr latest **0.999968** | **BLOCKED_HISTORICAL_REPLAY** (`exp_068`) |
| STATE-REWEIGHT | 2026-08-24 | A1 | EXP-057: target-free semantic state matching fixed UNC slot против exact within-state shuffle | slot REAL−SHUF **+0.000002**; REAL−STRONGEST **+0.000016**; halves disagree | **REJECT; full folds/test/LB NO** (`exp_057`) |
| LATE-SSL | 2026-08-24 | A1 | EXP-056: exact ETX input-only masked reconstruction на late histories против matched clean SSL control | LATE−CONTROL cal **+0.000032**; slot **+0.000007** | **REJECT; full folds NO** (`exp_056`) |
| FINGERPRINT | 2026-08-24 | A1 | EXP-058: dataset/user identity и extraction metadata против fixed incidence-matched joint PERM | slot REAL−PERM **+0.000071**; REAL−STRONGEST **−0.000023** | **REJECT; full folds/test/LB NO** (`exp_058`) |
| LANDMARK-MEMORY | 2026-08-24 | A1 | EXP-055: 16 cutoff-safe historical state→realized-GMV landmarks против matched outcome shuffle | scale **0/0**; late Δ **0**; REAL−SHUF **0** | **NO_GO_PREFLIGHT; GPU/pilot NO** (`exp_055`) |
| BURST-GAP-ETX | 2026-08-24 | A1 | EXP-054: fixed threshold-3 activity episodes + explicit gaps против matched joint shuffle | scale **0/0**; late Δ **0**; AUC excess **+0.000021** | **NO_GO_PREFLIGHT; GPU NO** (`exp_054`) |
| RESDISC | 2026-08-24 | A1 | EXP-053: artifact-only semantic/seed oracles + two-sided winner gate и signed-residual probe | 10-16 gate **−0.000006**; residual **≈0** | **NONE; full LOFO NO** (`exp_053`) |
| CHANNEL-SHAPLEY | 2026-08-24 | A1 | EXP-052: две Shapley Search/Catalog heads против matched shuffled-composition control | 10-16 residual **Δ=0**, alpha `0/0`; REAL−SHUF **+0.001266** | **REJECT; full folds NO** (`exp_052`) |
| BTYD-STABLE | 2026-08-23 | A1 | EXP-051: analytic-jac strict BG/NBD fit одинаково OOF/test + production support | nested **−0.000269, 4/4**; fixed .05 **−0.000321, 4/4** | **CASE B PASS; `submission_BTYD05.csv` создан** (`exp_051`) |
| BTYD05-PROD | 2026-08-23 | A1 | EXP-050: exact FRESH parity + two-sided BTYD production at 2026-02-13 | — | **BLOCK: FRESH parity FAIL; BTYD MLE unstable; no submission** (`exp_050`) |
| SELMATCH-PROD | 2026-08-23 | A1 | EXP-049: corrected same-fold 3F analysis + bootstrap/shuffle fix + production-support audit | conditional k>0 **−0.000551, 3/3** | **REJECT: validation PREFERRED, production artifacts absent** (`exp_049`) |
| SELMATCH-CV | 2026-08-23 | A1 | EXP-048: artifact-only future-continuation audit + natural reference + pseudo-matched/bootstrap/shuffle | conditional best **−0.000551** | **TECHNICAL_INCONCLUSIVE: k=0 unsupported** (`exp_048`) |
| BTYD-DAY-BGNBD | 2026-08-23 | A1 | EXP-047: common-origin BG/NBD purchase-day likelihood + fixed S2 monetary/aggregation | **−0.000269, 4/4** | **REJECT; classic BTYD closed** (`exp_047`) |
| TBR-REFRESH | 2026-08-23 | A1 | EXP-046: production UNC/CAP, exact replay + fixed factorial AVG3 × rounds | **D−H −0.000002**, 3/4 | **REJECT; production NO** (`exp_046`) |
| BUYCTRL-DET | 2026-08-23 | A1 | EXP-045: plain SEQ-01 + настоящая `buy30` BCE против cutoff-wise shuffle, paired 3 seeds | 10-16: **TRUE−SHUF +0.000436**, 1/3 | **FAIL; full folds NO** (`exp_045`) |
| FRESH-COND-FT | 2026-08-22 | A1 | EXP-044: paired encoder FT, fresh positive-only conditional supervision против equal-volume VOL | 10-16: **Δ=−0.000088**, 3/3 | **REJECT; full folds NO** (`exp_044`) |
| DET-PAIR | 2026-08-22 | A1 | EXP-043: checkpoint → новый AdamW, fixed plan + deterministic CUDA, 2 repeats | 10-16: **Δ=0**, `Var(Δz)=0` | **PASS; floor ≤1e-4** (`exp_043`) |
| ZERO2D-SHRINK | 2026-08-21 | A1 | EXP-042: negative-only EB/isotonic correction по amount × `p0_DIST`, AMOUNT/shuffle controls | **−0.000025 к STRONGEST** | **REJECT 2/4; LB probe готов** (`exp_042`) |
| FRESH-CONTRAST | 2026-08-21 | A1 | EXP-040: `z_FRESH-z_CLEAN` как GLOBAL/HIGH16 residual, full user cross-fit + VOL | **−0.000225 к STRONGEST** | **REJECT 4/4** (`exp_040`) |
| RIDGE15 | 2026-08-21 | A1 | Ridge на `S1-E10`, фиксированные 15% вместо UNC / UNC+DIST | **+0.000278 / +0.000207, 0/4** | **REJECT 0/4; LB probe готов** (`exp_041`) |
| BLOCK4-SAF | 2026-08-21 | A1 | EXP-039: `q·(nu_F−nu_C)` из двух guaranteed-active блоков, user cross-fit + shuffle | **+0.00024 к STRONGEST** | **REJECT 0/4** (`exp_039`) |
| FNL-FUNNEL | 2026-08-21 | A1 | EXP-038: future Search/Cart как aux-супервизия SEQ-D3A против контроля `buy30` | 10-16: −0.00016/−0.00118 | **REJECT** (`exp_038`) |
| STRONGEST-CURRENT | 2026-08-20 | A1 | EXP-037: слот `0.5·ETX-AVG3 + 0.5·SEQ-AVG3`, режим теста после фикса статика | **1.74751** | **ОТПРАВЛЕН, LB 1.6496571 — новый чемпион** (`exp_037`) |
| ETX-AVG3 | 2026-08-20 | A1 | EXP-037: 3 сида × 4 фолда + причина блокера `exp_036` | **1.74861** | **ACCEPT как соавтор** (`exp_037`) |
| ETX-01-S42 | 2026-08-20 | A1 | S_13: sparse event transformer, 4 фолда + LOFO слота | **1.74953** (ΔwCV −0.00331) | **пара с TCN −0.00091 4/4** (`exp_036`) |
| MIX9-SEL | 2026-08-19 | A1 | LOFO слота SEQ: `D3A-AVG3` против `SEQ-AVG3 @ clip289` | −0.00061 / −0.00055 | **выбран `SEQ-AVG3`** (`exp_035`) |
| S04-PROD-FRESH | 2026-08-19 | A1 | EXP-032B: боевой экстенсив `1−p0` из `S1-DIST` под тот же `μ_FRESH` | −0.00101 / −0.00247 (гр. A) | **PASS гейта, в смесь НЕ годится** (`exp_032b`) |
| SEQ-D3A-MS | 2026-08-19 | A1 | EXP-030: 3 сида × 4 фолда, парные дельты | ΔwCV **−0.00095** | **KEEP в смесь** (`exp_030c`) |
| S04-SEQ-FRESH | 2026-08-19 | A1 | EXP-032: интенсивная голова на CLEAN+EXTRA, frozen энкодер | −0.00128 (гр. A) | **CONTINUE** (`exp_032`) |
| SEQ-D3A-S43 | 2026-08-19 | A1 | EXP-030b: воспроизводится ли провал 09-18 (сид 43) | 09-18 −0.00035 | **SCALE TO 3 SEEDS** (`exp_030b`) |
| SEQ-D3A | 2026-08-18 | A1 | EXP-030: depth curriculum, обрезка РЕАЛЬНЫХ дней (wCV) | **1.75284** | **CONTINUE** 2/4 (`exp_030`) |
| SEQ-AVAIL-AUG | 2026-08-14 | A1 | train-only сдвиг границы `avail`, включая `avail≡1` (wCV) | — | **REJECT** (`exp_029`) |
| FRESH-DIST-MIX | 2026-08-13 | A2 | +13 физически размеченных, но selection-contaminated cutoff'ов (wCV) | — | **REJECT audit** (`exp_028`) |
| SEQ-DEPTH | 2026-08-13 | A1 | разбор провала LB: кросс-фолдовый стресс глубины + цена ухода `avail` (wCV) | — | **ACCEPT** (`exp_027`) |
| SEQ-AVG3 | 2026-08-13 | A1 | S_10: усреднение 3 сидов TCN + диагностика глубины истории (wCV) | 1.74963 | ACCEPT приём, но LB 1.6553136 (`exp_026`/`exp_027`) |
| SEQ-01 | 2026-08-13 | A1 | S_10 B: dilated TCN на сырой дневной истории 365д вместо 227 агрегатов (wCV) | 1.75270 | **CONTINUE** (`exp_025`) |
| MHZ-FULL | 2026-08-13 | A1 | S_03: multi-horizon hazard + счёт как супервизия (wCV) | 1.75234 | **REJECT** (`exp_024`) |
| HOLIDAY-YOY | 2026-08-12 | A1 | персональная holiday-response 2025→2026 + placebo (wCV) | 1.74958 | SEND_HIGH_RISK (`exp_023`) |
| S1-DIST-MIX | 2026-08-11 | A1 | смесь с головой распределения (wCV) | 1.74948 | LB 1.6507774 |
| S1-ROUNDS | 2026-08-12 | A1 | S_05 A: кривая по раундам `direct`, 25..1600 (wCV) | 1.75108 | в разработку; 600 = переобучение |
| S1-SEEDAVG5 | 2026-08-12 | A1 | S_05 B: усреднение 5 сидов direct при 300 раундах (wCV) | 1.75037 | в разработку; брать 3 сида |
| S1-GAPAXIS | 2026-08-12 | A1 | S_01: gap-axis k=5 + k=11 control | gCV 1.75637 | REJECT |
| S1-SAMPLE-A | 2026-08-12 | A1 | S_02A: `train_blocks=0`, capacity + avg3 | 1.75057 | REJECT |
| PT-FULL | 2026-08-12 | A1 | S_08: 30 признаков личного времени, avg3 (+контроль ρ) | 1.75045 | REJECT; FAIL гипотезы |
| S1-SAMPLE-B | 2026-08-12 | A1 | S_02B: step 3 при равнообъёмном hash-sampling (wCV) | 1.75234 | FAIL; gate HDN/TCN |
| S1-VAL-W | 2026-08-11 | A1 | единая схема валидации wCV, калибровка по LB (`exp_016`) | — | ПРИНЯТО; схема проекта |
| S1-MIX-E11 | 2026-08-11 | A1 | перебор весов смеси по wCV (+E11, −E03a) (wCV) | 1.74911 | REJECT; LB 1.6510029 |
| S1-DIST-F4 | 2026-08-11 | A1 | обучение только на выборке фолда 10-16 | неизмеримо | REJECT; LB 1.6512012 |
| S1-DIST | 2026-08-11 | A1 | голова распределения: 16 бинов z вместо L2 (wCV) | 1.75062 | KEEP; член лучшей смеси |
| S1-E11 | 2026-08-10 | A1 | двухчастная модель на признаках S1-E10 (wCV) | 1.75070 | KEEP; в смеси не помог |
| EXP-SIM | 2026-08-10 | A1 | 50% best + 50% rank-cohort similarity (wCV нет) | 1.85169 | REJECT; LB 1.6682180 |
| EXP-MIN | 2026-08-10 | A1 | 15 устойчивых фичей, extreme regularization (wCV 1.76497) | 1.77335 | REJECT; LB 1.6674246 |
| S1-BEST | 2026-08-10 | A1 | смесь 3 наборов + якорная калибровка (wCV 1.74997) | 1.75886 | LB 1.6512803 |
| S1-E10 | 2026-08-10 | A1 | длинные окна, нормированные на глубину истории (wCV 1.75170) | 1.75988 | KEEP; член лучшей смеси |
| S1-E03a | 2026-08-10 | A1 | усечение истории L=180 (wCV 1.76064) | 1.76787 | MODIFY |
| S1-E03c | 2026-08-10 | A1 | усечение истории L=90 (wCV 1.79323) | 1.79931 | REJECT |
| S1-E04 | 2026-08-10 | A1 | uncapped минус длинные признаки (2 фолда) | 1.75705 | REJECT |
| S1-E02 | 2026-08-10 | A1 | плотная сетка cutoff'ов, шаг 7 (`exp_003`) — входит в лучшую смесь (wCV 1.75151) | 1.76182 | KEEP |
| S1-E01 | 2026-08-10 | A1 | 3-блочная панель на обучении (`exp_002`) | 1.76981 | REJECT |
| S1-B0 | 2026-08-10 | A1 | бейзлайн: uncapped, 3 свежих cutoff'а (`exp_001`) | 1.76879 | база |
