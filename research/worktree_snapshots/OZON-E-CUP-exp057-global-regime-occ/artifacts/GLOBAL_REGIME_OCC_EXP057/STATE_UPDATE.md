# Готовые строки для STATE.md и experiments/log.csv

**Не применены намеренно.** В основном репозитории `STATE.md` и `experiments/log.csv`
имеют незакоммиченные правки (результаты exp_054…060), а этот worktree стоит на `HEAD`.
Правка HEAD-версии выглядела бы при слиянии как откат чужой работы, что запрещено.
Вставить вручную в рабочую копию основного репозитория.

**Внимание, коллизия ID:** номер `057` уже занят за `STATE-REWEIGHT`. Ниже используется
`EXP-061`; если предпочтителен другой номер — заменить в обоих местах и переименовать
`experiments/exp_057_global_regime_occ.md` и `artifacts/GLOBAL_REGIME_OCC_EXP057/`.

---

## 1. Строка в таблицу «Последние эксперименты» (сверху)

```
| GLOBAL-REGIME-OCC | 2026-08-25 | A1 | EXP-061: cutoff-safe global monetization regime + cross-sectional percentile trajectory как входы фиксированного `occ_r10_fast`, против architecture-matched placebo | ensemble GLOBAL−BASE **+0.000057**, 0/4; GLOBAL−PLACEBO **+0.000065**, 0/4; standalone wAUC **−0.000050** | **REJECT; full folds пройдены, test/LB NO** (`exp_057_global_regime_occ`) |
```

## 2. Строка в «Не повторять»

```
- **Global monetization regime и percentile-траектория пользователя как входы `occ_r10_fast`** (`exp_057_global_regime_occ`, EXP-061): baseline восстановлен точно (`latest = .12/.16/.72` с max|Δz| 8.9e-16, `friend` OOF wCV 1.7475098627), 140 признаков, 41 тест. Ensemble `GLOBAL−BASE` **+0.000057, 0/4**, 10-16 +0.000060; `GLOBAL−PLACEBO` **+0.000065, 0/4**; standalone wAUC −0.000050, logloss хуже, **хуже matched placebo**. Признаки честно выучены (sd(Δlogit) 0.06 = ровно как у placebo), но `corr(коррекция, остаток)` у REAL падает вдвое (0.00261 против 0.00542 BASE / 0.00603 PLACEBO). Хуже во всех 10 когортах, `rec_buy 15–60` **+0.000109** — худший сегмент. **Структурно неспасаемо:** в датасете ровно 250 000 пользователей = множество `sample_submit`, отобранное правилом организатора на ТЕСТОВОМ cutoff'е, поэтому «рост активных ×1.032» наполовину артефакт отбора (класс `e08`/`exp_028`); шок GMV лежит вне коридора — `g_d30_dlog_gmv` ∈ [−0.0092, +0.1077] на 29 чистых против **−0.1881** на тесте, 70.4% global-признаков вне обучающего диапазона, а 31 из 115 имеет Spearman с порядком cutoff'а ровно **±1.0000** (то есть это cutoff-index под другим именем). **REJECT; не спасать окнами, формой динамики, набором перцентильных метрик, стратами placebo, tau/rounds/leaves или сегментными гейтами; нормировка global-признаков support не чинит.**
```

## 3. Строка в experiments/log.csv

```
GLOBAL_REGIME_OCC_EXP057,2026-08-25T02:40:00+03:00,a28a71f,"exp_057_global_regime_occ (EXP-061): cutoff-safe global platform regime + cross-sectional percentile trajectory + preregistered user x global interactions as inputs to the exact occ_r10_fast recipe, against an architecture-matched cyclic-shift/within-strata-permutation placebo",S1,227/367,lgb-binary-occurrence,"{""recipe"":""occ_r10_fast exact"",""maxcuts"":10,""tau"":55.0,""rounds"":380,""leaves"":31,""min_leaf"":520,""feature_fraction"":0.82,""seed"":42,""new_features"":140,""arms"":[""BASE"",""GLOBAL"",""PLACEBO""],""base"":""table_core re-anchored (teammate OOF bank absent)"",""table_weight"":0.55,""overlay"":""p_apply asymmetric, walk-forward params""}",4 clean folds,,3,"[1.766963, 1.760465, 1.748574, 1.741263]",,,,,1.7474862831,,,,REJECT,"GLOBAL-BASE +0.000057 0/4; GLOBAL-PLACEBO +0.000065 0/4; standalone wAUC -0.000050; 70.4% global features outside test support; 31/115 Spearman +-1.0000 with cutoff order; dataset universe == 250k submission users",1.7474862831,"[1.766963, 1.760465, 1.748574, 1.741263]",,
```

Проверить перед вставкой, что порядок колонок совпадает с текущим заголовком `log.csv`:

```
exp_id,timestamp,commit,description,scenario,n_features,model,params,cutoffs,L,panel_blocks,
fold_scores,cv_mean,cv_std,bias_mean,best_offset,cv_mean_calib,delta_vs_b0,runtime_s,verdict,
conclusion,wcv,fold_cal,mean_z,lb_public
```
