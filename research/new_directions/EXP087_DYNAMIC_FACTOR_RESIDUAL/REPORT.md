# EXP087 — Dynamic Factor Residual

## Verdict

**REJECT_ORACLE**.

Cross-user synchronous temporal structure в cutoff-safe behavioral tensor существует: на каждом
fold все первые 16 singular modes выше fold-specific 99-percentile permutation null, поэтому rank
упирается в заранее заданный hard cap `K=16`. Но эта структура почти не содержит условной
информации о production residual даже при использовании реальных будущих behavioral channels.

Recency-weighted after-span oracle MSE headroom равен только **0.000312183**, то есть в `3.20x`
ниже обязательного gate `0.0010`. Ни один из четырёх folds не достигает gate. После поправки на
finite-sample degrees-of-freedom bias диагностический headroom снижается до `0.000098886`.

По правилу эксперимента branch завершён до forecasting/model training. Factor forecast,
`Z_hat`, nested coefficients, bootstrap, placebos, TEST inference и submission не строились.

## Factor construction

### Inputs and cutoff safety

Использован audited raw-derived panel
`C:\Users\Admin\Desktop\OZON-E-CUP\data\processed\seq_panel_v1.npy`, SHA256
`2ff92d6b3890a8aa66347f4063c99cdd312973f1ed6e7b69d1faa1240854e0bd`. Его raw parity и
user/date identity ранее проверены EXP075/EXP085. Для каждого cutoff cohort ограничена ровно
cutoff-safe eligible user IDs из exact EXP082 production components. Окно representation —
последние 180 календарных дней, включая cutoff; oracle window — строго следующие 30 дней.

Использованы восемь каналов:

```text
searches
cat
search_to_cart
search_to_ord
cat_to_cart
cat_to_ord
presence
buy_day
```

`gmv_search`, `gmv_cat` и `gmv` не использованы. Это заранее исключает превращение branch в
обычный monetary-level predictor.

### Normalization

Для шести count channels применён `log1p`; binary `presence/buy_day` оставлены индикаторами.
Затем отдельно для каждого channel:

1. удалён historical per-user mean на 180 днях;
2. удалён cohort mean каждого дня;
3. channel разделён на robust scale `(Q75-Q25)/1.349`, оценённый на фиксированной target-free
   выборке 8,192 users;
4. для oracle future использованы frozen historical user means и scales; future cohort-day mean
   удалён только внутри разрешённого oracle diagnostic.

Максимальная абсолютная корреляция отдельного loading с activity-days составляет
`0.0830 / 0.0676 / 0.0351 / 0.0356` по folds; с production baseline —
`0.0520 / 0.0860 / 0.0655 / 0.1019`. После обязательной full-span projection эти линейные
overlaps удалены.

### Low-rank procedure

Применён CPU-only эквивалент CP/SVD procedure без neural model. Для нормализованных matrices
`X_c[user, day]` построен day-mode Gram:

```text
S = sum_c X_c.T @ X_c
```

Eigenvectors `S` дают temporal modes. Для каждого mode его user-by-channel projection разложен
лучшей rank-1 аппроксимацией `L[user,k] * W[channel,k]`. Historical `F[day,k]` и oracle future
`F_future[day,k]` затем получены совместной least-squares projection на frozen
`L*W` basis. Median separable user-channel variance fraction находится примерно в диапазоне
`0.61–0.68`; condition number совместного factor Gram — `1.08–1.13`.

### Parallel analysis and K

Для каждого fold использованы 100 null repetitions. В каждом repetition users независимо
circularly permuted внутри каждой пары day/channel. Это сохраняет точное univariate
day/channel distribution и разрушает user identity synchrony. Spectrum считался на фиксированной
target-free подвыборке 1,024 users; factor model после выбора rank строился на всей fold cohort.

| Fold | N users | significant modes before cap | K | min/max real ÷ null-p99, first 16 | top-16 variance fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-07-03 | 181,338 | >=16 | 16 | 1.056 / 1.710 | 0.12049 |
| 2025-08-07 | 184,617 | >=16 | 16 | 1.119 / 1.520 | 0.11968 |
| 2025-09-11 | 189,815 | >=16 | 16 | 1.133 / 1.362 | 0.11802 |
| 2025-10-16 | 197,379 | >=16 | 16 | 1.122 / 1.575 | 0.11791 |

Вывод: latent cross-user structure достоверно присутствует, но `K=16` фиксирован cap и не
увеличивался после результата.

## Oracle headroom

Для frozen historical basis реальные следующие 30 дней behavioral activity спроецированы в
`F_oracle`; затем `G_oracle[k] = sum_t F_oracle[t,k]` и
`Z_oracle[user,k] = L[user,k] * G_oracle[k]`. После удаления constant `Z_oracle` дважды
спроецирован из exact span `[1, CAP, UNC, DIST, SEQ, ETX, z_production]`.

Headroom рассчитан точно по preregistered формуле
`b.T @ pinv(G) @ b`, без target-based rank/representation selection.

| Fold | K | oracle rho | after-span oracle MSE | debiased diagnostic | passes 0.001 |
| --- | ---: | ---: | ---: | ---: | :---: |
| 2025-07-03 | 16 | 0.009921 | 0.000318329 | 0.000032946 | NO |
| 2025-08-07 | 16 | 0.009630 | 0.000301617 | 0.000019084 | NO |
| 2025-09-11 | 16 | 0.014076 | 0.000616461 | 0.000353045 | NO |
| 2025-10-16 | 16 | 0.007307 | 0.000161917 | 0.000000000 | NO |
| **recency-weighted 1:2:4:8** | — | **0.009596 diagnostic** | **0.000312183** | **0.000098886** | **NO** |

`0/4` folds имеют headroom `>=0.0010`; требование “желательно минимум 3 folds” также не
выполнено. Latest fold — самый слабый. Поэтому oracle failure нельзя объяснить одним ранним
regime.

## Factor forecast quality

**NOT RUN — blocked by oracle gate.**

Fixed Ridge/VAR forecast `F(t-1), F(t-7), F(t-28), DOW` не обучался. Следовательно, текущий
эксперимент не утверждает ни forecastability, ни non-forecastability future factors: conditional
residual headroom недостаточен уже при идеальном знании future factor trajectory.

## Purged residual results

**NOT RUN — blocked by oracle gate.**

| Fold | rho | Delta MSE | Delta RMSLE |
| --- | ---: | ---: | ---: |
| 2025-08-07 | N/A | N/A | N/A |
| 2025-09-11 | N/A | N/A | N/A |
| 2025-10-16 | N/A | N/A | N/A |
| weighted | N/A | N/A | N/A |

Ни `Z_hat`, ни past-only coefficients `a`, ни cluster bootstrap не создавались. Expected robust
deployable `Delta RMSLE` для принятого решения равен **0**: correction не выпускается.

## Placebo controls

**NOT RUN — blocked by oracle gate.**

Shuffled-loadings и time-shuffled-factor controls определены для forecasted `Z_hat`. Запускать их
после `REJECT_ORACLE` означало бы продолжить branch вопреки обязательному раннему stop. Поэтому
REAL-vs-placebo deployable comparison отсутствует; oracle result сам по себе не объявляется
causal predictive effect.

## Production overlap

Ниже RMS рассчитан по всем entries feature matrix `Z_oracle`; fraction —
`RMS(Z_perp) / RMS(centered Z_raw)`.

| Fold | raw RMS | after-span RMS | perp fraction | second-pass relative error |
| --- | ---: | ---: | ---: | ---: |
| 2025-07-03 | 10.67259 | 10.61522 | 0.99462 | 1.72e-15 |
| 2025-08-07 | 8.11853 | 8.09807 | 0.99748 | 6.55e-16 |
| 2025-09-11 | 6.38505 | 6.36360 | 0.99664 | 7.26e-16 |
| 2025-10-16 | 6.61276 | 6.57641 | 0.99450 | 1.30e-15 |
| **weighted** | — | — | **0.99548** | — |

Correction не исчезает в production span: почти вся factor-feature energy остаётся perpendicular.
Следовательно, отрицательный verdict вызван отсутствием residual covariance, а не in-span
annihilation. Условие `perp_fraction >= 0.20` относится к TEST и не проверялось; показанные числа
— только historical oracle diagnostic.

## Mathematical headroom

| Quantity | Value |
| --- | ---: |
| required current MSE gap | 0.004909274 |
| weighted historical production residual MSE | 3.101528946 |
| weighted raw oracle headroom | 0.000312183 |
| equivalent oracle rho² | 0.000100655 |
| equivalent oracle rho | 0.010033 |
| raw oracle fraction of required gap | 6.3590% |
| debiased oracle fraction of required gap | 2.0143% |
| deployable robust fraction of gap | 0.0000% |

Даже nondeployable perfect-future oracle изменил бы weighted historical RMSLE примерно на
`-0.00008863`; debiased diagnostic — примерно на `-0.00002808`. Это существенно меньше требуемого
gap и является верхней диагностической границей, а не ожидаемым production uplift.

## TEST output

**NOT CREATED.** `STRONG_GO` невозможен после `REJECT_ORACLE`, поэтому cutoff `2026-02-13`,
current submission span и anchor не читались для inference. Candidate path и candidate SHA256
отсутствуют; автоматическая отправка не выполнялась.

## Final conclusion

1. Cross-user latent temporal factor structure в behavioral tensor существует и статистически
   превышает permutation null; значимых modes не меньше 16.
2. После полного production span oracle headroom всего `0.000312183 MSE` weighted и
   `0.000161917` на latest fold; `0/4` folds проходят gate.
3. Future factor forecastability не проверялась, потому что даже perfect-future oracle слишком
   слаб для продолжения.
4. Weighted/latest purged `rho` для `Z_hat` — `N/A / N/A`; deployable модель не обучалась.
5. REAL-vs-placebo — `N/A`: controls корректно остановлены вместе с forecast stage.
6. Expected robust deployable `Delta RMSLE = 0`; никакая correction не применяется.
7. Optimistic perfect-future oracle закрывает `6.36%` current MSE gap; после df-bias diagnostic —
   `2.01%`; доказанная deployable доля — `0%`.
8. Финальный verdict: **REJECT_ORACLE**. Rank/window/learner/scale rescue sweep не выполнялся.

Leaderboard data не использовались. GPU не использовался. Полный oracle run занял около 89 секунд
CPU wall time.
