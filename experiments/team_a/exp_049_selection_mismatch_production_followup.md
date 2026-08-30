# exp_049 — corrected EXP-048 same-fold analysis / production audit

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево
- **Код:** `src/selection_mismatch_followup.py`
- **Результаты:** `research/strategies/results/SELMATCH_EXP049/`
- **Тип:** analysis/production only; model training = **NO**
- **Фиксированный кандидат:** `BTYD05_FRESH1`

## Гипотеза и methodological correction

В `exp_048` standard endpoint использовал четыре fold с весами `1:2:4:8`, а
pseudo-matched endpoint — три fold `1:2:4`. Поэтому прежний selection penalty
смешивал reweighting с удалением `2025-10-16`. Здесь все три схемы используют
строго `09-04/09-18/10-02` и веса `1:2:4`:

- `A_STANDARD_3F`: вся обычная population, без reweighting;
- `B_K3_3F`: только `k=3`, с собственной weighted calibration;
- `C_MATCHED_KPOS_3F`: reference `pi(k)` renormalized на `k>0`, затем exact
  per-fold weighted calibration.

`pi_ref(k=0)=0.004951197`; `k=0` отсутствует во всех трёх folds. Поэтому C —
только conditional `k>0` sensitivity, а не fully identified matched-CV.

## Exact candidate semantics

OOF endpoint восстановлен без подбора весов:

```text
BTYD05             = 0.95 * STRONGEST_CURRENT + 0.05 * z_BTYD
FRESH              = STRONGEST_CURRENT + fresh_processed_nested
BTYD05_FRESH1      = 0.95 * STRONGEST_CURRENT
                   + 0.05 * z_BTYD
                   + fresh_processed_nested
```

`fresh_processed_nested` — exact honest outer prediction `exp_040`: raw
`z_FRESH-z_CLEAN`, donor-fold 0.5/99.5% winsorization, `GLOBAL`, alpha `1`,
centering after clipping. Для потенциального production `exp_040` задаёт
GLOBAL/alpha=1 и этот preprocessing, но production inference не реализован.
`exp_047` задаёт exact two-sided OOF cross-fit, но не задаёт production refit или
fold-ensemble recipe. Эти различия не замалчиваются и не заменяются догадкой.

## Same-fold results

| candidate | A_STANDARD_3F | B_K3_3F | C_MATCHED_KPOS_3F | signs C |
|---|---:|---:|---:|---:|
| `BTYD05` | −0.000395 | −0.000429 | −0.000340 | 3/3 |
| `FRESH` | −0.000192 | −0.000181 | −0.000252 | 3/3 |
| **`BTYD05_FRESH1`** | **−0.000547** | **−0.000570** | **−0.000551** | **3/3** |
| `ZERO2D` | −0.000019 | +0.000001 | −0.000056 | 3/3 |
| `SEQ_SLOT_25` | −0.000045 | −0.000048 | −0.000040 | 2/3 |
| `SEQ_SLOT_50/current` | 0 | 0 | 0 | 0/3 |
| `SEQ_SLOT_75` | +0.000201 | +0.000204 | +0.000196 | 0/3 |

`BTYD05_FRESH1` fold deltas:

```text
A: -0.000835 / -0.000704 / -0.000396
B: -0.001045 / -0.000716 / -0.000378
C: -0.000605 / -0.000736 / -0.000445
```

Главная corrected selection shift для кандидата теперь
`C_MATCHED_KPOS_3F - A_STANDARD_3F = -0.00000464`, а не старая смесь эффекта
reweighting и удаления fold. То есть signal кандидата сохраняется, но почти не
объясняется selection reweighting.

## Statistical diagnostics

500 cluster-bootstrap replicates по `user_id` переоценивают weighted calibration
внутри каждого replicate:

| scheme | mean | 95% interval | P(delta<0) |
|---|---:|---:|---:|
| A_STANDARD_3F | −0.000545 | [−0.000657, −0.000427] | 1.000 |
| C_MATCHED_KPOS_3F | −0.000550 | [−0.000672, −0.000426] | 1.000 |

Противоречие `exp_048` имело конкретную причину, а не неверный boolean:

```text
printed real value  = matched candidate effect
shuffle interval    = distribution of (matched - standard) selection shift
outside boolean     = comparison of that selection shift with its interval
```

То есть таблица поставила рядом два разных estimand. В исправленном отчёте:

- signal-correction shuffle (200 permutations внутри
  `fold × k × rec_buy_bin × w180_buy_bin`) имеет central 90%
  `[+0.000219,+0.000396]`; real matched effect `−0.000551` находится снаружи в
  improving direction — **PASS**;
- selection-k shuffle (100 permutations) проверяет отдельно `C-A`: central 90%
  `[−0.000050,+0.000054]`, real shift `−0.00000464`, `outside=False` — корректно.

Residual alignment кандидата положительна на 3/3: A `0.0326/0.0301/0.0213`,
C `0.0267/0.0313/0.0228`. Interaction относительно суммы отдельных эффектов
слегка антагонистична: `+0.0000406` A и `+0.0000402` C; combined signal всё
равно проходит основной gate.

## Missing k=0 sensitivity

Все результаты ниже остаются неидентифицированными сценариями для массы
`0.4951%`:

| assumption for missing k=0 | implied full-reference delta |
|---|---:|
| neutral candidate effect | −0.000549 |
| k=0 behaves like observed k=1 | −0.000557 |
| like k=2 | −0.000548 |
| like k=3 | −0.000551 |

Чтобы обнулить observed gain при базовом k=0 profile как у k=1, missing stratum
должен добавить кандидату `+0.3858` squared-log-error на строку относительно
базы. Это показывает малую численную sensitivity, но не превращает C в fully
identified estimate.

## Production/test audit

OOF-side audit пройден:

- exact row/user/target alignment: 770,616 строк;
- corrections finite; std `BTYD05/FRESH/combined = 0.02374/0.02591/0.03810`;
- FRESH donor/recipient overlap = 0 на всех четырёх folds, algebra error
  `<=1.6e-15`, saved two-sided support PASS;
- all 8 BG/NBD fits stable: max gradient `3.48e-5`, max mean-NLL spread
  `2.34e-7`, max log-parameter spread `0.0945`;
- `P_alive` mean `0.99367`, expected-count mean `1.51478`, все finite;
- STRONGEST_CURRENT test: 250,000 unique users, exact sample user set,
  finite; штатный level shift `−0.1353833` даёт
  `mean(log1p(pred))=2.3293000`; schema reference SHA256
  `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.

Production-support audit, однако, **FAIL_MISSING_EXACT_PRODUCTION_SUPPORT**:

1. `exp_047` явно остановлен до test inference; `ztest_BTYD` отсутствует, а
   production refit/fold-ensemble rule не зарегистрирован.
2. `exp_040` явно остановлен до production inference; `ztest_FRESH` отсутствует,
   conditional-head weights не сохранены, поэтому exact test raw contrast
   получить без нового обучения нельзя.
3. Усреднить восемь fold fits BTYD, выбрать последний fit или перенести
   user-level OOF FRESH correction на test означало бы выдумать новый recipe.

Следовательно test correction quantiles, clipping share,
`Var(c_test)/Var(c_oof)` и соответствие test regime OOF support не определены.
Diagnostic gate `0.6..1.4` не может быть проверен и не подменяется post-hoc
rescale.

## Final gate и verdict

Validation-only evidence проходит **PREFERRED**:

```text
matched delta <= -0.0005   PASS (-0.000551)
sign 3/3                   PASS
signal shuffle             PASS
standard sign              PASS (-0.000547)
bootstrap                  PASS (P=1.000)
```

Но production audit обязан быть полностью PASS. Он не проходит из-за
отсутствия exact authorized production artifacts/recipe. Итоговый verdict:
**REJECT**. `submission_BTYD05_FRESH1.csv` не создан, submission slot не
потрачен, SHA256 неприменим.

## Проверки и воспроизведение

```text
python src/selection_mismatch_followup.py
python -m pytest src/test_selection_mismatch_cv.py src/test_fresh_contrast.py \
  src/test_btyd_day_bgnbd.py src/test_validation.py -q
# 53 passed
```

Основные artifacts: `same_fold_results.csv`, `bootstrap.csv`,
`shuffle_controls.csv`, `residual_alignment.csv`, `interaction.csv`,
`missing_k0_sensitivity.csv`, `production_audit.json`, `summary.json`,
`diagnostic_distributions.npz`, `REPORT.md`.
