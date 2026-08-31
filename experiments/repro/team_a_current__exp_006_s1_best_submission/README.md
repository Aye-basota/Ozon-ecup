# exp_006 — S1-BEST: итоговая конфигурация и сабмит

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_006_s1_best_submission`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_006_s1_best_submission`
- **Original source:** `experiments/exp_006_s1_best_submission.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, CatBoost, calibration diagnostic
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** панель     обучение — 1-блочная, валидация и тест — 3-блочная (правило организатора)
- **Known score:** CV mean 1.75886, std 0.01162** (B0: 1.76879, std 0.01448)
- **Seed:** feature_fraction 0.7, bagging 0.8/1, lambda_l2 5, max_bin 63, seed 42
- **Postprocessing:** level 2.339 --out submission_strategy_1.csv
- **Submission:** level 2.339 --out submission_strategy_1.csv
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_006 — S1-BEST: итоговая конфигурация и сабмит

- **Дата:** 2026-08-10
- **Автор:** A1 (Strategy 1)
- **Ветка:** `team-a-strategy-1-impl`

## Что это

Первая итоговая версия решения Strategy 1: смесь трёх наборов признаков в
лог-пространстве + якорная калибровка уровня.

## Конфигурация

```
cutoff'ы   2025-04-03 .. 2025-10-16, шаг 7 дней, 29 штук, 6 065 972 строки
панель     обучение — 1-блочная, валидация и тест — 3-блочная (правило организатора)
таргет     сумма gmv в (T, T+30], обучение на log1p(y), L2
модель     LightGBM 600 раундов, lr 0.05, num_leaves 127, min_data_in_leaf 200,
           feature_fraction 0.7, bagging 0.8/1, lambda_l2 5, max_bin 63, seed 42

смесь (усреднение z, НЕ сырого GMV):
  0.45  S1-NORM  227 призн.  длинные окна нормированы на доступную глубину истории
  0.45  S1-UNC   236 призн.  ненормированные
  0.10  S1-CAP   195 призн.  история усечена до 180 дней (страховка, adv AUC 0.686)

постпроцесс  z_cal = max(z − 0.1486, 0);  pred = expm1(z_cal)
```

## Результат на локальной валидации

| фолд | B0 | S1-BEST | Δ |
|------|----|---------|---|
| 2025-09-04 | 1.78975 | 1.77449 | −0.01526 |
| 2025-09-18 | 1.77428 | 1.76484 | −0.00944 |
| 2025-10-02 | 1.75873 | 1.75170 | −0.00703 |
| 2025-10-16 | 1.75240 | 1.74443 | −0.00797 |

- **CV mean 1.75886, std 0.01162** (B0: 1.76879, std 0.01448)
- OOF 1.75871, bias −0.0737; после оптимального сдвига **1.75716** (B0: 1.76570)
- Прирост к B0: **−0.00993** по CV, **−0.00854** по калиброванному OOF, 4 фолда из 4

## Калибровка уровня

```
уровень модели на тесте (смесь)      2.4876
m_x на тесте (точно, sample_submit)  2.2421
целевой уровень (половина сезонной)  2.3390
δ = 2.3390 − 2.4876                  −0.1486
итог: mean(log1p(pred)) = 2.3391, якорный коридор 2.28..2.41 — пройден
```

## Сабмиты

| файл | уровень | δ | LB RMSLE |
|------|---------|---|----------|
| `submission_strategy_1.csv` | 2.3391 | −0.1486 | **1.6512803** |
| `submission_strategy_1_level_low.csv` | 2.2803 | −0.2076 | 1.6519790 |
| `submission_strategy_1_level_high.csv` | 2.4051 | −0.0825 | 1.6529909 |

Значения LB как их отдаёт платформа, без округления: `1.6512802628833827`,
`1.6519789982910107`, `1.6529908823677866`.

**Разбор по LB.** Три точки при известных уровнях дают точное решение
`MSE(L) = C + (L* − L)²` (модель совпадает с замерами до 7-го знака):

```
истинный уровень таргета  L* = 2.3293      (наша ставка 2.3391, промах +0.0098)
минимальный RMSLE          = 1.6512518     (мы получили 1.6512803, зазор 0.0000285)
фактический сезонный сдвиг dm = +0.0872    (YoY-аналог давал +0.1630, «нет сезона» +0.0308)
калибровка стоила          +0.00757        относительно некалиброванной модели (2.4876)
```

Проверки основного файла: 250 000 строк, порядок `user_id` == `sample_submit`,
дубликатов 0, NaN/inf 0, отрицательных 0, min 0.000000, max 3 987.4,
нулевых предсказаний 0.243%.

## Вердикт и вывод

Принято как **S1-BEST**. Выбран не по максимуму CV: одиночный `S1-E10` даёт
1.75988, смесь — 1.75886, разница мала, но смесь дополнительно снижает зависимость
результата от одного допущения об экстраполяции длинных признаков.

Главная неопределённость — **уровень** прогноза — снята замером на LB и оказалась
выбрана почти оптимально: зазор до минимума 0.0000285. Коэффициент 0.5 на сезонную
поправку, предписанный `strategy_1.md` из-за `n = 1 год`, попал в цель.

Дальнейший прирост возможен только за счёт уменьшения остаточной дисперсии, то есть
самой модели: число раундов, двухчастная постановка, CatBoost, веса cutoff'ов.

## Воспроизведение

```bash
python -m src.predict --exp S1-NORM --variant S1-NORM --L 0 --norm-long --min-history 90 --train-blocks 1
python -m src.predict --exp S1-UNC  --variant S1-UNC  --L 0             --min-history 90 --train-blocks 1
python -m src.predict --exp S1-CAP  --variant S1-CAP  --L 180           --min-history 90 --train-blocks 1
python -m src.submit --z S1-NORM S1-UNC S1-CAP --weights 0.45 0.45 0.10 \
                     --level 2.339 --out submission_strategy_1.csv
```
