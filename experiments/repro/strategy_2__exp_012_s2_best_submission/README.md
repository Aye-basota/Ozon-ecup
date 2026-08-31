# exp_012 — S2-BEST: полный CV и финальный submission

## Catalogue metadata

- **Catalogue ID:** `strategy_2__exp_012_s2_best_submission`
- **Namespace:** `strategy_2`
- **Experiment ID:** `exp_012_s2_best_submission`
- **Original source:** `git:3c1d86d836c7:experiments/exp_012_s2_best_submission.md`
- **Source ref:** `3c1d86d836c7b73519abe99f94686431852187cc`
- **Source commit:** `2e1d89d7904ee161939d9c6eed44fe16d4e4c549`
- **Kind:** git-history experiment card
- **Model:** LightGBM, two-part / hurdle, blend, calibration diagnostic
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Подтверждённая структурная конфигурация стабильна на четырёх фолдах и либо улучшит
- **Known score:** CV mean: **1.76831 ± 0.00967**; OOF 1.76817; mean bias −0.0442.
- **Seed:** seed из `src/config.py`.
- **Postprocessing:** калибровка `sigma=0.9, mu_shift=-0.1`; calendar `alpha=0.5`; level 2.3293;
- **Submission:** Сабмит: `submissions/submission_strategy_2.csv`. Проверки: 250k строк,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_012 — S2-BEST: полный CV и финальный submission

- **Дата:** 2026-08-11
- **Автор:** A2
- **Коммит:** working tree, `team-a-strategy-2-impl`

## Гипотеза

Подтверждённая структурная конфигурация стабильна на четырёх фолдах и либо улучшит
S1 в кросс-фолдовом бленде, либо останется самостоятельным S2-BEST.

## Что изменено относительно базы

Зафиксированы `K=5`, hurdle, train-only `(sigma,mu)` и проведены full CV,
calendar ablation, blend ablation и финальное обучение.

## Результат

- CV по фолдам: **1.78113 / 1.77351 / 1.76246 / 1.75612**.
- CV mean: **1.76831 ± 0.00967**; OOF 1.76817; mean bias −0.0442.
- S1-BEST benchmark: 1.75886; S2 хуже на 0.00945, но проходит порог разрыва 0.02.
- Blend: residual corr 0.9937, переносимый вес 0–10%, gain ≈0.0001 — REJECT.
- Calendar diagnostic: 1.96787 (`alpha=0`) → 1.95729 (`alpha=0.5`);
  принят предписанный консервативный `alpha=0.5`.
- Error analysis: S2 лучше S1 на `y=0` на −0.0254, хуже на `y>0` на +0.0353.
- **LB (public): 1.6619324597771563**, хуже S1-BEST (1.6512803) на **+0.01065**;
  лучше альтернативных сабмитов Strategy 1 — EXP-MIN 1.6674246 (−0.00549) и
  EXP-SIM 1.6682180 (−0.00629), обе карточки на ветке `team-a-strategy-1-impl`.

## Вердикт и вывод

Принят как **standalone S2-BEST**. Основной выигрыш внутри Strategy 2 дал hurdle;
value-компонент на покупающих остаётся причиной проигрыша S1 и отказа от бленда.

Замер на LB подтвердил локальный вывод: разрыв к S1-BEST на leaderboard (+0.01065)
воспроизводит разрыв по CV (+0.00945), уровень у обоих сабмитов одинаков (2.3293),
поэтому разница относится к модели, а не к калибровке. Порог продолжения 0.02 не
превышен, и S2 остаётся лучшей из проверенных альтернатив основному пайплайну.

## Конфиг прогона

29 cutoff'ов 2025-04-03..2025-10-16, шаг 7; train panel b1, val/test b3;
LightGBM 600 rounds, offset-Poisson + binary hurdle; EB `K=5`; финальная train-only
калибровка `sigma=0.9, mu_shift=-0.1`; calendar `alpha=0.5`; level 2.3293;
seed из `src/config.py`.

Сабмит: `submissions/submission_strategy_2.csv`. Проверки: 250k строк,
порядок sample совпадает, NaN/inf/negative 0, mean log1p 2.3293.
