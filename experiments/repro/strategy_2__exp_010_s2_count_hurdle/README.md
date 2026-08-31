# exp_010 — S2-E01: распределение числа покупательных дней

## Catalogue metadata

- **Catalogue ID:** `strategy_2__exp_010_s2_count_hurdle`
- **Namespace:** `strategy_2`
- **Experiment ID:** `exp_010_s2_count_hurdle`
- **Original source:** `git:3c1d86d836c7:experiments/exp_010_s2_count_hurdle.md`
- **Source ref:** `3c1d86d836c7b73519abe99f94686431852187cc`
- **Source commit:** `2e1d89d7904ee161939d9c6eed44fe16d4e4c549`
- **Kind:** git-history experiment card
- **Model:** LightGBM, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** LightGBM 600 rounds, 24 train-cutoff'а, 1-block train / 3-block validation,
- **Known score:** Offset практически нейтрален. Hurdle принят: он дал −0.00830 RMSLE и устранил
- **Seed:** seed из `src/config.py`. Команды: `count-screen`, затем `cv --hurdle --ks 3`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_010 — S2-E01: распределение числа покупательных дней

- **Дата:** 2026-08-11
- **Автор:** A2
- **Коммит:** working tree, `team-a-strategy-2-impl`

## Гипотеза

Poisson offset улучшит count-компонент, а отдельный классификатор `P(n>0)` нужен
только при расхождении нулевой массы больше 2 п.п.

## Что изменено относительно базы

Сначала plain Poisson сравнен с offset Poisson; затем добавлена hurdle-агрегация.

## Результат

- Plain/offset deviance: 1.46310 / 1.46290; raw structural 1.79165 / 1.79168.
- Poisson: `P0_pred=0.3182`, `P0_true=0.3871`, ошибка **−6.90 п.п.**
- Hurdle: ошибка P0 **−0.81 п.п.**
- На `2025-10-16`, train-only calibration: Poisson **1.76579**, hurdle **1.75749**.
- LB: не отправляли.

## Вердикт и вывод

Offset практически нейтрален. Hurdle принят: он дал −0.00830 RMSLE и устранил
основную часть ошибки нулевой массы.

## Конфиг прогона

LightGBM 600 rounds, 24 train-cutoff'а, 1-block train / 3-block validation,
seed из `src/config.py`. Команды: `count-screen`, затем `cv --hurdle --ks 3`.
