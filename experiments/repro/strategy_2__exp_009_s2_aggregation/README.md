# exp_009 — S2-E00: корректность структурной агрегации

## Catalogue metadata

- **Catalogue ID:** `strategy_2__exp_009_s2_aggregation`
- **Namespace:** `strategy_2`
- **Experiment ID:** `exp_009_s2_aggregation`
- **Original source:** `git:3c1d86d836c7:experiments/exp_009_s2_aggregation.md`
- **Source ref:** `3c1d86d836c7b73519abe99f94686431852187cc`
- **Source commit:** `2e1d89d7904ee161939d9c6eed44fe16d4e4c549`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** `python src/strategy_2.py aggregation --samples 10000`; seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_009 — S2-E00: корректность структурной агрегации

- **Дата:** 2026-08-11
- **Автор:** A2
- **Коммит:** working tree, `team-a-strategy-2-impl`

## Гипотеза

11-узловая квадратура и Fenton–Wilkinson воспроизводят `E[log1p(S)]` с ошибкой
не более 0.01 для `n∈{2,3,4}`, `sigma∈{0.8,1.1,1.4}`.

## Что изменено относительно базы

После провала чистого FW добавлен Sobol QMC lookup для `n≤4`; FW оставлен для `n≥5`.

## Результат

- Чистый FW: max abs error **0.10027**.
- Hybrid против независимого MC-200k: max abs error **0.00549**.
- QN=11 против QN=21: max difference `1.29e-11`.
- 1000 пользователей × 10k MC: mean abs error hybrid 0.00605 (у FW 0.02757).
- LB: не отправляли.

## Вердикт и вывод

Исходная гипотеза о FW опровергнута. Hybrid QMC/FW проходит обязательный контроль и
используется во всех дальнейших экспериментах.

## Конфиг прогона

`python src/strategy_2.py aggregation --samples 10000`; seed из `src/config.py`.
