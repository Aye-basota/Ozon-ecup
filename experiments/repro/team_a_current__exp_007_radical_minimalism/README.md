# exp_007 — radical feature minimalism

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_007_radical_minimalism`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_007_radical_minimalism`
- **Original source:** `experiments/exp_007_radical_minimalism.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, calibration diagnostic
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: 1.78521 / 1.77760 / 1.76795 / 1.76262.
- **Known score:** CV mean: **1.77335 ± 0.00870** (текущий лучший: S1-BEST, 1.75886 ± 0.01162).
- **Seed:** min leaf 800, lambda_l2 20, seed 42 из `config.py`, уровень test 2.3293.
- **Postprocessing:** min leaf 800, lambda_l2 20, seed 42 из `config.py`, уровень test 2.3293.
- **Submission:** Файл: `submissions/experimental_submission_1.csv`.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_007 — radical feature minimalism

- **Дата:** 2026-08-10
- **Автор:** A1
- **Коммит:** e3b1cde

## Гипотеза

Основной pipeline может переобучаться на слабые взаимодействия и признаки с
train/test shift. Маленькая модель на фиксированных окнах до 180 дней может дать
более устойчивое предсказание hidden test ценой небольшого падения локального CV.

## Что изменено относительно базы

Из 236 признаков оставлены 15 устойчивых колонок покупок, GMV, recency и активности;
LightGBM ограничен 31 листом и усиленной L2-регуляризацией.

## Результат

- CV по фолдам: 1.78521 / 1.77760 / 1.76795 / 1.76262.
- CV mean: **1.77335 ± 0.00870** (текущий лучший: S1-BEST, 1.75886 ± 0.01162).
- Калиброванный OOF: **1.77065**, хуже S1-BEST на **+0.01349**.
- Pearson / Spearman с current best на тесте: **0.97312 / 0.98518**;
  mean absolute prediction difference: **6.8803**.
- **LB: 1.667424590457357**, хуже S1-BEST (1.6512802628833827) на **+0.01614**.

Importance стабильна между фолдами: `w180_days_buy` даёт 40.1% gain при CV
importance 0.004; далее `w180_orders` (21.1%), `w90_days_buy` (17.5%) и
`w90_orders` (8.8%).

## Вердикт и вывод

**REJECT.** Гипотеза о hidden-test overfit полного pipeline опровергнута: уменьшение
с 236 до 15 признаков ухудшило LB на 0.01614. Локальный OOF заранее показал почти тот
же проигрыш (+0.01349), поэтому меньший fold std и стабильная importance не являются
достаточным основанием удалять слабые признаки. Точная калибровка уровня 2.3293
исключает level shift как объяснение провала.

## Конфиг прогона

`python src/final_experiments.py`; cutoff'ы 2025-04-03..2025-10-16, шаг 7,
train panel b1, validation/test panel b3, LightGBM 400 раундов, leaves 31,
min leaf 800, lambda_l2 20, seed 42 из `config.py`, уровень test 2.3293.

Файл: `submissions/experimental_submission_1.csv`.
