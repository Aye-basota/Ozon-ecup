# exp_033 — submit-кандидат: multi-cutoff с весенними cutoff'ами 2025

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_033_spring_submit`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_033_spring_submit`
- **Original source:** `git:88dc69163b1f:experiments/exp_033_spring_submit.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Тест-окно 2026-02-14..03-15 — «весеннее» (y=0 ≈ 60% против 44–46% на зимних фолдах). Ни один фолд не имитирует тест. Единственный способ дать модели весенние таргеты — обучающие cutoff'ы февраля–марта 2025.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_033 — submit-кандидат: multi-cutoff с весенними cutoff'ами 2025

- **Дата:** 2026-08-13
- **Автор:** Kimi
- **Коммит:** нет (LB-проба решается человеком; скрипт `artifacts/audit/exp_033_spring_submit.py`)

## Гипотеза

Тест-окно 2026-02-14..03-15 — «весеннее» (y=0 ≈ 60% против 44–46% на зимних фолдах). Ни один фолд не имитирует тест. Единственный способ дать модели весенние таргеты — обучающие cutoff'ы февраля–марта 2025.

## Конфиг

8 cutoff'ов: весенние `2025-02-14, 03-01, 03-15, 03-29` (построены заново, короткая история) + свежие `2025-11-17, 12-01, 12-15, 2026-01-14` (кэш). 50 признаков HEAD, δ=0, n_estimators=204. Файл: `submissions/H_multicutoff_spring_exp033.csv`.

## Результат

- Локальная валидация НЕВОЗМОЖНА (нет весеннего val-окна) — чистый LB-кандидат.
- Доли y=0 обучающих окон: весенние 59.4–59.9%, зимние 41.9–46.0% — подтверждает мотивацию.
- pred_mean=35.64, топ-признак по gain: `w365_buyday_rate`.

## Вердикт

**LB-проба состоялась: `1.6599`** (baseline δ=0 = `1.6615`, заявленный лучший G_hurdle = `1.6562`). Весеннее обучение дало +0.0016 к baseline на бою — гипотеза подтверждена слабо. Направление развивается в PLAN.md (программа сезонного переноса).

Риск: признаки весенних cutoff'ов посчитаны по 1.5–2.5 месяцам истории (короткий tenure), распределение признаков отличается от теста — эффект непредсказуем.
