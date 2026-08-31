# exp_030 — submit seasonality coefficient diagnostic

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_030_submit_season_yoy_coefficient`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_030_submit_season_yoy_coefficient`
- **Original source:** `git:88dc69163b1f:experiments/exp_030_submit_season_yoy_coefficient.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** two-part / hurdle, calibration diagnostic
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Для боевого окна `2026-02-14..2026-03-15` нельзя использовать отрицательную калибровку, подобранную на январской валидации. Можно оценить сезонный множитель по прошлому году: отношение GMV окна `2025-02-14..2025-03-15` к GMV окна `2025-01-14..2025-02-12`, потому что submit-модель учится на label-окне `2026-01-14..2026-02-12`, а предсказывает `2026-02-14..2026-03-15`.
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
# exp_030 — submit seasonality coefficient diagnostic

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (diagnostic; код модели не менялся)

## Гипотеза

Для боевого окна `2026-02-14..2026-03-15` нельзя использовать отрицательную калибровку, подобранную на январской валидации. Можно оценить сезонный множитель по прошлому году: отношение GMV окна `2025-02-14..2025-03-15` к GMV окна `2025-01-14..2025-02-12`, потому что submit-модель учится на label-окне `2026-01-14..2026-02-12`, а предсказывает `2026-02-14..2026-03-15`.

## Что изменено относительно базы

Код обучения не менялся; подготовлены submit-кандидаты из exp_029 с мультипликативными сезонными коэффициентами `1.1678` и `1.12` в денежной шкале.

## Результат

- CV по фолдам: NA (коэффициент target-specific, локальные year-ago окна для фолдов неполные из-за старта данных 2025-01-01)
- CV mean: NA
- LB: `F_hurdle` = `1.6566618522758063`; `G_hurdle` = `1.6561975700155196`

## Вердикт и вывод

Diagnostic confirmed by LB: коэффициент `1.1678` рассчитан без лукапа, из доступного 2025 года, и улучшил `E_hurdle`; вариант `1.12` оказался ещё лучше. Текущий лучший LB: `G_hurdle_variance_exp029_season_1_12.csv` со score `1.6561975700155196`.

## Конфиг прогона

Base file: `E_hurdle_variance_exp029.csv`; season coefficient `GMV(2025-02-14..2025-03-15) / GMV(2025-01-14..2025-02-12) = 1.1678007837`; outputs `F_hurdle_variance_exp029_season_yoy_1_168.csv` and `G_hurdle_variance_exp029_season_1_12.csv`.
