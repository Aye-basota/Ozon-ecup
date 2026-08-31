# exp_029 — variance predictability for hurdle

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_029_variance_predictability_hurdle`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_029_variance_predictability_hurdle`
- **Original source:** `git:88dc69163b1f:experiments/exp_029_variance_predictability_hurdle.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM, two-part / hurdle
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам с лучшим общим порогом `0.06`: fold1 `1.689357`, fold2 `1.742964`, fold3 `1.732228`
- **Known score:** CV mean: `1.716161` (exp_028 `1.716251`, baseline exp_001 `1.717017`, локальный exp_015 `1.708737`)
- **Seed:** LightGBM classifier + LightGBM positive-regressor, baseline 50 features + `var_predictability`, folds 1–2 для решения, fold3 справочно, threshold grid `0.00..0.80` step `0.01`, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_029 — variance predictability for hurdle

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject по метрике; признак оставлен в `src/features.py` по запросу пользователя)

## Гипотеза

Добавить один признак предсказуемости пользователя: сумма относительной вариативности gmv, вариативности интервалов покупок и вариативности активности. Идея: если пользователь стабилен, classifier `P(y>0)` должен быть полезнее; если поведение шумное, модель должна сама научиться меньше доверять жёсткому разделению.

## Что изменено относительно базы

В `src/features.py` добавлен новый признак `var_predictability`, который считается из уже построенных признаков `w365_lgmv_std`, `w365_lgmv_mean`, `buygap_cv`, `gap_cv`. Диагностический CV считался на baseline 50 features + `var_predictability`.

## Результат

- CV по фолдам с лучшим общим порогом `0.06`: fold1 `1.689357`, fold2 `1.742964`, fold3 `1.732228`
- CV mean: `1.716161` (exp_028 `1.716251`, baseline exp_001 `1.717017`, локальный exp_015 `1.708737`)
- LB: `E_hurdle` = `1.65841662470559`

## Вердикт и вывод

Локально reject по автоматическому правилу: улучшение к exp_028 всего `0.000091`, к baseline `0.000856`; это ниже порога `0.005` и хуже локального exp_015. Но LB оказался лучше baseline (`1.65841662470559` против `1.6615`), поэтому признак и файл оставлены как полезный LB-кандидат.

## Конфиг прогона

LightGBM classifier + LightGBM positive-regressor, baseline 50 features + `var_predictability`, folds 1–2 для решения, fold3 справочно, threshold grid `0.00..0.80` step `0.01`, seed из `src/config.py`.
