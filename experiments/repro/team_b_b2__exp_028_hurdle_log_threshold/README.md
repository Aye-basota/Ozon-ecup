# exp_028 — G6 log-space hurdle with threshold

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_028_hurdle_log_threshold`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_028_hurdle_log_threshold`
- **Original source:** `git:88dc69163b1f:experiments/exp_028_hurdle_log_threshold.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам с единым порогом `0.08`: fold1 `1.689540`, fold2 `1.742963`, fold3 `1.732212`
- **Known score:** CV mean: `1.716251` (baseline exp_001 `1.717017`, локальный exp_015 `1.708737`)
- **Seed:** LightGBM classifier + LightGBM positive-regressor, baseline 50 features, folds 1–2 для решения, fold3 справочно, threshold grid `0.00..0.80` step `0.01`, seed из `src/config.py`.
- **Postprocessing:** # exp_028 — G6 log-space hurdle with threshold
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_028 — G6 log-space hurdle with threshold

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код не менялся)

## Гипотеза

Проверить исправленную двухстадийную схему: classifier даёт `p = P(y>0)`, positive-regressor даёт `m_log = log1p(y)` для покупателей, итог считается в log-пространстве `log_pred = p * m_log`, затем `pred = expm1(log_pred).clip(0)`. Дополнительно подобрать порог `p`: если вероятность покупки ниже порога, прогноз зануляется, чтобы не перепрогнозировать настоящие нули.

## Что изменено относительно базы

Код пайплайна не менялся; диагностический прогон использовал 50 baseline-признаков, чтобы не смешивать результат с незавершёнными lag-признаками exp_027.

## Результат

- CV по фолдам с единым порогом `0.08`: fold1 `1.689540`, fold2 `1.742963`, fold3 `1.732212`
- CV mean: `1.716251` (baseline exp_001 `1.717017`, локальный exp_015 `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение к baseline всего `0.000766`, ниже порога `0.005`, а относительно локального exp_015 результат хуже. Порог почти не влияет: без порога mean `1.716259`, с лучшим общим порогом `1.716251`; classifier почти не даёт очень маленьких вероятностей, поэтому зануляется слишком мало пользователей.

## Конфиг прогона

LightGBM classifier + LightGBM positive-regressor, baseline 50 features, folds 1–2 для решения, fold3 справочно, threshold grid `0.00..0.80` step `0.01`, seed из `src/config.py`.
