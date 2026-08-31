# exp_015 — ensemble weight grid

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_015_ensemble_weight_grid`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_015_ensemble_weight_grid`
- **Original source:** `git:824f41575bc2:experiments/exp_015_ensemble_weight_grid.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `fbf1dc24eadaaa0d2a1e2bdc425270015b272123`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble, blend
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Добавлен `src/weight_grid.py`: компоненты `recency` и `long_buy` обучаются один раз на фолд, затем перебирается `recency_weight` в log-space при fixed `global_scale=1.20`.
- **Known score:** best `w_rec=0.5`: mean RMSLE `1.709007`
- **Seed:** Base components: LightGBM `recency` with component scale `0.64`, LightGBM `long_buy` with component scale `0.62`, log-space blend, global scale `1.20`, seed из `config.py`. Submit-кандидат на потом: `submissions/exp_015_dense8_logens_wrec0525_scale120.csv`.
- **Postprocessing:** Base components: LightGBM `recency` with component scale `0.64`, LightGBM `long_buy` with component scale `0.62`, log-space blend, global scale `1.20`, seed из `config.py`. Submit-кандидат на потом: `submissions/exp_015_dense8_logens_wrec0525_scale120.csv`.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_015 — ensemble weight grid

- **Дата:** 2026-08-14
- **Автор:** Codex
- **Коммит:** 0366bd9

## Гипотеза

Текущий exp_011 использует log-space blend `recency/long_buy = 0.5/0.5`. Возможно, чуть больший вес одного компонента даст прирост без новых фич и без изменения scale.

## Что изменено относительно базы

Добавлен `src/weight_grid.py`: компоненты `recency` и `long_buy` обучаются один раз на фолд, затем перебирается `recency_weight` в log-space при fixed `global_scale=1.20`.

## Результат

- CV по фолдам: `2025-12-15 -> 2026-01-14`, `2025-11-15 -> 2025-12-15`.
- Грубая сетка `0.0..1.0`:
  - best `w_rec=0.5`: mean RMSLE `1.709007`
  - `w_rec=0.6`: mean RMSLE `1.709032`
  - `w_rec=0.4`: mean RMSLE `1.709077`
- Узкая сетка:
  - best `w_rec=0.525`: mean RMSLE `1.709004`
  - `w_rec=0.500`: mean RMSLE `1.709007`
  - `w_rec=0.550`: mean RMSLE `1.709008`
- LB: `w_rec=0.525` отправляли, результат чуть хуже exp_011 `w_rec=0.5, scale=1.20` (`1.6549097093483665`).

## Вердикт и вывод

Нейтрально / провал по LB. Веса вокруг `0.5` образуют почти плоское плато; формальный лучший `w_rec=0.525` улучшал CV всего на `0.000003`, но на LB оказался чуть хуже текущего чемпиона. Текущий `0.5/0.5` оставляем.

## Конфиг прогона

Base components: LightGBM `recency` with component scale `0.64`, LightGBM `long_buy` with component scale `0.62`, log-space blend, global scale `1.20`, seed из `config.py`. Submit-кандидат на потом: `submissions/exp_015_dense8_logens_wrec0525_scale120.csv`.
