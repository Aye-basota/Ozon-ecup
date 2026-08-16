# exp_017 — dist head on post-order features

- **Дата:** 2026-08-15
- **Автор:** Codex
- **Коммит:** 18c3bdd

## Гипотеза

Переносим табличную технику из `origin/team-a`: вместо прямой регрессии `log1p(y)` обучаем multiclass distribution head по бинам `z = log1p(y)`, затем берём математическое ожидание по центроидам train-бинов. На данных с массой в нуле и тяжёлым хвостом это может оценивать `E[z|x]` устойчивее обычной L2-регрессии.

## Что изменено относительно базы

Добавлен `src/dist_head_ensemble.py`: компонент `long_buy_post_order` заменён на dist-head LightGBM с 16 бинами, а `recency` остаётся direct LightGBM. Смешивание остаётся log-space `0.5/0.5`, global scale `1.20`.

## Результат

- CV по фолдам: `2025-12-15 -> 2026-01-14`, `2025-11-15 -> 2025-12-15`.
- Baseline exp_016 `recency + long_buy_post_order direct`: mean RMSLE `1.708883`.
- exp_017 `recency + long_buy_post_order dist`:
  - `w_rec=0.4`: mean RMSLE `1.708324`
  - `w_rec=0.5`: mean RMSLE `1.708295`
  - `w_rec=0.6`: mean RMSLE `1.708394`
- По фолдам для `w_rec=0.5`:
  - fold 1: `1.672956 -> 1.672166`
  - fold 2: `1.744811 -> 1.744423`
- Submit-кандидат: `submissions/exp_017_dist_post_order_wrec050_scale120.csv`.
- LB: `1.6546318191`.

## Вердикт и вывод

Успех: новый лучший LB `1.6546318191`. Улучшение CV заметнее, чем у exp016, держится на обоих фолдах и перенеслось на leaderboard. Повышенный submit-уровень `mean_log1p(pred)=2.36643` не испортил результат.

## Конфиг прогона

Recency component: LightGBM direct, feature set `recency`, component scale `0.64`. Dist component: LightGBM multiclass, feature set `long_buy_post_order`, 16 bins, 250 rounds, learning rate `0.05`, component scale `0.62`. Log-space blend `w_rec=0.5`, global scale `1.20`, dense8 train cutoffs `2025-08-28..2025-10-16`, seed из `config.py`.
