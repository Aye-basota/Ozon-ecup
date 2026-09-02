# exp_015 — ensemble weight grid

- **Дата:** 2026-08-14
- **Автор:** B1
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
