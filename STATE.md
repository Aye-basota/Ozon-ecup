# STATE — читать целиком перед каждой задачей (лимит ~80 строк)

## Текущий лучший

- Лучший LB: exp_019 behavior_v1 dist-head, LB 1.6545023535300867.
- Лучший локальный CV-кандидат: exp_024 CatBoost+XGBoost blend, mean RMSLE 1.706955; LB pending.

## Метрика и валидация

Метрика: RMSLE с log1p, отрицательные предсказания зануляются. Основная validation для scale/model selection: два single-cutoff фолда из main `2025-12-15 -> 2026-01-14` и `2025-11-15 -> 2025-12-15`.

## Не повторять

(провалившиеся гипотезы; список только растёт, ничего не удалять)
- exp_003: isolated trend-фичи recent/previous и short/long для active_days/searches/cart/orders/gmv ухудшили RMSLE 1.711195 → 1.711856.
- exp_003 ранний stacked-прогон trends поверх conversions тоже был хуже exp_002: 1.710919 → 1.711416.
- exp_013: fine scale grid выше `1.20` для exp_011 dense8 не улучшил LB: `1.275` → `1.655708`, `1.300` → `1.656280` против `1.20` → `1.654910`.
- exp_014: hard classifier gate для зануления ухудшил CV: threshold `0.10` mean RMSLE `1.709356` против базы `1.709007`, большие пороги ещё хуже.
- exp_015: fine weight grid вокруг `w_rec=0.5` не улучшил LB; `w_rec=0.525` был чуть хуже exp_011 `w_rec=0.5`.

## Последние эксперименты (макс 10 строк, переполнение → HISTORY.md)

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_024 | 2026-08-26 | Codex | CatBoost behavior_v1 как 5-й компонент поверх XGBoost blend | CV 1.707119 → 1.706955; submit ready | Ждёт LB |
| exp_023 | 2026-08-26 | Codex | XGBoost behavior_v1 как 4-й компонент ансамбля | CV 1.707699 → 1.707119; submit ready | Ждёт LB |
| exp_022 | 2026-08-26 | Codex | Model-level weight grid: recency + post_order_dist + behavior_dist | CV 1.707699 → 1.707575; submit ready | Ждёт LB |
| exp_021 | 2026-08-26 | Codex | Submission-level log-space blends вокруг exp019 | CV не считали; 7 submit-кандидатов | Ждёт LB |
| exp_020 | 2026-08-17 | Codex | behavior_v1_slim: top-50 новых b1_ вместо 114 | CV 1.707699 → 1.707796; LB 1.654736 | Хуже exp017 на LB |
| exp_019 | 2026-08-17 | Codex | behavior_v1: 114 поведенческих фичей в dist-head | CV 1.708295 → 1.707699; LB 1.654502 | Лучший LB |
| exp_018 | 2026-08-16 | Codex | CatBoost как третий компонент поверх recency + post-order dist-head | CV 1.708295 → 1.707921; submit ready | Локально лучший, LB pending |
| exp_017 | 2026-08-15 | Codex | Dist-head из team-a на post-order фичах | CV 1.708883 → 1.708295; LB 1.654632 | Лучший LB |
| exp_016 | 2026-08-15 | Codex | Post-order activity фичи: активен после последней покупки или ушёл | CV 1.709007 → 1.708883; LB 1.654779 | Лучший LB |
| exp_015 | 2026-08-14 | Codex | Grid весов recency/long_buy при scale 1.20 | w_rec 0.525 CV 1.709004, но LB чуть хуже exp_011 | Нейтрально/провал |

## Backlog

- [x] RFM/recency-сигналы за 7/14/30/60 дней
- [x] Конверсии воронки search→cart→buy
- [x] Тренд активности: последние 7 дней / предыдущие 7
- [x] log1p таргета
- [x] Взвешенный ансамбль
- [x] LightGBM baseline
