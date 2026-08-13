# STATE — читать целиком перед каждой задачей (лимит ~80 строк)

## Текущий лучший

- Лучший LB: exp_009 long-buy log-ensemble, LB 1.6568530995317488.
- Лучший локальный CV-кандидат: exp_009 long-buy log-ensemble, RMSLE 1.670716.

## Метрика и валидация

Метрика: RMSLE с log1p, отрицательные предсказания зануляются. Валидация: out-of-time; основной single-cutoff использует train 2025-10-01/2025-11-01/2025-12-01 и val 2026-01-15.

## Не повторять

(провалившиеся гипотезы; список только растёт, ничего не удалять)
- exp_003: isolated trend-фичи recent/previous и short/long для active_days/searches/cart/orders/gmv ухудшили RMSLE 1.711195 → 1.711856.
- exp_003 ранний stacked-прогон trends поверх conversions тоже был хуже exp_002: 1.710919 → 1.711416.

## Последние эксперименты (макс 10 строк, переполнение → HISTORY.md)

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_011 | 2026-08-13 | Codex | Dense weekly clean-cutoff ensemble | clean val best RMSLE 1.712473 @ scale 1.4 | LB-кандидаты |
| exp_010 | 2026-08-12 | Codex | Тюнинг веса/scale для exp_009 ensemble | best near RMSLE 1.670716 | LB-кандидаты |
| exp_009 | 2026-08-12 | Codex | Long-buy фичи из importance + log-space ensemble | RMSLE 1.670716; LB 1.656853 | Лучший LB |
| exp_008 | 2026-08-11 | Codex | Champion: recency + LightGBM + scale 0.64 | RMSE 254.269612; RMSLE 1.671639; LB 1.657 | Лучший LB |
| exp_007 | 2026-08-11 | Codex | Multi-cutoff CV для baseline | mean RMSLE 1.729343; std 0.015957 | Нейтрально: validation audit |
| exp_006 | 2026-08-11 | Codex | LightGBM вместо HGBR на baseline | RMSE 243.895440; RMSLE 1.710143 | Успех |
| exp_005 | 2026-08-11 | Codex | Scale calibration baseline-прогноза | scale 0.65; RMSLE 1.672748 | Сильный успех |
| exp_004 | 2026-08-11 | Codex | Recency по последним действиям | RMSE 244.749083; RMSLE 1.710617 | Успех |
| exp_003 | 2026-08-11 | Codex | Isolated тренды активности | RMSE 244.595270; RMSLE 1.711856 | Провал |
| exp_002 | 2026-08-11 | Codex | Конверсии search→cart/order, cart→order, средние чеки, доли GMV | RMSE 243.974514; RMSLE 1.710919 | Успех: маленькое улучшение |

## Backlog

- [x] RFM/recency-сигналы за 7/14/30/60 дней
- [x] Конверсии воронки search→cart→buy
- [x] Тренд активности: последние 7 дней / предыдущие 7
- [x] log1p таргета
- [x] Взвешенный ансамбль
- [x] LightGBM baseline
