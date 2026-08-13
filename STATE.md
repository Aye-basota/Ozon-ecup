# STATE — читать целиком перед каждой задачей (лимит ~80 строк)

## Текущий лучший

- Лучший LB: exp_011 dense8 log-ensemble scale 1.2, LB 1.6549097093483665.
- Лучший локальный CV-кандидат для scale-тюнинга: exp_012/013 validation из main хорошо выбирает грубо, но мелкую сетку выше 1.20 переоценила; LB-чемпион остаётся scale 1.20.

## Метрика и валидация

Метрика: RMSLE с log1p, отрицательные предсказания зануляются. Основная validation для scale/model selection: два single-cutoff фолда из main `2025-12-15 -> 2026-01-14` и `2025-11-15 -> 2025-12-15`.

## Не повторять

(провалившиеся гипотезы; список только растёт, ничего не удалять)
- exp_003: isolated trend-фичи recent/previous и short/long для active_days/searches/cart/orders/gmv ухудшили RMSLE 1.711195 → 1.711856.
- exp_003 ранний stacked-прогон trends поверх conversions тоже был хуже exp_002: 1.710919 → 1.711416.
- exp_013: fine scale grid выше `1.20` для exp_011 dense8 не улучшил LB: `1.275` → `1.655708`, `1.300` → `1.656280` против `1.20` → `1.654910`.

## Последние эксперименты (макс 10 строк, переполнение → HISTORY.md)

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_013 | 2026-08-13 | Codex | Scale grid 1.10..1.35 на validation из main | CV best scale 1.30, but LB worse: 1.275=1.655708, 1.300=1.656280 | Провал |
| exp_012 | 2026-08-13 | Codex | Проверка validation из main против LB на exp_011 scales | scale 1.2 best: CV 1.709007; LB 1.654910 | Успех: rank совпал |
| exp_011 | 2026-08-13 | Codex | Dense weekly clean-cutoff ensemble | clean val best scale 1.4, LB best scale 1.2 = 1.654910 | Лучший LB |
| exp_010 | 2026-08-12 | Codex | Тюнинг веса/scale для exp_009 ensemble | best near RMSLE 1.670716 | LB-кандидаты |
| exp_009 | 2026-08-12 | Codex | Long-buy фичи из importance + log-space ensemble | RMSLE 1.670716; LB 1.656853 | Лучший LB |
| exp_008 | 2026-08-11 | Codex | Champion: recency + LightGBM + scale 0.64 | RMSE 254.269612; RMSLE 1.671639; LB 1.657 | Лучший LB |
| exp_007 | 2026-08-11 | Codex | Multi-cutoff CV для baseline | mean RMSLE 1.729343; std 0.015957 | Нейтрально: validation audit |
| exp_006 | 2026-08-11 | Codex | LightGBM вместо HGBR на baseline | RMSE 243.895440; RMSLE 1.710143 | Успех |
| exp_005 | 2026-08-11 | Codex | Scale calibration baseline-прогноза | scale 0.65; RMSLE 1.672748 | Сильный успех |
| exp_004 | 2026-08-11 | Codex | Recency по последним действиям | RMSE 244.749083; RMSLE 1.710617 | Успех |

## Backlog

- [x] RFM/recency-сигналы за 7/14/30/60 дней
- [x] Конверсии воронки search→cart→buy
- [x] Тренд активности: последние 7 дней / предыдущие 7
- [x] log1p таргета
- [x] Взвешенный ансамбль
- [x] LightGBM baseline
