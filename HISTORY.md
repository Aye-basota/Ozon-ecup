# HISTORY — архив экспериментов

Сюда переносятся старые строки таблицы из STATE.md, когда их становится больше 10.
Новые сверху. Провалившиеся гипотезы при архивации обязаны остаться
строкой в «Не повторять» в STATE.md.

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_014 | 2026-08-14 | B1 | Classifier gate зануляет low-proba покупателей | best same as no gate; thr 0.10 CV 1.709356 > base 1.709007 | Провал |
| exp_013 | 2026-08-13 | B1 | Scale grid 1.10..1.35 на validation из main | CV best scale 1.30, but LB worse: 1.275=1.655708, 1.300=1.656280 | Провал |
| exp_012 | 2026-08-13 | B1 | Проверка validation из main против LB на exp_011 scales | scale 1.2 best: CV 1.709007; LB 1.654910 | Успех: rank совпал |
| exp_011 | 2026-08-13 | B1 | Dense weekly clean-cutoff ensemble | clean val best scale 1.4, LB best scale 1.2 = 1.654910 | Лучший LB |
| exp_010 | 2026-08-12 | B1 | Тюнинг веса/scale для exp_009 ensemble | best near RMSLE 1.670716 | LB-кандидаты |
| exp_009 | 2026-08-12 | B1 | Long-buy фичи из importance + log-space ensemble | RMSLE 1.670716; LB 1.656853 | Лучший LB |
| exp_008 | 2026-08-11 | B1 | Champion: recency + LightGBM + scale 0.64 | RMSE 254.269612; RMSLE 1.671639; LB 1.657 | Лучший LB |
| exp_007 | 2026-08-11 | B1 | Multi-cutoff CV для baseline | mean RMSLE 1.729343; std 0.015957 | Нейтрально: validation audit |
| exp_006 | 2026-08-11 | B1 | LightGBM вместо HGBR на baseline | RMSE 243.895440; RMSLE 1.710143 | Успех |
| exp_005 | 2026-08-11 | B1 | Scale calibration baseline-прогноза | scale 0.65; RMSLE 1.672748 | Сильный успех |
| exp_004 | 2026-08-11 | B1 | Recency по последним действиям | RMSE 244.749083; RMSLE 1.710617 | Успех |
| exp_003 | 2026-08-11 | B1 | Isolated тренды активности | RMSE 244.595270; RMSLE 1.711856 | Провал |
| exp_002 | 2026-08-11 | B1 | Конверсии search→cart/order, cart→order, средние чеки, доли GMV | RMSE 243.974514; RMSLE 1.710919 | Успех: маленькое улучшение |
| exp_001 | 2026-08-11 | B1 | Baseline на оконных агрегатах активности + log1p target | RMSE 244.596747; RMSLE 1.711195 | Нейтрально: рабочая точка отсчёта |
