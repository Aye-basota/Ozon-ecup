# STATE — читать целиком перед каждой задачей (лимит ~80 строк)

## Текущий лучший

- exp_015: baseline LightGBM + fixed log shift `-0.17`, CV mean RMSLE `1.708737` (fold1 `1.675509`, fold2 `1.741966`, fold3 `1.745607`), best LB пока `1.6615` из exp_001/PLAN.md.

## Метрика и валидация

Метрика: RMSLE по `log1p`, отрицательные предсказания клипаются в 0. Валидация: out-of-time, фолды 1–2 из `PLAN.md`, fold3 только справочно.

## Не повторять

(провалившиеся гипотезы; список только растёт, ничего не удалять)
- exp_002 G1 SPLY: mean `1.717011`, улучшение `0.000006`; ниже порога, структуру ошибки почти не меняет.
- exp_003 G2 calendar в single-cutoff: mean `1.717017`, улучшения нет; признаки константны внутри train.
- exp_004 G3 EWM: mean `1.716725`, улучшение `0.000292`; ниже порога, fold2 чуть хуже.
- exp_005 G4 multi-cutoff 8x14d: mean `1.717040`, хуже baseline; усиливает ошибку на y=0.
- exp_006 G6 hurdle: mean `2.045762`, сильная деградация; перепрогноз нулевых y.
- exp_007 G7 tweedie: mean `2.391320`, сильная деградация; сырой y/loss перепрогнозирует нули.
- exp_008 G7 huber: mean `1.732425`, оба основных фолда хуже baseline.
- exp_009 G8 search/catalog split: mean `1.716802`, улучшение `0.000215`; ниже порога.
- exp_010 G9 AOV/intensity: mean `1.716961`, улучшение `0.000056`; ниже порога, fold2 чуть хуже.
- exp_011 G10 trends: mean `1.717095`, хуже baseline; fold1 деградирует.
- exp_012 G11 target clip p999: mean `1.716910`, улучшение `0.000107`; ниже порога, fold2 чуть хуже.
- exp_013 G12 seed bagging: mean `1.716366`, улучшение `0.000651`; ниже порога.
- exp_014 G13 second LGBM blend: mean `1.716513`, улучшение `0.000504`; ниже порога.

## Последние эксперименты (макс 10 строк, переполнение → HISTORY.md)

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_015 | 2026-08-13 | Codex | G14 fixed calibration delta -0.17 | 1.708737 | accept, fold3 risk |
| exp_014 | 2026-08-13 | Codex | G13 second LGBM blend | 1.716513 | reject, код откатан |
| exp_013 | 2026-08-13 | Codex | G12 seed bagging 5 seeds | 1.716366 | reject, код откатан |
| exp_012 | 2026-08-13 | Codex | G11 target clip p999 | 1.716910 | reject, код откатан |
| exp_011 | 2026-08-13 | Codex | G10 простые trend ratio | 1.717095 | reject, код откатан |
| exp_010 | 2026-08-13 | Codex | G9 AOV/intensity | 1.716961 | reject, код откатан |
| exp_009 | 2026-08-13 | Codex | G8 search/catalog split | 1.716802 | reject, код откатан |
| exp_008 | 2026-08-13 | Codex | G7 huber log target | 1.732425 | reject, код откатан |
| exp_007 | 2026-08-13 | Codex | G7 tweedie raw y | 2.391320 | reject, код откатан |
| exp_006 | 2026-08-13 | Codex | G6 hurdle classifier+positive regressor | 2.045762 | reject, код откатан |

## Backlog

- [x] RFM-агрегаты за 7/14/30/60 дней
- [x] G1 SPLY-признаки — reject
- [x] G2 календарные признаки от cutoff — reject в single-cutoff
- [x] G3 EWM-агрегаты — reject
- [x] G4 multi-cutoff обучение 8x14d — reject
- [x] G6 hurdle / классификатор P(y>0) — reject
- [x] G7 tweedie loss — reject
- [x] G7 huber loss — reject
- [x] G8 search/catalog split features — reject
- [x] G9 AOV and intensity features — reject
- [x] G10 simple trend ratios — reject
- [x] G11 clipped target — reject
- [x] G12 seed bagging — reject
- [x] G13 second LGBM/blend — reject
- [x] G14 calibration shift — accept delta -0.17
- [ ] G15 light tuning
- [ ] Конверсии воронки search→cart→buy
- [ ] Тренд активности: последние 7 дней / предыдущие 7
- [x] log1p таргета
- [ ] Взвешенный ансамбль
