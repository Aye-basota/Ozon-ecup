# STATE — читать целиком перед каждой задачей (лимит ~80 строк)
## Текущий лучший
- Best LB: `G_hurdle_variance_exp029_season_1_12.csv`, LB `1.6561975700155196`.
- Другие hurdle LB: `F_hurdle` `1.6566618522758063`, `E_hurdle` `1.65841662470559`; baseline exp_001 LB `1.6615`, exp_015 LB `1.6700`.
## Метрика и валидация
Метрика: RMSLE по `log1p`, отрицательные предсказания клипаются в 0. Валидация: out-of-time, фолды 1–2 из `PLAN.md`, fold3 только справочно. Глобальные калибровочные константы подбирать per-fold; в сабмит брать только если знак и порядок подтверждены на fold2, иначе сабмит без калибровки.
## Не повторять
(провалившиеся гипотезы; список только растёт, ничего не удалять)
- exp_015 global calibration submit: local +`0.008` → LB −`0.009`; причина — нетрансферабельность глобального `δ` через сезонность, правило обновлено.
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
- exp_016 G15 num_leaves=31: mean `1.708734`, улучшение `0.000003`; ниже порога.
- exp_017 G15 min_data_in_leaf=50: mean `1.708632`, улучшение `0.000105`; ниже порога.
- exp_018 G15 min_data_in_leaf=300: mean `1.708783`, хуже exp_015.
- exp_019 G15 num_leaves=127: mean `1.709231`, хуже exp_015.
- exp_020 G15 learning_rate=0.03: mean `1.708634`, улучшение `0.000103`; ниже порога.
- exp_021 G15 combo lr0.03/leaves31/minleaf50: mean `1.708309`, улучшение `0.000428`; ниже порога.
- exp_022 G6 classifier-prob feature: mean `1.711550`, хуже exp_015.
- exp_023 CatBoost/LGBM blend diagnostic: mean `1.707779`, улучшение `0.000959`; ниже порога.
- exp_024 post-G14 diagnostics: best mean `1.708145`, улучшение `0.000592`; ниже порога.
- exp_025 funnel features: mean `1.708564`, улучшение `0.000173`; ниже порога.
- exp_032 delta-modeling (поправка к naive30): mean `1.720406`, хуже baseline на всех фолдах; reject.
- exp_031 G4 recheck multi-cutoff 8x14d δ=0: mean `1.717040`, хуже baseline; подтверждён reject exp_005, сезонный сдвиг не лечит.
- exp_026 activity trend 7/prev7: mean `1.708532`, улучшение `0.000205`; ниже порога.
- exp_028 log-space hurdle + threshold: mean `1.716251`, улучшение к baseline `0.000766`, хуже exp_015; порог почти не влияет.
- exp_029 variance predictability hurdle: mean `1.716161`, file `E_hurdle_variance_exp029.csv`, LB `1.65841662470559`.
- exp_030 season coefficients `1.1678` and `1.12`: `F_hurdle` LB `1.6566618522758063`, `G_hurdle` LB `1.6561975700155196` (best).
- exp_034 A1 temporal ensemble 3 neighboring cutoffs: mean(1,2) `1.721106`, хуже central same-run на `0.004264`; fold1 деградация `0.009866`, reject.
## Последние эксперименты (макс 10 строк, переполнение → HISTORY.md)
| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_034 | 2026-08-13 | Codex | A1 temporal ensemble 3 neighboring cutoffs | 1.721106 | reject, fold1 деградация |
| exp_030 | 2026-08-13 | Codex | target-window YoY season coefficient | NA | LB best `1.6561975700155196` |
| exp_029 | 2026-08-13 | Codex | variance predictability for hurdle | 1.716161 | LB `1.65841662470559` |
| exp_028 | 2026-08-13 | Codex | G6 log-space hurdle + threshold | 1.716251 | reject, код не менялся |
| exp_031 | 2026-08-13 | Kimi | G4 recheck multi-cutoff 8x14d δ=0 | 1.717040 | reject, подтверждён exp_005 |
| exp_026 | 2026-08-13 | Codex | backlog activity trend last7/prev7 | 1.708532 | reject, код откатан |
| exp_025 | 2026-08-13 | Codex | backlog funnel conversion features | 1.708564 | reject, код откатан |
| exp_024 | 2026-08-13 | Codex | post-G14 diagnostics | 1.708145 | reject, код не менялся |
| exp_023 | 2026-08-13 | Codex | G13 CatBoost+LGBM blend diagnostic | 1.707779 | reject, код не менялся |
| exp_022 | 2026-08-13 | Codex | G6 classifier-prob feature | 1.711550 | reject, код откатан |
## Backlog
- [x] RFM-агрегаты за 7/14/30/60 дней
- [x] G1 SPLY-признаки — reject
- [x] G2 календарные признаки от cutoff — reject в single-cutoff
- [x] G3 EWM-агрегаты — reject
- [x] G4 multi-cutoff обучение 8x14d — reject
- [x] G6 hurdle / классификатор P(y>0): старый amount-space, prob-feature и log-space+threshold — reject
- [x] G7 tweedie loss — reject
- [x] G7 huber loss — reject
- [x] G8 search/catalog split features — reject
- [x] G9 AOV and intensity features — reject
- [x] G10 simple trend ratios — reject
- [x] G11 clipped target — reject
- [x] G12 seed bagging — reject
- [x] G13 second LGBM/blend — reject
- [x] G14 calibration shift — accept delta -0.17
- [x] G15 light tuning — 6 points done, no accept
- [x] post-G14 segmented calibration/blends — no accept
- [x] Конверсии воронки search→cart→buy — reject
- [x] Тренд активности: последние 7 дней / предыдущие 7 — reject
- [x] log1p таргета
- [ ] Взвешенный ансамбль
