# HISTORY — архив экспериментов

Сюда переносятся старые строки таблицы из STATE.md, когда их становится больше 10.
Новые сверху. Провалившиеся гипотезы при архивации обязаны остаться
строкой в «Не повторять» в STATE.md.

| ID | Дата | Автор | Гипотеза | CV | Вердикт |
|----|------|-------|----------|-----|---------|
| exp_021 | 2026-08-13 | Codex | G15 lr0.03 leaves31 minleaf50 | 1.708309 | reject, код откатан |
| exp_020 | 2026-08-13 | Codex | G15 learning_rate=0.03 | 1.708634 | reject, код откатан |
| exp_019 | 2026-08-13 | Codex | G15 num_leaves=127 | 1.709231 | reject, код откатан |
| exp_018 | 2026-08-13 | Codex | G15 min_data_in_leaf=300 | 1.708783 | reject, код откатан |
| exp_017 | 2026-08-13 | Codex | G15 min_data_in_leaf=50 | 1.708632 | reject, код откатан |
| exp_016 | 2026-08-13 | Codex | G15 num_leaves=31 | 1.708734 | reject, код откатан |
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
| exp_005 | 2026-08-13 | Codex | G4 multi-cutoff 8x14d | 1.717040 | reject, код откатан |
| exp_004 | 2026-08-13 | Codex | G3 EWM-агрегаты | 1.716725 | reject, код откатан |
| exp_003 | 2026-08-13 | Codex | G2 календарные признаки cutoff | 1.717017 | reject, код откатан |
| exp_002 | 2026-08-13 | Codex | G1 SPLY same-period year-ago | 1.717011 | reject, код откатан |
| exp_001 | 2026-08-12 | Codex | LightGBM на 50 user-based признаках | 1.717017 | baseline accept, LB 1.6615 |
