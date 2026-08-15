# exp_023 — G13 CatBoost blend diagnostic

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код не менялся)

## Гипотеза

Проверить второй вариант G13: CatBoostRegressor на `log1p(y)` и blend с accepted calibrated LightGBM в log-пространстве. CatBoost может дать ошибки, отличающиеся от LightGBM.

## Что изменено относительно базы

Код не менялся; диагностический расчёт запускался inline на cached datasets. CatBoost уже есть в `requirements.txt` и `.venv`.

## Результат

- CV по фолдам для best blend: fold1 `1.674957`, fold2 `1.740600`, fold3 не прогоняли
- CV mean: `1.707779` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: improvement `0.000959`, ниже порога. CatBoost alone с delta `-0.16` дал mean `1.708933`; best blend weight LGBM `0.5`.

## Конфиг прогона

CatBoostRegressor `depth=6`, `learning_rate=0.05`, `iterations=2000`, early stopping 200; LightGBM exp_015; folds 1–2, seed из `src/config.py`.
