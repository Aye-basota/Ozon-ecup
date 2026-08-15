# exp_024 — post-G14 extra diagnostics

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код не менялся)

## Гипотеза

После accepted calibration проверить дешёвые постобработки и разнородные blends: segmented calibration, naive blends, XGBoost blend и thresholding малых предсказаний.

## Что изменено относительно базы

Код не менялся; расчёты запускались inline на cached datasets.

## Результат

- `rec_buy` segmented calibration: fold1 `1.675442`, fold2 `1.741799`, mean `1.708621`
- `w365_gmv` decile calibration: fold1 `1.675252`, fold2 `1.741765`, mean `1.708509`
- naive30/naive90 log blends: best weight LGBM `1.0`, mean `1.708737`
- XGBoost blend: fold1 `1.674912`, fold2 `1.741379`, mean `1.708145`
- thresholding low predictions: best threshold `0`, mean `1.708737`
- LB: не отправляли

## Вердикт и вывод

Reject: лучшая extra diagnostic (`XGBoost` blend) улучшила exp_015 только на `0.000592`, ниже порога. Пост-G14 локальный прогресс упёрся в шумовые улучшения.

## Конфиг прогона

Accepted LightGBM exp_015; XGBoost `max_depth=6`, `learning_rate=0.05`, `n_estimators=2000`, hist tree method; folds 1–2, seed из `src/config.py`.
