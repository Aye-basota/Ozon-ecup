# exp_008 — G7 huber log target loss

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Заменить обычную regression objective по `log1p(y)` на Huber objective по тому же `log1p(y)`. Ожидание: устойчивость к крупным покупателям может улучшить общий RMSLE.

## Что изменено относительно базы

Менялся только LightGBM objective: `huber`, `alpha=0.9`; таргет `log1p(y)` и прогноз `expm1` сохранялись. После reject код возвращён к `objective="regression"`.

## Результат

- CV по фолдам: fold1 `1.703652`, fold2 `1.761199`, fold3 `1.758791`
- CV mean: `1.732425` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: оба основных фолда хуже baseline. Huber немного снижает общий bias на fold1, но ухудшает RMSLE и сегменты покупателей.

## Конфиг прогона

LightGBM Huber, `alpha=0.9`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
