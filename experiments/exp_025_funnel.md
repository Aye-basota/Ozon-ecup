# exp_025 — backlog funnel conversion features

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить признаки воронки search→cart→order по окнам 30/90/all: conversion rates, cart→order rate, дни с successful search-to-cart/search-to-order.

## Что изменено относительно базы

Добавлялись только новые `funnel_*` признаки в конец `FEATURES` через `build_features`; после reject код возвращён к exp_015.

## Результат

- CV по фолдам: fold1 `1.675062`, fold2 `1.742066`, fold3 `1.745565`
- CV mean: `1.708564` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean `0.000173`, ниже порога. Фичи немного помогают fold1, но не дают достаточного сдвига к целевой зоне.

## Конфиг прогона

LightGBM exp_015, `CALIBRATION_DELTA=-0.17`, новые `funnel_*` признаки, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
