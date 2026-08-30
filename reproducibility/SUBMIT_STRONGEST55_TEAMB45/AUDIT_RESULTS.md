# Результаты воспроизводимости

Дата проверки: 2026-08-30.

## Итог

- Exact rebuild `submission_STRONGEST_CURRENT.csv`: PASS, SHA-256
  `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.
- Exact rebuild `SUBMIT_STRONGEST55_TEAMB45.csv`: PASS, SHA-256
  `1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4`.
- Полный retrain `team-b-final`: PASS, production CSV совпал байт-в-байт.
- Полный retrain трёх табличных STRONGEST-компонентов: выполнен, но не
  побитово идентичен historical frozen-векторам.
- Ограниченный SEQ/ETX smoke: PASS, 114 tests passed, 1 skipped.

## Табличные модели

Raw data совпал по SHA-256. Фактическое время:

- STRONGEST tabular: 2 470.21 с;
- team-b-final: 1 921.86 с (из них training 1 809.63 с);
- весь tabular audit: 4 395.15 с.

Сверка fresh STRONGEST-компонентов с production в log-space:

| Компонент | RMS | Mean abs | Max abs | Corr |
|---|---:|---:|---:|---:|
| S1-UNC | 0.104034 | 0.073509 | 0.942314 | 0.997789 |
| S1-CAP | 0.080880 | 0.055433 | 0.969962 | 0.998694 |
| S1-DIST | 0.062264 | 0.044532 | 0.733360 | 0.999191 |

После исходных весов и level calibration fresh tabular + frozen DL дали для
`STRONGEST_CURRENT` RMS `0.027246`, corr `0.999850`. Финальный retrained submit
дал RMS `0.014985`, corr `0.999956` относительно target CSV.

Причину исторического расхождения нельзя доказательно локализовать по
сохранившимся данным: production booster weights и exact historical feature
matrices отсутствуют. Поэтому exact rebuild использует checksummed production
векторы, а fresh retraining хранится как отдельный проверочный путь.

Ветка `team-b-final` в точном окружении (`numpy 2.4.6`, `pandas 3.0.5`,
`duckdb 1.5.5`, `lightgbm 4.7.0`, `catboost 1.2.10`, `xgboost 3.2.0`)
воспроизвела `final_classic_ml.csv` байт-в-байт:

```text
4ed2916baca85c13d51dcfc4f99877b5d06c03abce90ea0c1aae8c0506d44aba
```

## SEQ/ETX smoke

- budget: 30 минут;
- фактически учтённый wall-time: 21.42 с;
- checkpoint inference: 1.95 с;
- выборка: 512 пользователей на checkpoint;
- SEQ seed 43/44: побитовое совпадение;
- ETX seed 42/43/44: max abs log error `0.03125`, допуск `0.05` пройден;
- `SEQ-01` seed 42: historical checkpoint отсутствует, frozen prediction есть.

Полный ETX/SEQ training не запускался, что соответствует ограничению задачи.
Машиночитаемые детали: `audits/TABULAR_REPRO_AUDIT.json` и
`audits/DL_SMOKE_AUDIT.json`.
