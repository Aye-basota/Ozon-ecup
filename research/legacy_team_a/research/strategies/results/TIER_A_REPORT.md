# Tier A checkpoint

Дата: 2026-08-12. Главная метрика — wCV; знак Δ: кандидат минус база.

## Results

| Strategy | Variant | Best config | wCV | Δ | Gap-axis result | Seed robustness | Decision |
|---|---|---|---:|---:|---|---|---|
| STRATEGY_05 | A+B | `direct`, 300 rounds, avg3 | **1.750456** | −0.00124 к E10/600; −0.000018 в fixed mix | n/a | avg3 подтверждает; 3→5 даёт лишь −0.00013 | **PARTIAL** |
| STRATEGY_01 | k=5 + k=11 control | E10: 150 rounds at G126 | 1.756366 (gCV) | +0.004951 к G35 | bias slope +0.000543/day вместо отрицательного; E03a сокращает дефицит, но не меняет rank | эффекты > seed-noise; вывод нестабилен по k | **REJECT** |
| STRATEGY_02 | A, `train_blocks=0` | 200 rounds, avg3 | 1.750569 | **+0.000113** к capacity-matched base | S01 не принят как gate | avg3; 2/4 fold, последний хуже | **REJECT** |

## Tier A verdict

- **STRATEGY_05 A+B — PARTIAL.** Одиночный `direct` честно выигрывает от 300
  rounds и avg3, но замена только его 15%-ного члена в текущей смеси даёт
  −0.000018 wCV и слегка проигрывает последний fold. C/D не выполнены.
- **STRATEGY_01 — REJECT.** Gap-деградация реальна, но зарегистрированный slope
  не воспроизведён, страховка не меняет порядок, а результат зависит от k.
  gCV остаётся диагностикой, не критерием выбора для 02C/04/09/10/12.
- **STRATEGY_02 A — REJECT.** +9.23% train-строк не дали качества после отдельной
  capacity curve и avg3; `train_blocks=1` остаётся рабочим значением.

## Submission

Конфиг `TIER-A-CHECKPOINT`:

```text
0.15 * TIER-A-DIRECT-AVG3-R300 + 0.30 * S1-UNC + 0.10 * S1-CAP + 0.45 * S1-DIST
level = 2.3293, log-space blend, train_blocks=1
```

Включено только подтверждённое изменение STRATEGY_05 для проверенного `E10`;
структура и веса production mixture сохранены. STRATEGY_02A исключена,
STRATEGY_01 prediction pipeline не меняет. Файл:
`submissions/submission_tier_a_checkpoint.csv`. Он прошёл стандартные проверки,
но автоматически на leaderboard не отправлялся: uploader в репозитории отсутствует.
Для отправки нужно вручную загрузить этот CSV в форму submission соревнования; файл
уже содержит 250,000 строк в исходном порядке `user_id`.

Ожидаемое отличие от `submission_dist_head.csv`: OOF wCV 1.749483597 →
1.749465322 (**−0.000018**), последний fold +0.000003; это ниже локального и
public-LB разрешения. Таблица — [`tier_a_fixed_mix_oof.csv`](tier_a_fixed_mix_oof.csv).
На test предсказания старого и нового submission имеют log-correlation 0.999970 и
среднее `|Δ log1p|=0.00857`; это диагностическая замена члена смеси, а не новый blend.

## Что изменилось в winning-pipeline hypothesis

Усреднение трёх seed и capacity около 300 остаются гигиеной проверенного direct,
но их нельзя без Variant C/D переносить на все члены смеси. Train-панель остаётся
1-блочной. Предложенный gap-CV не снимает неопределённость 120-дневного переноса,
поэтому фиксированный вес 0.10 страховки сохраняется как prior.

## Следующий эксперимент

**STRATEGY_02 Variant B — плотная сетка при равном объёме.** Это следующий пункт
`STRATEGIES_INDEX.md` и прямой gate для STRATEGY_10/HDN; отрицательный результат A
не проверял механизм разнообразия target-окон, который изолирует B.
