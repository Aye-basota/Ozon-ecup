# STRATEGY_02 — Variant A (`train_blocks=0`)

- **Дата:** 2026-08-12
- **Scope:** только Variant A; B/C/D не запускались
- **База:** `S1-E10`, `direct`, 227 признаков, `L=None`, `norm_long`, 4 fold S1
- **Артефакты:** `artifacts/sample_design_S1-SAMPLE-A-FINAL.json`, OOF и reports
  `SAMPLE-TB{0,1}-AVG3-R{200,300}`

## Capacity-matched screening

Оба плеча получили одну curve `150/200/250/300/450`; полная таблица —
[`capacity_curve.csv`](capacity_curve.csv). По правилу STRATEGY_05 (минимум
калиброванного последнего fold, затем wCV) база выбирает 300 rounds, Variant A —
200 rounds. На seed 42 дельта A к базе равна **−0.000049 wCV**, но последний fold
хуже на +0.000134: это меньше seed-noise, поэтому решение перенесено на avg3.

## 3-seed result

| config | rounds | wCV avg3 | Δ к базе | fold wins | 2025-10-16 |
|---|---:|---:|---:|---:|---:|
| `train_blocks=1` | 300 | **1.750456** | — | — | 1.743906 |
| `train_blocks=0` | 200 | 1.750569 | **+0.000113** | 2/4 | 1.744099 (+0.000194) |

Подробные значения — [`seed_robustness.csv`](seed_robustness.csv). Variant A
увеличил train последнего fold с 4,955,174 до 5,412,520 строк (**+9.23%**), но
не улучшил качество. OOF bias изменился лишь на +0.000934; смены уровня нет.

Диагностика также не показывает полезной новой функции: после avg3
`Var(z_A-z_base)=0.00279`, corr остатков 0.99955. Против `S1-DIST`
`Var(Δ)=0.00776`, лишь на 0.00064 выше seed-noise floor 0.00712 из STRATEGY_05.
Adversarial AUC к тестовой панели уже почти предельный у базы и слегка выше у A:
0.9999935 → 0.9999967.

## Решение: **REJECT**

Порог `Δ≤−0.0005`, ≥3/4 fold и обязательная победа 10-16 не выполнены ни по
одному условию. `train_blocks=0` не входит в winning pipeline и representative
submission; ручка фиксируется на `train_blocks=1`.

Воспроизведение:

```bash
python -m src.sampleval --run --train-blocks 0 --seeds 42 --rounds 450 --curve 150 200 250 300 450
python -m src.sampleval --run --train-blocks 0 --seeds 43 44 --rounds 200 --curve 200
python -m src.sampleval --summarize --adversarial --curve 150 200 250 300 450 --out S1-SAMPLE-A-FINAL
```
