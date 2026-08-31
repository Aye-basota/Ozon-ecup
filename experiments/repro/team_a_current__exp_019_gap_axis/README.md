# exp_019 — STRATEGY_01 gap-axis validation

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_019_gap_axis`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_019_gap_axis`
- **Original source:** `experiments/exp_019_gap_axis.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** # exp_019 — STRATEGY_01 gap-axis validation
- **Known score:** `E10`: stress wCV 1.751415 → 1.756366 (**+0.004951**).
- **Seed:** k=5, gaps={30,60,90,120}, actual={35,63,91,126}, seed=config.SEED
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_019 — STRATEGY_01 gap-axis validation

- **Дата:** 2026-08-12
- **Автор:** A1
- **Коммит:** `3c62fa3`

## Гипотеза

При фиксированном числе train cutoff'ов рост разрыва train→validation должен
воспроизвести известный slope bias и показать относительную ценность страховки
`S1-E03a`, создав второй критерий для shift-sensitive стратегий.

## Что изменено относительно базы

Только список train cutoff'ов: последние `k` дат с `T+G≤V`; features, target,
validation panel, loss и параметры моделей не менялись.

## Результат

- Requested G=30/60/90/120 соответствует actual 35/63/91/126; основной `k=5`,
  контроль `k=11` на fold 10-16.
- В каждом плече снята отдельная capacity curve; выбранные rounds лежат в 100–200.
- `E10`: stress wCV 1.751415 → 1.756366 (**+0.004951**).
- Зарегистрированный bias slope провален: **+0.000543/day** против требуемых
  −0.0019…−0.0037; на фиксированных rounds знак также положительный.
- Дефицит `E03a−E10` сужается +0.009488 → +0.006268, но `E03a` проигрывает
  на 4/4 fold в каждой точке; `k=11` даёт тот же вывод.
- f4-control локально тождественен при G30 и G60; заявка о его измеримости при
  G60 противоречит фактическим cutoff-наборам.
- Для дорогого `DIST` полностью сняты решающие G30/G120 и G60; промежуточный G90
  остановлен после первого fold при завершении этапа, поскольку крайние точки уже
  подтверждали тот же rank и не влияли на зарегистрированные A/B/C checks.
- LB: не отправляли; стратегия является validation-инструментом.

Таблицы и полный разбор: `research/strategies/results/STRATEGY_01/`; OOF/reports
и итоговый JSON — в `artifacts/`.

## Вердикт и вывод

**REJECT.** Gap-деградация качества реальна, но pre-registered механизм не
воспроизводится, порядок моделей содержательно не меняется, а bias зависит от k.
`gCV` не становится критерием приёмки; допускается только как диагностика.

## Конфиг прогона

```text
probes={E10,E03a,E02,DIST}, VAL_FOLDS_S1, panel_blocks=3, train_blocks=1
k=5, gaps={30,60,90,120}, actual={35,63,91,126}, seed=config.SEED
direct curve={50,75,100,150,200,250,300}; dist curve={50,100,150,200}
k=11 control: E10/E03a, fold=2025-10-16, та же direct curve
```
