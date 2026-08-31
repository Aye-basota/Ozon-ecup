# exp_020 — STRATEGY_02A: `train_blocks=0`

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_020_train_blocks_zero`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_020_train_blocks_zero`
- **Original source:** `experiments/exp_020_train_blocks_zero.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Seed 42: Δ = −0.000049 wCV, 3/4 fold, но последний хуже на +0.000134.
- **Known score:** Seed 42: Δ = −0.000049 wCV, 3/4 fold, но последний хуже на +0.000134.
- **Seed:** VAL_FOLDS_S1, seeds={config.SEED=42,43,44}, log-space avg3
- **Postprocessing:** VAL_FOLDS_S1, seeds={config.SEED=42,43,44}, log-space avg3
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_020 — STRATEGY_02A: `train_blocks=0`

- **Дата:** 2026-08-12
- **Автор:** A1
- **Коммит:** `27098ea`

## Гипотеза

Снятие последнего фильтра train-панели добавит 10–20% строк и улучшит direct-модель.
После STRATEGY_05 capacity выбирается отдельно в каждом плече, а малая дельта
подтверждается средним трёх seed.

## Что изменено относительно базы

Только `train_blocks: 1 → 0`; features, target, folds, loss и остальные параметры
совпадают с `S1-E10`.

## Результат

- Capacity curve `150/200/250/300/450`: база выбрала 300, Variant A — 200 rounds.
- Seed 42: Δ = −0.000049 wCV, 3/4 fold, но последний хуже на +0.000134.
- Avg3 (42/43/44): база **1.750456**, Variant A **1.750569**,
  Δ = **+0.000113**, 2/4 fold, последний хуже на +0.000194.
- Строк последнего fold: 4,955,174 → 5,412,520 (**+9.23%**).
- `Var(z_A-z_base)=0.00279` после avg3; adversarial AUC 0.9999935 → 0.9999967.
- LB: не отправляли.

Полные таблицы: `research/strategies/results/STRATEGY_02/`; OOF/reports:
`artifacts/SAMPLE-TB*`, сводка `artifacts/sample_design_S1-SAMPLE-A-FINAL.json`.

## Вердикт и вывод

**REJECT.** После честной capacity и avg3 вариант хуже, выигрывает лишь два fold
и проигрывает 10-16. Дополнительные пользователи не дают полезного сигнала;
`train_blocks=1` остаётся в winning pipeline.

## Конфиг прогона

```text
direct, L=None, norm_long=True, min_history=90, step=7, train_blocks={1,0}
panel_blocks=3, cutoffs=all, rounds curve={150,200,250,300,450}
VAL_FOLDS_S1, seeds={config.SEED=42,43,44}, log-space avg3
```
