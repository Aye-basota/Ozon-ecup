# exp_061 — OPEN-FUNNEL unresolved intent preflight

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_061_open_funnel`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_061_open_funnel`
- **Original source:** `experiments/exp_061_open_funnel.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, ensemble
- **Features:** funnel features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Exact base fold RMSLE: `1.766883356 / 1.760509577 / 1.748629224 / 1.741278566`; wCV `1.747509862`.
- **Known score:** Exact base fold RMSLE: `1.766883356 / 1.760509577 / 1.748629224 / 1.741278566`; wCV `1.747509862`.
- **Seed:** seed=config.SEED=42; split=splitmix64(user_id); scales=0/.25/.50/.75/1
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_061 — OPEN-FUNNEL unresolved intent preflight

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** `a28a71f` + shared working tree

## Гипотеза

Search/Cart activity strictly after a user's last positive-GMV day measures unresolved purchase intent and should explain signed residuals of exact `STRONGEST-CURRENT`, especially for low-buy and stale-buy users. The full frozen Hypothesis Card was written before results to `research/strategies/results/OPEN_FUNNEL_EXP061/HYPOTHESIS_CARD.md`.

## Что изменено относительно базы

The baseline predictions and all ensemble weights were unchanged. Fourteen opt-in, cutoff-safe columns were added through `build_features(cutoff)`: Search/Cart days, volumes, no-order days, overlap, oldest ages and spans within the last 90 days but strictly after the last purchase. An identical probe with the 14 columns jointly shuffled inside 392--396 fixed baseline-state strata was the control.

## Leakage / integrity audit

- Source filter is exactly `event_date <= cutoff`; the boundary purchase day is excluded because intraday order is unknown.
- The accumulation window is fixed at 90 days; the last-buy boundary uses only history observed at the same cutoff.
- Unit test includes a huge future event and proves it cannot affect any column.
- Exact OOF alignment passed at `188518 / 191025 / 193694 / 197379` unique users; finite audit passed.
- Joint shuffle moved `99.77--99.81%` of rows, preserved every column marginal exactly, and had singleton share below `0.015%`.

## Результат

- Exact base fold RMSLE: `1.766883356 / 1.760509577 / 1.748629224 / 1.741278566`; wCV `1.747509862`.
- REAL fold deltas: `0 / 0 / 0 / 0`; nested two-sided `Delta wCV = 0`.
- SHUFFLED fold deltas: `0 / 0 / 0 / 0`; REAL-SHUFFLED `= 0`.
- CONTROL_ONLY is also exactly zero. Every one of the 24 donor-side scale selections (3 arms x 4 folds x 2 sides) chose frozen `scale=0`.
- Raw feature/residual correlations reached about `|r|=0.0223`, but none survived controls and cross-user prediction. LOW_BUY90 and REC_BUY_GT30 also received zero correction on both halves.
- Runtime `~50.9 s` CPU, four LightGBM threads. GPU/full model/test inference/LB/submission were not run.

## Вердикт и вывод

**REJECT.** The temporal state is real enough to correlate marginally with residuals, but provides no usable incremental correction over the baseline state. The frozen success gate fails maximally: 0/4, latest fold zero, REAL-SHUFFLED zero and scales zero on both recipient halves. Do not rescue this exact family with windows, thresholds, probe learner or scale tuning.

Information gain: aggregate recencies/counts and the champion already absorb the useful part of post-purchase Search/Cart state; the remaining marginal correlation is conditional redundancy.

## Конфиг прогона и provenance

```text
python src/open_funnel.py
folds=2025-09-04/09-18/10-02/10-16; weights=1:2:4:8
seed=config.SEED=42; split=splitmix64(user_id); scales=0/.25/.50/.75/1
probe=LightGBM L1, leaves=15, min_data=1000, rounds=120, lambda_l2=50, threads=4
aligned OOF SHA256=40aa9719d4fd903467eceb5f7be5e5d19ce6a67e12856f523946e5fdcda0ed0a
summary SHA256=6351e453f43e9808f20d31c852cdd2229ac6628ab3bee654a187ad275a3f57d0
```

Artifacts: `artifacts/OPEN_FUNNEL_EXP061/`; registered results: `research/strategies/results/OPEN_FUNNEL_EXP061/`.
