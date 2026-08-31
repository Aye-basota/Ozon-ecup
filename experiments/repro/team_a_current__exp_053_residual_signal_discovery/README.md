# exp_053 — RESIDUAL SIGNAL DISCOVERY

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_053_residual_signal_discovery`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_053_residual_signal_discovery`
- **Original source:** `experiments/exp_053_residual_signal_discovery.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, ensemble
- **Features:** gap/burst features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Alignment/reconstruction: PASS, 770616 строк; folds 188518/191025/193694/197379; exact `(cutoff,user_id)`; `STRONGEST_CURRENT` 1.766883357 / 1.760509577 / 1.748629224 / 1.741278566, wCV 1.747509863.
- **Known score:** В residual `STRONGEST_CURRENT` может оставаться cutoff-safe сигнал, достаточный для улучшения RMSLE не менее чем на 0.001. Проверяем это без новых base-моделей: через semantic ETX/SEQ gate и прямую residual correction, обученные только на сохранённых OOF и существующих признаках.
- **Seed:** Artifact-only; 227 existing cutoff-safe state columns + 34 fixed disagreement columns; LightGBM CPU `num_leaves=31`, `min_data_in_leaf=2000`, `learning_rate=0.03`, 200 rounds, no early stopping, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=20`, `max_bin=63`, row-wise, seed 42 from `config.py`. Donor folds 09-04/09-18/10-02, untouched recipient 10-16, `splitmix64(user_id)&1`; residual scales 0/.25/.50/1 chosen donor-only. Full LOFO not run because late gate failed.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_053 — RESIDUAL SIGNAL DISCOVERY

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** a28a71f

## Гипотеза

В residual `STRONGEST_CURRENT` может оставаться cutoff-safe сигнал, достаточный для улучшения RMSLE не менее чем на 0.001. Проверяем это без новых base-моделей: через semantic ETX/SEQ gate и прямую residual correction, обученные только на сохранённых OOF и существующих признаках.

## Что изменено относительно базы

Добавлен artifact-only audit и два заранее фиксированных CPU probe с two-sided temporal/user cross-fit; production prediction не менялся.

## Результат

- Alignment/reconstruction: PASS, 770616 строк; folds 188518/191025/193694/197379; exact `(cutoff,user_id)`; `STRONGEST_CURRENT` 1.766883357 / 1.760509577 / 1.748629224 / 1.741278566, wCV 1.747509863.
- Oracle headroom: ETX-vs-SEQ +0.020172 wCV; seed-null +0.031483; semantic excess −0.011311; best-existing-member +0.037524. Raw row oracle целиком объясняется seed-selection noise и не является actionable.
- Cutoff-safe predictability на untouched 10-16: winner AUC 0.526759, weighted AUC 0.513530 (shuffle 0.488625 / 0.501694); continuous advantage Pearson −0.000003 (shuffle 0.002143). Seed-winner controls близки к chance.
- Bounded gate: RMSLE 1.741278566 → 1.741272147, Δ −0.000006419; halves −0.000009044 / −0.000003812; shuffled Δ +0.000001454. Gain только в `y=0` (−0.0000959), при `y>0` +0.0000562.
- Signed residual probe: donor-only scale selection выбрал 0 в обеих directions; combined Δ −2.3e−10, identical shuffled control.
- Residual map: устойчивые, но слабые signed axes — buy-day rate / purchase cadence, conversion `cart→order`, long-vs-recent activity and neural-vs-tabular disagreement; максимальный Pearson 0.0259, и на 10-16 он снижается до 0.0170.
- CV mean: baseline mean 1.754325181; основной wCV 1.747509863 (лучший на момент: `exp_037`, 1.747509863).
- LB: не отправляли; test inference и submission не запускались.

## Вердикт и вывод

**NONE.** Ни gate, ни signed-residual correction не достигают даже порога WEAK −0.0003; semantic oracle меньше seed-null на 0.011311, а residual scale честно схлопнулся в 0. Ensemble design не оставляет cutoff-safe gain ≥0.001 и не объясняет gap 0.005. Ветку probe/gating закрыть без tuning; следующий representation bet только как отдельная будущая гипотеза: **BURST-STATE REPRESENTATION — activity episodes + explicit inactivity gaps + regime transitions**.

## Конфиг прогона

Artifact-only; 227 existing cutoff-safe state columns + 34 fixed disagreement columns; LightGBM CPU `num_leaves=31`, `min_data_in_leaf=2000`, `learning_rate=0.03`, 200 rounds, no early stopping, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=20`, `max_bin=63`, row-wise, seed 42 from `config.py`. Donor folds 09-04/09-18/10-02, untouched recipient 10-16, `splitmix64(user_id)&1`; residual scales 0/.25/.50/1 chosen donor-only. Full LOFO not run because late gate failed. Primary runtime 592.5 s; analysis-only hash replay PASS; 37 focused/regression tests PASS.

Artifacts: `research/strategies/results/RESIDUAL_SIGNAL_DISCOVERY/` and `artifacts/RESDISC_053/`.
