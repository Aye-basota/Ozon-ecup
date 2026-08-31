# exp_053 — RESIDUAL SIGNAL DISCOVERY

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** a28a71f

## Гипотеза

В residual `STRONGEST_CURRENT` может оставаться cutoff-safe сигнал, достаточный для улучшения RMSLE не менее чем на 0.001. Проверяем это без новых base-моделей: через semantic ETX/SEQ gate и прямую residual correction, обученные только на сохранённых OOF и существующих признаках.

## Что изменено относительно базы

Добавлен artifact-only audit и два заранее фиксированных CPU probe с two-sided temporal/user cross-fit; production prediction не менялся.

## Результат

- Exact reconstruction PASS: 1.766883357 / 1.760509577 / 1.748629224 / 1.741278566; wCV 1.747509863.
- Oracle: ETX-vs-SEQ +0.020172; seed-null +0.031483; semantic excess −0.011311; best-existing +0.037524.
- Winner probe: AUC 0.526759, weighted AUC 0.513530; advantage Pearson −0.000003.
- Gate on 10-16: Δ −0.000006419; halves −0.000009044 / −0.000003812; shuffle +0.000001454. `y=0` improves, `y>0` worsens.
- Residual probe: donor-selected scale 0 in both directions; Δ −2.3e−10, same as shuffle.
- LB/test/submission: not run.

## Вердикт и вывод

**NONE.** Actionable cutoff-safe gain ≥0.001 не найден; full artifact LOFO не разрешён. Следующая отдельная гипотеза: **BURST-STATE REPRESENTATION — activity episodes + explicit inactivity gaps + regime transitions**.

## Конфиг прогона

227 state + 34 disagreement features; fixed LightGBM CPU recipe, 200 rounds, seed 42; donor 09-04/09-18/10-02, recipient 10-16, two-sided `splitmix64(user_id)&1`; runtime 592.5 s; analysis-only hashes PASS; tests 37 PASS.
