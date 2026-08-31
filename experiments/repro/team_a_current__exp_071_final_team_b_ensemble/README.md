# exp_071 — финальный ансамбль STRONGEST-CURRENT + team-b-B2

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_071_final_team_b_ensemble`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_071_final_team_b_ensemble`
- **Original source:** `experiments/exp_071_final_team_b_ensemble.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** sequence model, ensemble, blend, calibration diagnostic
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Вес выбирать без LB: непрерывный optimum на канонической S1 wCV, с обязательной
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** веса фолдов 1:2:4:8, пофолдовая log-калибровка; production level `2.3293`.
- **Submission:** Submission: `submissions/submission_FINAL_CAP_UNC_DIST_SEQ_ETX_TEAM_B.csv`,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_071 — финальный ансамбль STRONGEST-CURRENT + team-b-B2

- **Дата:** 2026-08-29
- **Автор:** A1
- **Коммит:** `a28a71f` + working tree; team source pinned at `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`

## Гипотеза

По явному запросу собрать финальный submission из нашего `STRONGEST-CURRENT`
(`CAP + UNC + DIST + SEQ-AVG3 + ETX-AVG3`) и лучшего pinned решения `team-b-B2`.
Вес выбирать без LB: непрерывный optimum на канонической S1 wCV, с обязательной
LOFO-проверкой и OOF→TEST regime gate.

## Что изменено относительно базы

`team-b-B2` частично заменяет только фиксированный 55%-й табличный слот; веса
`SEQ-AVG3=.225` и `ETX-AVG3-DCW=.225` не меняются.

## Результат

- Выбранная доля team B внутри tab-slot: **0.449575571**; абсолютный вес
  **0.247266564**.
- Финальные веса: `CAP=.055042443`, `UNC=.110084886`, `DIST=.137606107`,
  `TEAM_B=.247266564`, `SEQ=.225`, `ETX=.225`.
- База: **1.747509863**; кандидат: **1.747127499**, дельта
  **−0.000382364**, лучше **4/4**.
- CV по фолдам: **1.766770 / 1.760047 / 1.748265 / 1.740873**.
- Честный LOFO: **1.747128374**, дельта **−0.000381489**, лучше **4/4**;
  held-out доли team в tab-slot `0.4603 / 0.4417 / 0.4510 / 0.4404`.
- OOF→TEST `Var(Δz)` ratio: **1.021814, PASS**.
- LB: **не использовался и не отправляли**.
- Submission: `submissions/submission_FINAL_CAP_UNC_DIST_SEQ_ETX_TEAM_B.csv`,
  SHA256 `d6cdb218a4d149acf3479f32ab875bfbe4d89eca222f09910750309de19698b9`.

## Вердикт и вывод

**FINAL PREPARED по явному запросу.** Вес не опирается на LB и устойчив по LOFO;
файл прошёл проверку формата, порядка 250,000 пользователей, конечности,
неотрицательности и production regime. Исследовательский вывод `exp_070` не
меняется: выигрыш меньше project gate `0.0005`, поэтому `STRONGEST-CURRENT`
остаётся строго подтверждённым research/private-safe anchor.

## Конфиг прогона

Training NONE; artifact-only log-space blend. S1 09-04/09-18/10-02/10-16,
веса фолдов 1:2:4:8, пофолдовая log-калибровка; production level `2.3293`.
Воспроизведение одной командой: `python src/final_team_b_ensemble.py`.
