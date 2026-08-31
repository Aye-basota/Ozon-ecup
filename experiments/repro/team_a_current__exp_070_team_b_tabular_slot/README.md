# exp_070 — TEAM-B tabular slot replacement

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_070_team_b_tabular_slot`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_070_team_b_tabular_slot`
- **Original source:** `experiments/exp_070_team_b_tabular_slot.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** sequence model
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Её wCV: **1.747127499**, дельта **−0.000382364**, лучше 4/4; по фолдам `1.766770 / 1.760047 / 1.748265 / 1.740873`.
- **Known score:** Её wCV: **1.747127499**, дельта **−0.000382364**, лучше 4/4; по фолдам `1.766770 / 1.760047 / 1.748265 / 1.740873`.
- **Seed:** Canonical S1 09-04/09-18/10-02/10-16, веса фолдов 1:2:4:8, per-fold log calibration; seed/training NONE. Вес `alpha∈[0,1]` минимизирован непрерывно только по OOF, затем проверен LOFO. Компоненты и team predictions выровнены по `(cutoff,user_id)` с exact key и target parity.
- **Postprocessing:** Production regime коррекции: `Var(Δz)` OOF `0.0013873`, TEST `0.0014176`, ratio **1.022 PASS**; mean сдвиг убирается штатной production level calibration.
- **Submission:** LB/submission: не запускали и не создавали.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_070 — TEAM-B tabular slot replacement

- **Дата:** 2026-08-27
- **Автор:** A1
- **Коммит:** `a28a71f` + working tree; team source/predictions pinned at `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`

## Гипотеза

Вместо добавления team B2 ко всему `STRONGEST-CURRENT` заменить только его табличный блок `0.10 CAP + 0.20 UNC + 0.25 DIST`, сохранив sequence-слот `0.225 SEQ-AVG3 + 0.225 ETX-AVG3`. Проверить полную замену и одномерный optimum доли team внутри фиксированного 55%-го табличного слота.

## Что изменено относительно базы

Training NONE: только artifact-only линейная комбинация canonical OOF и сохранённых `exp_069` team OOF.

## Результат

- База: **1.747509863**.
- Полная native log-space замена `0.55 team + 0.225 SEQ + 0.225 ETX`: **1.747700578**, дельта **+0.000190716**, хуже 3/4; raw-GMV аналог ещё хуже, **+0.000300946**.
- Оптимальная доля team внутри tab-slot: **0.449575668**, то есть абсолютный вес team **0.247266617**.
- Оптимальная смесь: `0.055042 CAP + 0.110085 UNC + 0.137606 DIST + 0.247267 TEAM + 0.225 SEQ + 0.225 ETX`.
- Её wCV: **1.747127499**, дельта **−0.000382364**, лучше 4/4; по фолдам `1.766770 / 1.760047 / 1.748265 / 1.740873`.
- Честный LOFO: **1.747128374**, дельта **−0.000381489**, 4/4; held-out доля team внутри слота `0.4404…0.4603`.
- Production regime коррекции: `Var(Δz)` OOF `0.0013873`, TEST `0.0014176`, ratio **1.022 PASS**; mean сдвиг убирается штатной production level calibration.
- LB/submission: не запускали и не создавали.

## Вердикт и вывод

**Полная замена REJECT.** Частичная slot-интеграция устойчива и лучше общего raw-бленда `exp_069`, но честный выигрыш `0.000381` всё ещё ниже project gate `0.0005`; это development/noise, не новый чемпион и не основание для отправки. Одномерная кривая исчерпана, соседние веса не тюнить.

## Конфиг прогона

Canonical S1 09-04/09-18/10-02/10-16, веса фолдов 1:2:4:8, per-fold log calibration; seed/training NONE. Вес `alpha∈[0,1]` минимизирован непрерывно только по OOF, затем проверен LOFO. Компоненты и team predictions выровнены по `(cutoff,user_id)` с exact key и target parity.
