# exp_029 — CALENDAR-PLACEBO-01

## Catalogue metadata

- **Catalogue ID:** `calendar_branch__exp_029_calendar_placebo`
- **Namespace:** `calendar_branch`
- **Experiment ID:** `exp_029_calendar_placebo`
- **Original source:** `git:96271398780a:experiments/exp_029_calendar_placebo.md`
- **Source ref:** `96271398780a33b5423a9971128d4bd946051f0f`
- **Source commit:** `96271398780a33b5423a9971128d4bd946051f0f`
- **Kind:** git-history experiment card
- **Model:** LightGBM, sequence model
- **Features:** calendar features, gap/burst features, history-depth features, dataset/user fingerprint
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Error score: Spearman с squared error **0.029/0.050/0.057/0.050**, RMSLE по
- **Seed:** LightGBM 31 leaves/80 rounds, balanced prior, seed `config.SEED`; standardized
- **Postprocessing:** None documented
- **Submission:** LB/submission: не создавались.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_029 — CALENDAR-PLACEBO-01

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** реализация и карточка в commit ветки `codex/calendar-placebo-01`
- **Код:** `src/calendar_placebo.py`, `src/test_calendar_placebo.py`
- **Полный отчёт:** `research/calendar_placebo_01/README.md`

## Гипотеза

Остаточный fixed-L180 domain AUC около 0.64 из DOMAIN-01 может быть либо обычным
temporal drift, либо особым календарным fingerprint real test. Проверяются
повторные historical placebo-переходы, signed drift vectors, строгая доступность
YoY-аналога и связь test-direction с ошибкой production model.

## Что изменено относительно базы

Production pipeline не менялся. Выполнены 9 balanced fixed-L180 domain placebo
tasks с gap 7/28/56/105 дней и один identically specified real-test comparator;
никакой adaptation, specialist или submission не обучались.

## Результат

- Historical mean ROC-AUC: **0.52219 (7d), 0.55294 (28d), 0.58186 (56d),
  0.62339 (105d)**. Real 120d: **0.64435**, PR-AUC **0.63177**.
- Линейная placebo gap-curve ожидает AUC **0.64550** на 120d; real residual
  **-0.00115** — magnitude полностью обычна для temporal distance.
- Signed real direction отличается: cosine/rank **-0.603/-0.397** с 105d
  placebo и **-0.501/-0.314** со средним девяти placebo vectors.
- Точный YoY cutoff `2025-02-13` невалиден для L180/3-block: доступно лишь
  44 календарных дня и только 1-block panel. Среди валидных состояний
  calendar-nearest совпадает с chronological-nearest (`2025-10-16`).
- Error score: Spearman с squared error **0.029/0.050/0.057/0.050**, RMSLE по
  квантилям немонотонен; в high-score quintile текущий `SEQ-01-MIX` лучший
  (1.57255), ни один компонент не выигрывает по folds.
- LB/submission: не создавались.

## Вердикт и вывод

**STOP-CALENDAR.** В X есть зимний разворот относительно summer/autumn placebo,
но separability по величине полностью объясняется временем, строгий annual
analog недоступен, а test-direction не показывает actionable production-error
или component-win signal. Совместные критерии `CONTINUE-CALENDAR` не выполнены;
Calendar/YoY specialist не обучать.

Ровно один следующий эксперимент: **SEQ-DEPTH-AUG-01** — random depth cropping
при обучении существующего sequence encoder.

## Конфиг прогона

Fixed `L=180`, `panel_blocks=3`, 120k строк на класс, 5 user-hash folds,
LightGBM 31 leaves/80 rounds, balanced prior, seed `config.SEED`; standardized
logistic coefficients только как secondary signed direction. Запуск:
`python src/calendar_placebo.py run --baseline-artifacts <path> --resume`.
