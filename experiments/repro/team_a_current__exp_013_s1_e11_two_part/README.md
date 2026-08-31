# exp_013 — S1-E11: двухчастная модель на нормированных длинных окнах

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_013_s1_e11_two_part`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_013_s1_e11_two_part`
- **Original source:** `experiments/exp_013_s1_e11_two_part.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, calibration diagnostic
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: 1.77344 / 1.76548 / 1.75209 / 1.74469.
- **Known score:** CV mean: **1.75893 ± 0.01121**; bias −0.05520, оптимальный сдвиг −0.055,
- **Seed:** LightGBM 600 раундов, параметры по умолчанию из config.LGB_PARAMS, seed из config.py
- **Postprocessing:** после закрытия вопроса об уровне. `eda_findings.md` (`e08`) до этого показал, что
- **Submission:** принятой смеси `S1-BEST`, поэтому в состав сабмита не входила и на leaderboard не
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_013 — S1-E11: двухчастная модель на нормированных длинных окнах

- **Дата:** 2026-08-10
- **Автор:** A1
- **Коммит:** e3b1cde

> Карточка восстановлена по строке `S1-E11` в `experiments/log.csv`
> (2026-08-10T23:52:14) и артефактам `artifacts/oof_S1-E11.npz`,
> `artifacts/feats_S1-E11.txt`. Автор прогона своего вердикта не записал,
> поэтому ниже приведены только измеренные величины и сравнения с уже
> задокументированными прогонами.

## Гипотеза

`strategy_1_results.md` §4.5 оставил двухчастную постановку в списке сознательно
отложенных проверок, а `exp_006` назвал её одним из первых кандидатов на прирост
после закрытия вопроса об уровне. `eda_findings.md` (`e08`) до этого показал, что
двухчастная модель сама по себе примерно равна прямой (1.77008 против 1.77114),
но даёт декорреляцию ошибок.

## Что изменено относительно базы

Модель `two_part` (LightGBM binary для `P(y>0)` × LightGBM регрессия
`E[log1p(y) | y>0]`, `src/models.py`) вместо `direct` на том же наборе признаков
S1-E10 (`norm_long`, 227 колонок).

## Результат

- CV по фолдам: 1.77344 / 1.76548 / 1.75209 / 1.74469.
- CV mean: **1.75893 ± 0.01121**; bias −0.05520, оптимальный сдвиг −0.055,
  калиброванный OOF **1.75792**.
- Сравнение с той же конфигурацией на `direct` (`S1-E10`, `exp_005`):
  CV 1.75988 → 1.75893 (**−0.00095**), калиброванный OOF 1.75889 → 1.75792
  (**−0.00097**).
- Сравнение с принятым `S1-BEST` (`exp_006`): CV 1.75886 против 1.75893
  (+0.00007), калиброванный OOF 1.75716 против 1.75792 (**+0.00076**).
- runtime 1985 с против 1589 с у `S1-E10`.
- **LB: не отправляли** — сабмита из S1-E11 не собиралось.

## Вердикт и вывод

**OPEN.** Одиночная двухчастная модель повторяет результат `e08`: она немного
лучше прямой на том же наборе признаков, но по калиброванному OOF остаётся хуже
принятой смеси `S1-BEST`, поэтому в состав сабмита не входила и на leaderboard не
проверялась. Прогон не отменяет и не заменяет `S1-BEST`; открытый вопрос —
двухчастная модель **внутри** смеси (как источник декорреляции по `e08`), а не
вместо неё; это ещё не измерено.

## Конфиг прогона

```
model=two_part, norm_long=True, L=None, min_history=90
cutoffs=all, шаг 7 (24 обучающих cutoff'а на фолд), train_blocks=1, panel_blocks=3
LightGBM 600 раундов, параметры по умолчанию из config.LGB_PARAMS, seed из config.py
227 признаков
```
