# exp_020 — G15 learning_rate 0.03

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_020_lr003`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_020_lr003`
- **Original source:** `git:88dc69163b1f:experiments/exp_020_lr003.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.675434`, fold2 `1.741833`, fold3 `1.745412`
- **Known score:** CV mean: `1.708634` (лучший на момент: exp_015, `1.708737`)
- **Seed:** LightGBM, `learning_rate=0.03`, `CALIBRATION_DELTA=-0.17`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_020 — G15 learning_rate 0.03

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Снизить `learning_rate` с 0.05 до 0.03 при accepted calibration delta `-0.17`. Более плавное обучение может дать чуть лучший early-stopping optimum.

## Что изменено относительно базы

Менялся только `learning_rate=0.03`; после reject код возвращён к `learning_rate=0.05`.

## Результат

- CV по фолдам: fold1 `1.675434`, fold2 `1.741833`, fold3 `1.745412`
- CV mean: `1.708634` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean `0.000103`, ниже порога. Это лучшая из одиночных G15-точек, но недостаточная.

## Конфиг прогона

LightGBM, `learning_rate=0.03`, `CALIBRATION_DELTA=-0.17`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
