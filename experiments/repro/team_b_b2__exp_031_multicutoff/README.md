# exp_031 — G4 recheck: multi-cutoff 8x14d на δ=0

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_031_multicutoff`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_031_multicutoff`
- **Original source:** `git:88dc69163b1f:experiments/exp_031_multicutoff.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV mean(1,2): `1.717040` против baseline `1.717017` → Δ = −0.000023
- **Known score:** CV mean(1,2): `1.717040` против baseline `1.717017` → Δ = −0.000023
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_031 — G4 recheck: multi-cutoff 8x14d на δ=0

- **Дата:** 2026-08-13
- **Автор:** Kimi
- **Коммит:** нет (reject; скрипт `artifacts/audit/exp_031_multicutoff.py`)

## Гипотеза

Перепроверка exp_005 (multi-cutoff reject) на честном δ=0 и ровно тех же 50 признаках из HEAD: 8 обучающих cutoff'ов с шагом 14 дней, все ≤ val_cutoff − 30 дней, датасеты из кэша.

## Результат

- fold1 `1.691941` (baseline `1.690608`), fold2 `1.742140` (baseline `1.743426`), fold3 `1.733246` (baseline `1.732506`)
- CV mean(1,2): `1.717040` против baseline `1.717017` → Δ = −0.000023

## Вердикт и вывод

Reject, подтверждён вывод exp_005: простой multi-cutoff 8x14d — шум (знаки дельт по фолдам разные). Структура ошибки не изменилась: fold1 y=0 → 52.5% SLE (было 51.7%). Сам по себе multi-cutoff не лечит сезонный сдвиг тест-окна.
