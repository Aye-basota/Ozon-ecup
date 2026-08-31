# exp_032 — C: delta-modeling (поправка к naive30)

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_032_delta_model`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_032_delta_model`
- **Original source:** `git:88dc69163b1f:experiments/exp_032_delta_model.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV mean(1,2): `1.720406` против baseline `1.717017` → Δ = −0.003389
- **Known score:** CV mean(1,2): `1.720406` против baseline `1.717017` → Δ = −0.003389
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_032 — C: delta-modeling (поправка к naive30)

- **Дата:** 2026-08-13
- **Автор:** Kimi
- **Коммит:** нет (reject; скрипт `artifacts/audit/exp_032_delta_model.py`)

## Гипотеза

Таргет = `log1p(y) − log1p(w30_gmv + 1)`; модель учит поправку к наивному прогнозу, финал = `expm1(log1p(w30_gmv+1) + поправка)`.

## Результат

- fold1 хуже (см. лог), fold2 `1.746123` (baseline `1.743426`), fold3 `1.734228` (baseline `1.732506`)
- CV mean(1,2): `1.720406` против baseline `1.717017` → Δ = −0.003389

## Вердикт и вывод

Reject: хуже на всех фолдах. Переформулировка таргета как поправки к naive30 не помогает — модель и так использует `w30_gmv` как сильнейший признак, явная декомпозиция только добавляет смещение.
