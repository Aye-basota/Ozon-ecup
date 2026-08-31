# exp_028 — DOMAIN-01 / Test-Like Validation

## Catalogue metadata

- **Catalogue ID:** `domain_branch__exp_028_domain_shift`
- **Namespace:** `domain_branch`
- **Experiment ID:** `exp_028_domain_shift`
- **Original source:** `git:c219dabe3169:experiments/exp_028_domain_shift.md`
- **Source ref:** `c219dabe316982bb9de23e15c4f24d3e9eb16ae0`
- **Source commit:** `c219dabe316982bb9de23e15c4f24d3e9eb16ae0`
- **Kind:** git-history experiment card
- **Model:** LightGBM, sequence model, ensemble
- **Features:** calendar features, gap/burst features, history-depth features, dataset/user fingerprint, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** # exp_028 — DOMAIN-01 / Test-Like Validation
- **Known score:** recipe: ordinary wCV **1.751164 vs 1.751076 (+0.000087)**, weighted wCV
- **Seed:** ablations 80 rounds, seed `config.SEED`.  Adaptation: direct, 300 rounds,
- **Postprocessing:** None documented
- **Submission:** LB/submission: не отправлялись и candidate CSV не создавался — gates не пройдены.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_028 — DOMAIN-01 / Test-Like Validation

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** реализация и карточка в commit ветки `codex/domain-01`
- **Код:** `src/domain01.py`, `src/test_domain01.py`
- **Полный отчёт:** `research/domain_01/README.md`

## Гипотеза

Исторические CV-состояния могут отличаться от real test настолько, что текущая
валидация неверно ранжирует модели.  Проверяется честным domain classifier с
user-group OOF, distribution diagnostics, importance-weighted CV и одним
минимальным weighted-training экспериментом.

## Что изменено относительно базы

Production pipeline не менялся.  Диагностически построены D0/D1 и строго OOF
`p_test_like`; единственный production experiment — direct 300 rounds с теми же
227 features/cutoffs и мягкими behavioral odds weights (T=0.5, clip 0.25..4).

## Результат

- D0 linear OOF ROC-AUC **0.935485**; D1 production **0.998567**, PR-AUC
  **0.996046**, Brier 0.017225, ECE-10 0.041512.
- Raw depth-only AUC **0.986102**; fixed-L180 **0.647465**; behavioral-only
  **0.639283**; missingness-only 0.559672.  Почти идеальный primary AUC — главным
  образом technical history-depth/support fingerprint.
- 5 главных драйверов: `tenure_frac` (50.1% gain; permutation ΔAUC 0.2504),
  `w365_days_cat`, `gap_max_frac`, `trend_gmv_90_365`, `gap_cv`; далее `gap_std`,
  `w365_days_buy`, `w365_days_presence_only`, `rec_cat`, `first_buy_frac`.
- Наиболее test-like fold — **2025-10-16**: mean primary p 0.04854 против
  0.03491/0.03569/0.03870; behavioral p 0.23847 против 0.23559/0.23582/0.23689.
- Ranking сильных моделей под пятью заранее заданными weight schemes не меняется:
  `SEQ-01-MIX` остаётся первым, далее DIST-MIX/SEEDAVG3/DIST/ROUNDS.
- Behavioral weights сохраняют `n_eff/n=0.968`; weighted direct против того же
  recipe: ordinary wCV **1.751164 vs 1.751076 (+0.000087)**, weighted wCV
  **1.736810 vs 1.736725 (+0.000086)**. Fold deltas
  `+0.000010/-0.000088/-0.000285/+0.000327`: 2/4, последний хуже.
- Production-slot delta к ROUNDS-control +0.000007; LOFO **+0.000010**.
  `Var(Δ)=0.006578=0.924x` seed floor, corr residuals с base 0.998936.
- LB/submission: не отправлялись и candidate CSV не создавался — gates не пройдены.

## Вердикт и вывод

**STOP.** Shift доказан, но почти весь separability — техническая глубина истории.
Остаточный behavioral/calendar shift реален (AUC около 0.64), однако не меняет
production ranking; мягкая adaptation ухудшает и обычный, и test-like weighted
CV и не даёт incremental ensemble signal.

Ровно один следующий эксперимент: **CALENDAR-PLACEBO-01** — grouped domain
classification между historical cutoff pairs при фиксированном L180 и разных
time gaps, с сопоставлением направления drift против real-test L180 classifier.

## Конфиг прогона

Domain: 5 user-hash folds, D0 SGD-logistic, D1 LightGBM 31 leaves/120 rounds,
ablations 80 rounds, seed `config.SEED`.  Adaptation: direct, 300 rounds,
behavioral D1 odds, temperature 0.5, final clip `[0.25,4]`; CV и production
mixture по текущей схеме проекта.  Запуск одной командой на этап:
`python src/domain01.py diagnose ...`; `python src/domain01.py adapt --resume ...`.
