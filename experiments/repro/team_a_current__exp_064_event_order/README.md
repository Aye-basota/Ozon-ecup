# exp_064 — EVENT-ORDER

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_064_event_order`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_064_event_order`
- **Original source:** `experiments/exp_064_event_order.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** LightGBM
- **Features:** funnel features, window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** REAL / SHUFFLED / CONTROL_ONLY по всем folds: exact base scores; `Delta wCV = 0 / 0 / 0`, 0/4, latest `0`.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** CPU, seed 42 из `src/config.py`; exact aligned exp_037 SHA256 `40aa9719...0ed0a`; 15 event-order features + 11 fixed controls; LightGBM residual 120 rounds, leaves 15, min leaf 1000, L2 50; 4-way cross-user + two-sided scale `0/.25/.5/.75/1`; canonical cutoffs, weights 1:2:4:8. Runtime 62.95s; summary SHA256 `4ad55d3c...1f3d`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_064 — EVENT-ORDER

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** a28a71f + working tree

## Гипотеза

Направление переходов между дневными Search/Cart/Buy состояниями пользователя несёт intent-resolution сигнал, которого нет в оконных маргинальных агрегатах. Explicit 90-day transition motifs должны объяснять остаток exact exp_037 лучше placebo, переставляющего полные state-векторы между наблюдаемыми датами того же пользователя.

## Что изменено относительно базы

Через opt-in `build_features(cutoff_date, event_order_source=...)` добавлены 15 новых `eo_*` колонок; существующие признаки не менялись. SHUFFLED сохраняет пользователя, даты/гэпы и exact per-user multiset трёхбитных состояний, разрушая только порядок.

## Результат

- REAL / SHUFFLED / CONTROL_ONLY по всем folds: exact base scores; `Delta wCV = 0 / 0 / 0`, 0/4, latest `0`.
- Все 24 donor-side arms (4 folds × 3 arms × 2 halves) выбрали scale `0`.
- Transition-count parity exact; changed feature-row share `97.62% / 97.71% / 97.76% / 97.77%`; cutoff/alignment/finiteness audits PASS.
- Raw feature↔residual corr до `0.0171`, но conditional correction полностью обнуляется.
- LB: не отправляли; canonical model/test не запускались.

## Вердикт и вывод

**REJECT.** Matched-order shuffle действительно разрушил представление, но REAL не лучше SHUFFLED/CONTROL. Explicit daily funnel-order family условно избыточна с чемпионом; не спасать state alphabets, windows/lags, high-capacity learners, segment gates или neural retraining.

## Конфиг прогона

CPU, seed 42 из `src/config.py`; exact aligned exp_037 SHA256 `40aa9719...0ed0a`; 15 event-order features + 11 fixed controls; LightGBM residual 120 rounds, leaves 15, min leaf 1000, L2 50; 4-way cross-user + two-sided scale `0/.25/.5/.75/1`; canonical cutoffs, weights 1:2:4:8. Runtime 62.95s; summary SHA256 `4ad55d3c...1f3d`.
