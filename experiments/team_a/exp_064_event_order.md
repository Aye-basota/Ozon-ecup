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
