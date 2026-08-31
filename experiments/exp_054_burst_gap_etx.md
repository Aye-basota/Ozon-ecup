# exp_054 — BURST-GAP-ETX: activity episodes and inactivity transitions

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** a28a71f + рабочее дерево

## Гипотеза

Явные activity bursts, gaps и transitions могут содержать purchase-occurrence signal, который теряется в отдельных event-day токенах ETX и плотных calendar-day ячейках TCN. Эксперимент audit-gated: до GPU допускается только фиксированный threshold-3 CPU probe против matched joint shuffle.

## Что изменено относительно базы

Добавлен изолированный cutoff-safe BURST/GAP tokenizer и 12 фиксированных episode summaries; архитектура/target/loss не запускались, потому что CPU gate провален.

## Результат

- Exact baseline audit: PASS; `STRONGEST_CURRENT` fold 10-16 = **1.741278566**, wCV **1.747509863**.
- Structural novelty: PASS; overflow **0%**; лучший mid-activity `P(y>0)` spread **0.01998** при требуемых 0.03.
- REAL/SHUFFLED signed-residual probe: donor scales **0/0** у обоих; late delta **0.000000**, REAL−SHUFFLED **0.000000**; обе recipient halves — tie.
- Activity AUC REAL/SHUFFLED = **0.847581 / 0.847560**, gain **+0.000021** при требуемых +0.002.
- CV mean: CPU endpoint не является новой моделью; текущий лучший остаётся `exp_037`, wCV **1.747509863**.
- LB: не отправляли; GPU, test inference и submission не запускались.

## Вердикт и вывод

**NO_GO_PREFLIGHT.** Episode summaries частично новы относительно state, но их residual/occurrence signal полностью совпадает с matched SHUFFLED control, а donor-only scale схлопывается в ноль. Exact threshold-3 BURST/GAP route закрыт без tuning; neural pilot запрещён собственным гейтом.

## Конфиг прогона

Фолды 2025-09-04/09-18/10-02/10-16; fixed burst threshold 3; base EXP-053 COMBINED 261 колонка + 12 episode summaries; LightGBM leaves 31, min leaf 2000, lr .03, 200 rounds, feature/bagging .8, L2 20, max_bin 63; scales 0/.25/.50/1; split `splitmix64(user_id)&1`; seed 42 из `config.py`. Полный отчёт: `research/strategies/results/BURST_GAP_EXP054/REPORT.md`.
