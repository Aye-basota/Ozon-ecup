#!/usr/bin/env bash
# STRATEGY_04 — интенсив на всём календаре. Полный прогон, namespace S04.
# Все команды идемпотентны; общие модули проекта не изменяются.
set -eu
export PYTHONIOENCODING=utf-8

# 0. кэш признаков для 13 cutoff'ов вне коридора (~1.4 ГБ, 13 x 6-9 c)
python -m src.features --L 0 --min-history 90 --norm-long --extra \
  2025-10-22 2025-10-29 2025-11-05 2025-11-12 2025-11-19 2025-11-26 2025-12-03 \
  2025-12-10 2025-12-17 2025-12-24 2025-12-31 2026-01-07 2026-01-14

# 1. легальность cutoff'ов, баланс расщепления, уровни c(T)
python -m src.calval audit

# 2. анти-лукап: 12 проверок, в том числе побитовая независимость признаков от будущего
python -m pytest src/test_calval.py -q

# 3. четыре фолда, seed 42. Один процесс x 6 потоков: пик ~6.2 ГБ
LGB_THREADS=6 python -m src.calval run --val 2025-10-16 2025-10-02 2025-09-18 2025-09-04 --seed 42

# 4. сборка OOF S04-A / S04-B / S04-C и кривая по ёмкости
python -m src.calval merge --seed 42

# 5. диагностика отравления интенсива и метрики сегментов
python -m src.calval diag --seed 42
python -m src.calval seg  --seed 42

# 6. ось свежести для интенсивной головы (то, ради чего стратегия и нужна на тесте)
LGB_THREADS=6 python -m src.calval gap --val 2025-10-16

# 7. смесь с боевой S1-DIST-MIX: LOFO при фиксированной страховке S1-E03a=0.10
python -m src.calval blend --new S04-B
