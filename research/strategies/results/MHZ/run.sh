#!/usr/bin/env bash
# MHZ (exp_024) — multi-horizon hazard + count supervision. Полный прогон.
# Общие модули проекта не изменяются: весь код в src/mhz.py и src/test_mhz.py.
set -eu
export PYTHONIOENCODING=utf-8

# 0. анти-лукап: 18 проверок разметки, легальности cutoff'ов и кросс-фиттинга
python -m pytest src/test_mhz.py -q

# 1. четыре фолда, seed 42. ОДИН процесс x 6 потоков: пик ~9.9 ГБ на фолде 10-16,
#    два процесса в 31.6 ГБ не помещаются. Порядок — от решающего фолда к раннему.
for V in 2025-10-16 2025-10-02 2025-09-18 2025-09-04; do
  LGB_THREADS=6 python -m src.mhz fold --val "$V"
done

# 2. склейка четырёх арок лестницы, отчёты, строки в experiments/log.csv
python -m src.mhz merge

# 3. диагностики по сохранённым OOF: сегменты, горизонты, головы, форма кривой
python research/strategies/results/MHZ/analyze.py

# 4. мета-модель поверх боевой смеси + hazard/count (temporal-safe)
python -m src.mhz meta

# 5. смесь с боевой S1-DIST-MIX: LOFO при фиксированной страховке S1-E03a = 0.10
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST MHZ-FULL \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo
