#!/usr/bin/env bash
# SEQ-01 (exp_025) — dilated TCN на сырой дневной последовательности.
# Общие модули проекта не меняются: весь код в src/seq.py и src/test_seq.py,
# единственная правка shared core — обратно совместимый флаг --fix у src/blend.py.
set -eu
export PYTHONIOENCODING=utf-8

# 0. плотная панель (250000 x 409 x 14) fp16 = 2.9 ГБ, один раз, ~20 c
python -m src.seq build

# 1. анти-лукап: 24 проверки представления, таргета, схемы фолдов и причинности
python -m pytest src/test_seq.py -q

# 2. диагностический фолд с кривой по эпохам — им фиксируется бюджет ёмкости
python -m src.seq fold --val 2025-10-16 --curve --epochs 4

# 3. остальные три фолда тем же бюджетом (RTX 4060 Ti, ~13-16 мин на эпоху)
for V in 2025-10-02 2025-09-18 2025-09-04; do
  python -m src.seq fold --val "$V" --epochs 4
done

# 4. склейка четырёх фолдов в один OOF + отчёт
python -m src.seq merge --exp SEQ-01-S42

# 5. сегменты, AUC, разнообразие, подстановка при фиксированных весах
python -m src.ptime_eval --exps S1-DIST-MIX-SLOT SEQ-01-S42 --base S1-ROUNDS \
  --out research/strategies/results/SEQ

# 6. пофолдовая таблица, разнообразие, кривая доли в смеси
python research/strategies/results/SEQ/analyze.py --exp SEQ-01-S42

# 7. LOFO с боевой смесью. Первый вызов — штатный (веса свободны), второй —
#    со страховкой S1-E03a, зафиксированной на 0.10 (MIX-E11: обнулять нельзя).
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST SEQ-01-S42 \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST SEQ-01-S42 \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo --fix S1-E03a=0.10

# 8. контроль сида на решающем фолде (шум сида у сети втрое выше, чем у GBDT)
python -m src.seq fold --val 2025-10-16 --epochs 4 --seed 43

# 9. тестовая модель (все 29 cutoff'ов) и сабмит. --depth-clip 289 = глубина,
#    прожитая моделью на валидации; на тесте доступны все 365 дней.
python -m src.seq predict --exp SEQ-01 --epochs 4 --depth-clip 289
python -m src.submit --z S1-NORM S1-UNC S1-CAP S1-DIST SEQ-01 \
  --weights 0.15 0.20 0.10 0.25 0.30 --level 2.3293 --out submission_SEQ01_mix.csv
