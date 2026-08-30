#!/usr/bin/env bash
# SEQ-02 (exp_026) — усреднение сидов SEQ-01 + диагностика по глубине истории.
# Архитектура, представление, валидация и таргет НЕ меняются относительно exp_025.
# Единственная правка shared core остаётся прежней (флаг --fix у src/blend.py).
set -eu
export PYTHONIOENCODING=utf-8

# 0. панель и анти-лукап (29 проверок: 24 из SEQ-01 + 5 на глубину истории)
python -m src.seq build
python -m pytest src/test_seq.py -q

# 1. профиль времени до/после. --compile ускоряет шаг вдвое, но ТОЛЬКО при
#    фиксированной форме батча: без добивки dynamo перекомпилируется на каждую
#    из ~25 форм за эпоху и выигрыш падает до 1.2x.
python -m src.seq bench --n-cutoffs 3 --iters 15 --compile

# 2. контроль: тот же сид и тот же фолд, что уже посчитаны в eager на A10.
#    Нужен, чтобы --compile не оказался систематическим сдвигом качества.
python -m src.seq fold --val 2025-09-04 --epochs 4 --seed 42 --exp SEQ-01C-S42 --compile

# 3. сид 43 на трёх недостающих фолдах (фолд 2025-10-16 уже посчитан в exp_025).
#    Веса фолдов теперь сохраняются: без них диагностика по глубине невозможна.
for V in 2025-09-04 2025-09-18 2025-10-02; do
  python -m src.seq fold --val "$V" --epochs 4 --seed 43 --compile
done
python -m src.seq merge --exp SEQ-01-S43 --desc "SEQ-01, сид 43"

# 4. ЭТАП 1 — чувствительность к глубине истории, inference-only, без переобучения.
#    Сетка дополняется максимальной глубиной, прожитой моделью на обучении.
for P in SEQ-01-S43-V0904 SEQ-01-S43-V0918 SEQ-01-S43-V1002; do
  python -m src.seq depth --ckpt "$P" --depths 180 220 254 289
done

# 5. ЭТАП 2 — среднее двух сидов в лог-пространстве
python -m src.seq avg --exp SEQ-AVG2 --seeds 42 43

# 6. диагностики: пофолдово, AUC, сегменты, разнообразие, разложение шум/функция
python -m src.ptime_eval --exps SEQ-01-S42 SEQ-01-S43 SEQ-AVG2 --base S1-ROUNDS \
  --out research/strategies/results/SEQ2
PYTHONPATH=. python research/strategies/results/SEQ2/analyze_avg.py \
  --seeds SEQ-01-S42 SEQ-01-S43 --avgs SEQ-AVG2

# 7. честный LOFO со страховкой S1-E03a, зафиксированной на 0.10 (MIX-E11)
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST SEQ-AVG2 \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo --fix S1-E03a=0.10

# --- ЭТАП 3: гейт после AVG2 пройден (LOFO -0.00160, 4/4) ---------------------
for V in 2025-10-16 2025-10-02 2025-09-18 2025-09-04; do
  python -m src.seq fold --val "$V" --epochs 4 --seed 44 --compile
done
python -m src.seq merge --exp SEQ-01-S44 --desc "SEQ-01, сид 44"
python -m src.seq depth --ckpt SEQ-01-S44-V1016 --depths 150 180 220 254 270
python -m src.seq avg --exp SEQ-AVG3 --seeds 42 43 44

PYTHONPATH=. python research/strategies/results/SEQ2/analyze_avg.py \
  --seeds SEQ-01-S42 SEQ-01-S43 SEQ-01-S44 --avgs SEQ-AVG2 SEQ-AVG3
python -m src.ptime_eval --exps SEQ-AVG2 SEQ-AVG3 --base S1-ROUNDS \
  --out research/strategies/results/SEQ2
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST SEQ-AVG3 \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo --fix S1-E03a=0.10

# --- тест и сабмит ------------------------------------------------------------
# Глубина ПОЛНАЯ (--depth-clip 0) — по результату этапа 1. Сид 42 на полной
# глубине уже посчитан в exp_025 как ztest_SEQ-01-FULL.
for S in 43 44; do
  python -m src.seq predict --exp "SEQ-S$S" --seed "$S" --epochs 4 --depth-clip 0 --compile
done
# доля SEQ 0.45 набирается тремя сидами по 0.15; страховка S1-CAP = 0.10 сохранена
python -m src.submit --z S1-NORM S1-UNC S1-CAP S1-DIST SEQ-01-FULL SEQ-S43-FULL SEQ-S44-FULL \
  --weights 0.10 0.15 0.10 0.20 0.15 0.15 0.15 --level 2.3293 \
  --out submission_SEQAVG3_mix.csv
