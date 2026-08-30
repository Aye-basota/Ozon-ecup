#!/usr/bin/env bash
# Забрать артефакты фолдов с арендованной A10 и посчитать всё, что считается на CPU.
# Обучение идёт на сервере, а OOF боевых табличных моделей (S1-*) лежат только
# локально, поэтому blend/ptime_eval/analyze_avg запускаются здесь.
set -eu
export PYTHONIOENCODING=utf-8
HOST=${HOST:-root@185.182.108.210}
SEEDS=${SEEDS:-"42 43"}
AVG=${AVG:-SEQ-AVG2}
SRCS=${SRCS:-"SEQ-01-S42 SEQ-01-S43"}

# 1. OOF, веса фолдов и кривые эпох
for f in oof_SEQ-01-S4[34]-V*.npz model_SEQ-01-S4[34]-V*.pt curve_SEQ-01-S4[34]-V*.json \
         depth_SEQ-01-S4[34]-V*.csv; do
  scp -q "$HOST:/root/ecup/artifacts/$f" artifacts/ 2>/dev/null || true
done
ls -la artifacts/ | grep -E "S4[34]" || true

# 2. склейка сидов, у которых есть все четыре фолда
for s in $SEEDS; do
  [ "$s" = 42 ] && continue                     # SEQ-01-S42 уже склеен в exp_025
  python -m src.seq merge --exp "SEQ-01-S$s" --desc "SEQ-01, сид $s"
done

# 3. усреднение в лог-пространстве
python -m src.seq avg --exp "$AVG" --seeds $SEEDS

# 4. сегменты и AUC против лучшей одиночной табличной модели
python -m src.ptime_eval --exps $SRCS "$AVG" --base S1-ROUNDS \
  --out research/strategies/results/SEQ2

# 5. разложение разнообразия на шум сида и устойчивую часть
PYTHONPATH=. python research/strategies/results/SEQ2/analyze_avg.py \
  --seeds $SRCS --avgs "$AVG"

# 6. честный LOFO со страховкой S1-E03a = 0.10 (обнулять запрещено, MIX-E11)
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST "$AVG" \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo --fix S1-E03a=0.10
