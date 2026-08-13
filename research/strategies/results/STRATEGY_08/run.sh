#!/usr/bin/env bash
# STRATEGY_08 — личное время как представление. Три варианта против S1-SEEDAVG3.
#
# База (STRATEGY_05 §«Лучший найденный config»): direct, признаки S1-E10 (227),
# --L 0 --norm-long --min-history 90, train_blocks 1, cutoffs all (шаг 7),
# panel_blocks 3, 300 раундов, среднее сидов 42/43/44. wCV 1.7504564.
# Единственное изменение в каждом варианте — набор признаков.
#
#   PT-OD    вариант A: 9 колонок просрочки относительно СВОЕГО распределения интервалов
#   PT-FULL  вариант B: все 30 колонок личного времени
#   PT-SHUF  вариант C: те же 30 колонок, но профиль личного времени переставлен
#            между пользователями (контроль честности)
#
# Фолды считаются двумя процессами по 6 потоков (research/compute_profile.md);
# разбиение {09-04, 10-16} / {09-18, 10-02} выравнивает число обучающих cutoff'ов.
#
# Запуск: bash research/strategies/results/STRATEGY_08/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8

BASE="--L 0 --min-history 90 --norm-long --train-blocks 1 --cutoffs all \
      --panel-blocks 3 --model direct --rounds 300"
REF=S1-SEEDAVG3
LOG=artifacts/ptime
mkdir -p "$LOG"

run_variant () {          # $1 = имя, $2 = --ptime, $3 = --ptime-source, $4 = описание
  local name=$1 pt=$2 src=$3 desc=$4
  for seed in 42 43 44; do
    if [ -f "artifacts/oof_${name}-S${seed}.npz" ]; then
      echo "== ${name}-S${seed}: уже посчитан, пропуск"; continue
    fi
    echo "== ${name}-S${seed}"
    LGB_THREADS=6 python -m src.train --exp "${name}-S${seed}-A" \
      --desc "$desc, seed $seed (фолды 09-04, 10-16)" $BASE --seed "$seed" \
      --ptime "$pt" --ptime-source "$src" --val 2025-09-04 2025-10-16 \
      --no-log --ref "$REF" > "$LOG/${name}-S${seed}-A.log" 2>&1 &
    local pa=$!
    LGB_THREADS=6 python -m src.train --exp "${name}-S${seed}-B" \
      --desc "$desc, seed $seed (фолды 09-18, 10-02)" $BASE --seed "$seed" \
      --ptime "$pt" --ptime-source "$src" --val 2025-09-18 2025-10-02 \
      --no-log --ref "$REF" > "$LOG/${name}-S${seed}-B.log" 2>&1 &
    local pb=$!
    wait $pa; wait $pb
    python -m src.merge_oof --out "${name}-S${seed}" \
      --parts "${name}-S${seed}-A" "${name}-S${seed}-B" --ref "$REF" \
      --desc "$desc, seed $seed" --model direct --no-log \
      > "$LOG/${name}-S${seed}.log" 2>&1
    grep -E "wCV|AUC" "$LOG/${name}-S${seed}.log" | head -4 || true
  done
  python -m src.seedavg --out "${name}-AVG3" \
    --exps "${name}-S42" "${name}-S43" "${name}-S44" --ref "$REF" \
    --desc "$desc, среднее 3 сидов" --n-features "$5" --stats --no-log \
    | tee "$LOG/${name}-AVG3.log" | tail -30
}

run_variant PT-OD   od   real "STRATEGY_08 A: просрочка относительно личного распределения интервалов" 236
run_variant PT-FULL full real "STRATEGY_08 B: полное представление в личном времени (30 колонок)"      257
run_variant PT-SHUF full shuf "STRATEGY_08 C: контроль — профиль личного времени переставлен"          257

echo "готово"
