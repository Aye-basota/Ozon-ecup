#!/usr/bin/env bash
# EXP-030 `SEQ-D3A` — диагностика глубины. Только inference на сохранённых весах,
# ни одна модель не переобучается.
#
# Три вопроса, на которые обязан ответить прогон (постановка EXP-030 и STOP-сценарий
# `exp_029` «инвариантность куплена отказом от длинной истории»):
#
#  1. `depth`      — где минимум RMSLE по глубине на СВОЁМ фолде и сохранился ли
#                    выигрыш длинной истории (cal@212 -> cal@289, «gain +77»);
#  2. `crossdepth` — модель раннего фолда на поздней панели: ровно тот же режим
#                    экстраполяции +77 дней, что на тесте (`exp_027`);
#  3. `availprobe` — цена ухода канала `avail` в режим `avail ≡ 1`. У curriculum
#                    она обязана ОСТАТЬСЯ (приём добавляет нули, а не убирает их);
#                    если она исчезла — это уже `exp_029`, а не EXP-030.
#
# Запуск: bash research/strategies/results/SEQ5/diag.sh
set -u
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8
L=artifacts
DEPTHS="90 120 150 180 212 220 240 254 275 289"

for E in SEQ-D3A-BASE-S42 SEQ-D3A-S42; do
  for T in V0904 V0918 V1002 V1016; do
    [ -f "$L/model_${E}-${T}.pt" ] || { echo "нет весов ${E}-${T}, пропуск"; continue; }
    echo "== depth ${E}-${T}"
    python -m src.seq depth --ckpt "${E}-${T}" --depths $DEPTHS \
        > "$L/SEQ5_depth_${E}-${T}.log" 2>&1
    tail -n 3 "$L/SEQ5_depth_${E}-${T}.log"
  done
  # +77 дней экстраполяции: ранняя модель (train max 212) на панели 10-16 (289)
  if [ -f "$L/model_${E}-V0904.pt" ]; then
    echo "== crossdepth ${E}-V0904 -> 2025-10-16"
    python -m src.seq crossdepth --ckpt "${E}-V0904" --val 2025-10-16 \
        --depths 212 230 247 261 275 289 > "$L/SEQ5_xdepth_${E}.log" 2>&1
    tail -n 3 "$L/SEQ5_xdepth_${E}.log"
  fi
  # цена бита `avail ≡ 1` на гейтовом фолде
  if [ -f "$L/model_${E}-V1016.pt" ]; then
    echo "== availprobe ${E}-V1016"
    python -m src.seq availprobe --ckpt "${E}-V1016" \
        > "$L/SEQ5_availprobe_${E}.log" 2>&1
    tail -n 3 "$L/SEQ5_availprobe_${E}.log"
  fi
done
echo "== диагностика готова $(date '+%F %T')"
