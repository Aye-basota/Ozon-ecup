#!/usr/bin/env bash
# EXP-030b `SEQ-D3A-S43` — разделяющий замер: ТОЛЬКО фолд 2025-09-18, seed 43.
#
# Вопрос: провал +0.00355 на 09-18 (seed 42, `exp_030`) — эффект приёма или бросок сида?
#
# Меняется РОВНО одна величина против `SEQ5/run.sh`: `--seed 43` вместо 42.
# Всё остальное побитово то же: 4 эпохи, DEFAULT_CFG, депт-сетка по умолчанию,
# единственное отличие варианта от BASE — флаг `--depth-aug 0.5`.
#
# Оценка времени: ~59 мин на прогон (по логам SEQ5 V0918), два прогона ~2 ч.
# Последовательно: две панели одновременно не влезают в RAM.
#
# Запуск: bash research/strategies/results/SEQ6/run.sh
set -u
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8
L=artifacts

run () {          # run <exp> <val> [доп. флаги]
  local exp=$1 val=$2; shift 2
  local tag="${exp}-V$(echo "$val" | cut -c6-7)$(echo "$val" | cut -c9-10)"
  if [ -f "$L/oof_${tag}.npz" ]; then
    echo "== ПРОПУСК ${tag}: OOF уже есть"
    return 0
  fi
  echo "== СТАРТ ${tag} $(date '+%F %T')"
  python -m src.seq fold --val "$val" --epochs 4 --seed 43 --exp "$exp" "$@" \
      > "$L/SEQ6_${tag}.log" 2>&1
  echo "== ГОТОВО ${tag} $(date '+%F %T') rc=$?"
  tail -n 8 "$L/SEQ6_${tag}.log"
}

run SEQ-D3A-BASE-S43 2025-09-18
run SEQ-D3A-S43      2025-09-18 --depth-aug 0.5

echo "== SEQ6 фолд посчитан $(date '+%F %T')"
