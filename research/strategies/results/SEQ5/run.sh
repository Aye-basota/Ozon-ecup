#!/usr/bin/env bash
# EXP-030 `SEQ-D3A` — depth curriculum, seed 42, все 4 фолда.
#
# Что делается и почему именно так:
#
#  * BASE и D3A отличаются РОВНО одним флагом `--depth-aug 0.5`. Архитектура,
#    target, оптимизатор, число эпох, seed и порядок батчей те же (отдельный
#    поток случайности аугментации, `src/test_seq.py`).
#  * BASE фолда 2025-10-16 НЕ переобучается: это побитово тот же прогон
#    `SEQ-03A-BASE-S42-V1016` из `exp_029` (тот же коммит-код обучающего пути,
#    seed 42, локальный eager, epochs 4, aug=none). Файлы скопированы под именем
#    `SEQ-D3A-BASE-S42-V1016`. Переобучение дало бы ДРУГОЕ число из-за
#    недетерминизма GPU, то есть было бы хуже, а не честнее.
#  * Порядок: сначала гейтовый фолд 10-16 (у него вес 8 из 15 в wCV и BASE уже
#    есть), затем пары BASE/D3A от поздних фолдов к ранним. Так решение о
#    3 сидах становится видно раньше, чем закончится весь прогон.
#
# Оценка времени: ~196 с на обучающий cutoff, всего 84 + 60 = 144 cutoff-эквивалента,
# то есть примерно 7.9 ч на RTX 4060 Ti в eager-режиме.
#
# Запуск: bash research/strategies/results/SEQ5/run.sh
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
  python -m src.seq fold --val "$val" --epochs 4 --seed 42 --exp "$exp" "$@" \
      > "$L/SEQ5_${tag}.log" 2>&1
  echo "== ГОТОВО ${tag} $(date '+%F %T') rc=$?"
  tail -n 8 "$L/SEQ5_${tag}.log"
}

BASE=SEQ-D3A-BASE-S42
D3A=SEQ-D3A-S42

run $D3A  2025-10-16 --depth-aug 0.5
run $BASE 2025-10-02
run $D3A  2025-10-02 --depth-aug 0.5
run $BASE 2025-09-18
run $D3A  2025-09-18 --depth-aug 0.5
run $BASE 2025-09-04
run $D3A  2025-09-04 --depth-aug 0.5

echo "== все фолды посчитаны $(date '+%F %T')"
