#!/usr/bin/env bash
# EXP-030 multi-seed, GPU-A (A10 #1, vm8966) — СИД 43, все 4 фолда, парно BASE vs D3A.
#
# Отличие от `SEQ5/run.sh`: `--seed 43` и `--compile` (65.2 мс/шаг против 129.9
# eager, замер `seq bench` на этой же карте). Компиляция включена У ОБОИХ членов
# пары, поэтому контраст приёма остаётся внутрисредовым; абсолютные уровни с
# локальными eager-прогонами `exp_030` не сравниваются — сравниваются ТОЛЬКО
# пофолдовые дельты.
#
# Порядок: пары по убыванию веса фолда в wCV (8:4:1:2 -> 10-16, 10-02, 09-04, 09-18),
# внутри пары сначала D3A, затем BASE. Фолд полезен только целой парой, поэтому
# аренда режется по границам пар.
set -u
cd /root/ecup
export PYTHONIOENCODING=utf-8
L=artifacts
SEED=43
BASE=SEQ-D3A-G1-BASE-S43
D3A=SEQ-D3A-G1-S43

run () {          # run <exp> <val> [доп. флаги]
  local exp=$1 val=$2; shift 2
  local tag="${exp}-V$(echo "$val" | cut -c6-7)$(echo "$val" | cut -c9-10)"
  if [ -f "$L/oof_${tag}.npz" ]; then echo "== ПРОПУСК ${tag}: OOF уже есть"; return 0; fi
  echo "== СТАРТ ${tag} $(date -u '+%F %T')"
  python3 -m src.seq fold --val "$val" --epochs 4 --seed $SEED --compile --exp "$exp" "$@" \
      > "$L/G1_${tag}.log" 2>&1
  echo "== ГОТОВО ${tag} $(date -u '+%F %T') rc=$?"
  tail -n 6 "$L/G1_${tag}.log"
}

for V in 2025-10-16 2025-10-02 2025-09-04 2025-09-18; do
  run $D3A  "$V" --depth-aug 0.5
  run $BASE "$V"
done
echo "== G1 ALL_DONE $(date -u '+%F %T')"
