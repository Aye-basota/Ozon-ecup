#!/usr/bin/env bash
# EXP-038 (FNL) — добор упавших прогонов + контроль шума.
#
# `FNL-CART-L30-S42` упал молча на 9-й секунде (rc=1, ни трейсбека, ни записи в
# журнале Windows) сразу после того, как предыдущий прогон отдал свои ~5.6 ГБ
# VRAM и ~6 ГБ RAM. Тот же код в составе `FNL-FUNNEL-L30` отработал полностью,
# поэтому это не ошибка постановки, а гонка за ресурсы на стыке прогонов.
# Лечение — пауза на осадку между прогонами и одна повторная попытка.
#
# `run.sh` перезапускаем по построению: прогон с готовым OOF пропускается,
# поэтому повторный вызов доберёт ровно то, чего не хватает.
set -u
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8
L=artifacts
V=2025-10-16
EPOCHS=4
SEED=42

settle () { echo "-- осадка 90 с $(date '+%F %T')"; sleep 90; }

one () {          # one <arm> <lam-в-процентах>
  local arm=$1 pct=$2 try
  local lam
  lam=$(awk "BEGIN{printf \"%.2f\", $pct/100}")
  local exp
  exp=$(printf 'FNL-%s-L%02d-S%d' "$arm" "$pct" "$SEED")
  local tag="${exp}-V1016"
  for try in 1 2; do
    if [ -f "$L/oof_${tag}.npz" ]; then
      echo "== ПРОПУСК ${tag}: OOF уже есть"; return 0
    fi
    settle
    echo "== СТАРТ ${tag} lam=${lam} попытка ${try} $(date '+%F %T')"
    python -m src.fnl fold --val "$V" --arm "$arm" --lam "$lam" --exp "$exp" \
        --epochs "$EPOCHS" --seed "$SEED" --depth-aug 0.5 --curve \
        > "$L/FNL1_${tag}.log" 2>&1
    local rc=$?
    if ! grep -q "эпоха ${EPOCHS}/${EPOCHS}" "$L/FNL1_${tag}.log"; then
      echo "!! РЕЦЕПТ ${tag}: нет «эпоха ${EPOCHS}/${EPOCHS}», попытка ${try} невалидна"
    fi
    echo "== ГОТОВО ${tag} rc=${rc} попытка ${try} $(date '+%F %T')"
    tail -n 6 "$L/FNL1_${tag}.log"
    [ -f "$L/oof_${tag}.npz" ] && return 0
  done
  echo "!! ${tag} не собрался за две попытки"
}

# Добор всех семи арок очереди — готовые пропускаются мгновенно.
one BASE    0
one FUNNEL  30
one BUYCTRL 30
one CART    30
one FUNNEL  10
one BUYCTRL 10
one CART    10

# Контроль шума прогона: повтор BASE тем же сидом в другом процессе.
settle
bash research/strategies/results/FNL1/run_ctrl.sh
echo "== FNL1 MAKEUP_DONE $(date '+%F %T')"
