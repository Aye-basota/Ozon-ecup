#!/usr/bin/env bash
# EXP-038 (FNL) — пилот: фолд 2025-10-16, сид 42, 7 прогонов ПОСЛЕДОВАТЕЛЬНО.
#
# Порядок — по информативности, а не по симметрии: сначала BASE и главная пара
# FUNNEL/BUYCTRL при lam=0.3, потому что именно она отвечает на вопрос
# эксперимента («новый источник или просто вторая голова»); затем CART, который
# отделяет Search от Cart; реплика lam=0.1 идёт последней и нужна лишь затем,
# чтобы вывод не зависел от единственного значения веса.
#
# Строго последовательно: два процесса, читающих плотную панель, не помещаются в
# ~16.8 ГБ свободной RAM (память проекта `ecup-seq-gpu-budget`). Прогон, у
# которого OOF уже на диске, пропускается — очередь перезапускаема.
#
# ЧИСЛО ЭПОХ ЗАДАЁТСЯ ЯВНО. `seq.DEFAULT_CFG` содержит `epochs=3`, а
# подтверждённый рецепт `SEQ-D3A` (`exp_030c`, `SEQ7/run_g1.sh`) — 4 эпохи.
# Первый запуск очереди ушёл на 3 эпохи именно потому, что полагался на
# умолчание; поэтому здесь все ключевые ручки рецепта проставлены руками, а
# после каждого прогона стоит проверка, что в логе действительно «эпоха 4/4».
set -u
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8
L=artifacts
V=2025-10-16
EPOCHS=4
SEED=42
DEPTH_AUG=0.5

run () {          # run <arm> <lam-в-процентах>
  local arm=$1 pct=$2
  local lam
  lam=$(awk "BEGIN{printf \"%.2f\", $pct/100}")
  local exp
  exp=$(printf 'FNL-%s-L%02d-S%d' "$arm" "$pct" "$SEED")
  local tag="${exp}-V1016"
  if [ -f "$L/oof_${tag}.npz" ]; then echo "== ПРОПУСК ${tag}: OOF уже есть"; return 0; fi
  echo "== СТАРТ ${tag} lam=${lam} epochs=${EPOCHS} $(date '+%F %T')"
  python -m src.fnl fold --val "$V" --arm "$arm" --lam "$lam" --exp "$exp" \
      --epochs "$EPOCHS" --seed "$SEED" --depth-aug "$DEPTH_AUG" --curve \
      > "$L/FNL1_${tag}.log" 2>&1
  local rc=$?
  if ! grep -q "эпоха ${EPOCHS}/${EPOCHS}" "$L/FNL1_${tag}.log"; then
    echo "!! РЕЦЕПТ ${tag}: в логе нет «эпоха ${EPOCHS}/${EPOCHS}» — прогон невалиден"
  fi
  echo "== ГОТОВО ${tag} rc=${rc} $(date '+%F %T')"
  tail -n 8 "$L/FNL1_${tag}.log"
}

run BASE    0
run FUNNEL  30
run BUYCTRL 30
run CART    30
run FUNNEL  10
run BUYCTRL 10
run CART    10
echo "== FNL1 ALL_DONE $(date '+%F %T')"
