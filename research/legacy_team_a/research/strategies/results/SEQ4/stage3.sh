#!/usr/bin/env bash
# SEQ-03A этап 3 — победитель на всех четырёх фолдах, ОДИН сид.
#
# Запускается только если гейт этапа 1 пройден на 10-16 (availprobe <= +0.0010
# при normal RMSLE не хуже BASE более чем на +0.0005). Фолды 10-16 (этап 1) и
# 09-04 (этап 2) уже посчитаны, здесь досчитываются два оставшихся.
#
# Сиды 43/44 НЕ запускаются, пока 4-фолдовый результат одного сида не покажет
# >=3/4 фолдов с обязательным 10-16 и ΔwCV <= -0.0005.
#
# Использование: bash stage3.sh <ТЕГ-ПОБЕДИТЕЛЯ>
set -eu
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../../.."
WIN=${1:?нужен тег победителя этапа 1}
TAG="SEQ-03A-$WIN-S42"

case "$WIN" in
  B)   AUG="--aug avail_bnd --aug-p 0.5 --aug-full 0.5" ;;
  C)   AUG="--aug no_avail" ;;
  A50) AUG="--aug avail_drop --aug-p 0.5" ;;
  A25) AUG="--aug avail_drop --aug-p 0.25" ;;
  *) echo "неизвестный тег $WIN" >&2; exit 1 ;;
esac

for V in 2025-09-18 2025-10-02; do
  S=$(echo "$V" | tr -d '-' | cut -c5-8)
  [ -f "artifacts/oof_${TAG}-V${S}.npz" ] && { echo "== $TAG V$S уже есть"; continue; }
  echo "== $TAG $V $(date +%H:%M:%S)"
  python -m src.seq fold --val "$V" --epochs 4 --seed 42 --exp "$TAG" $AUG \
    > "artifacts/SEQ4_${TAG}_V${S}.log" 2>&1
  tail -3 "artifacts/SEQ4_${TAG}_V${S}.log"
done

# availprobe на каждом фолде: гейт обязан держаться не только на 10-16
for S in 0904 0918 1002 1016; do
  python -m src.seq availprobe --ckpt "${TAG}-V${S}" > "artifacts/SEQ4_probe_${WIN}_V${S}.log" 2>&1
  tail -2 "artifacts/SEQ4_probe_${WIN}_V${S}.log"
done

python -m src.seq merge --exp "$TAG" --desc "SEQ-03A вариант $WIN: аугментация границы avail"

# ансамблевая ценность — тот же честный LOFO, что в exp_026/exp_027,
# со страховкой S1-E03a = 0.10 (обнулять запрещено, MIX-E11)
python -m src.blend --exps S1-E10 S1-E02 S1-E03a S1-DIST "$TAG" \
  --ref 0.15 0.30 0.10 0.45 0.0 --lofo --fix S1-E03a=0.10 \
  | tee "artifacts/SEQ4_lofo_${WIN}.log"
echo "== этап 3 закончен $(date +%H:%M:%S)"
