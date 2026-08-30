#!/usr/bin/env bash
# SEQ-03A этап 1 — ОДИН вариант на диагностическом фолде 2025-10-16, сид 42.
#
# Меняется ровно одна вещь: как канал `avail` подаётся на обучении. Архитектура,
# ёмкость, таргет, эпохи, пулинг, сетка cutoff'ов и панели — как в exp_026.
# Опора — BASE: он обязан воспроизвести exp_025 (фолд 10-16, сид 42, eager,
# калибр. RMSLE 1.74704), иначе сравнивать не с чем.
#
# Варианты запускаются ПО ОДНОМУ и с ручным гейтом между ними: фолд стоит 86
# минут локальной 4060 Ti, и очередь без проверки промежуточного результата
# сжигает GPU на вариантах, которые уже не нужны. Порядок: BASE -> B -> C,
# дальше A только если B/C не дали ясного ответа.
#
# Использование: bash stage1.sh BASE|B|C|A50|A25
set -eu
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../../.."
V=${1:?нужен тег варианта: BASE|B|C|A50|A25}

case "$V" in
  BASE) AUG="--aug none" ;;
  B)    AUG="--aug avail_bnd --aug-p 0.5 --aug-full 0.5" ;;
  C)    AUG="--aug no_avail" ;;
  A50)  AUG="--aug avail_drop --aug-p 0.5" ;;
  A25)  AUG="--aug avail_drop --aug-p 0.25" ;;
  *) echo "неизвестный вариант $V" >&2; exit 1 ;;
esac

TAG="SEQ-03A-$V-S42"
if [ -f "artifacts/oof_${TAG}-V1016.npz" ]; then
  echo "== $TAG уже посчитан"; exit 0
fi
echo "== $TAG старт $(date +%H:%M:%S)  [$AUG]"
python -m src.seq fold --val 2025-10-16 --epochs 4 --seed 42 --exp "$TAG" $AUG \
  > "artifacts/SEQ4_${TAG}.log" 2>&1
tail -4 "artifacts/SEQ4_${TAG}.log"

# те же две диагностики, что у всех вариантов, сразу после обучения
python -m src.seq availprobe --ckpt "${TAG}-V1016" 2>&1 | tail -4
python -m src.seq availcurve --ckpt "${TAG}-V1016" --shifts 0 13 26 38 51 64 76 2>&1 | tail -10
echo "== $TAG готов $(date +%H:%M:%S)"
