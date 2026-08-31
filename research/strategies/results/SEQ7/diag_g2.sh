#!/usr/bin/env bash
# EXP-030 multi-seed, GPU-B (сид 44) — диагностика глубины. ТОЛЬКО inference на
# сохранённых весах, ни одна модель не переобучается.
#
# Порядок — по убыванию ценности, потому что аренда жёсткая и хвост может не влезть:
#   1. `crossdepth` V0904 -> 2025-10-16: ровно тот режим +77 дней, что на тесте.
#      Именно здесь `exp_030` намерил рост выигрыша в 2.4 раза на одном сиде, и
#      именно это `exp_030b` просил перепроверить на трёх сидах;
#   2. `depth` на гейтовом фолде 10-16: польза длинной истории на своём фолде;
#   3. `availprobe` на 10-16: цена непрожитого режима `avail = 1`. У curriculum
#      она обязана ОСТАТЬСЯ — приём добавляет нули, а не убирает их;
#   4. `depth` на остальных фолдах.
set -u
cd /root/ecup
export PYTHONIOENCODING=utf-8
L=artifacts
P=G2
BASE=SEQ-D3A-G2-BASE-S44
D3A=SEQ-D3A-G2-S44
DEPTHS="150 180 212 240 254 275 289"

have () { [ -f "$L/model_$1.pt" ]; }

for E in $BASE $D3A; do
  if have "${E}-V0904"; then
    echo "== crossdepth ${E}-V0904 -> 2025-10-16 $(date -u '+%T')"
    python3 -m src.seq crossdepth --ckpt "${E}-V0904" --val 2025-10-16 \
        --depths 212 230 247 261 275 289 > "$L/${P}_xdepth_${E}.log" 2>&1
    tail -n 3 "$L/${P}_xdepth_${E}.log"
  fi
done
for E in $BASE $D3A; do
  if have "${E}-V1016"; then
    echo "== depth ${E}-V1016 $(date -u '+%T')"
    python3 -m src.seq depth --ckpt "${E}-V1016" --depths $DEPTHS \
        > "$L/${P}_depth_${E}-V1016.log" 2>&1
    tail -n 2 "$L/${P}_depth_${E}-V1016.log"
    echo "== availprobe ${E}-V1016 $(date -u '+%T')"
    python3 -m src.seq availprobe --ckpt "${E}-V1016" \
        > "$L/${P}_availprobe_${E}.log" 2>&1
    tail -n 2 "$L/${P}_availprobe_${E}.log"
  fi
done
for T in V1002 V0918 V0904; do
  for E in $BASE $D3A; do
    have "${E}-${T}" || continue
    echo "== depth ${E}-${T} $(date -u '+%T')"
    python3 -m src.seq depth --ckpt "${E}-${T}" --depths $DEPTHS \
        > "$L/${P}_depth_${E}-${T}.log" 2>&1
    tail -n 2 "$L/${P}_depth_${E}-${T}.log"
  done
done
echo "== ${P} DIAG_DONE $(date -u '+%F %T')"
