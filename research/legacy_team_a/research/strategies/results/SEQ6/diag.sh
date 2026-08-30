#!/usr/bin/env bash
# EXP-030b `SEQ-D3A-S43` — диагностика глубины на сиде 43, фолд 2025-09-18.
# Только inference на сохранённых весах, ни одна модель не переобучается.
#
# Что можно и чего нельзя на одном фолде:
#  * `depth`      — МОЖНО: кривая RMSLE по глубине на своём фолде и «gain +49»
#                   (cal@212 -> cal@261, полная глубина 09-18 = 261 день).
#                   Именно здесь на сиде 42 дельта D3A-BASE была КОНСТАНТОЙ по
#                   глубине (+0.0015..+0.0039), то есть провал 09-18 не глубинный.
#  * `availprobe` — МОЖНО на панели своего фолда: цена бита `avail ≡ 1`.
#  * `crossdepth` — НЕЛЬЗЯ: нужен чекпойнт фолда 09-04, а сид 43 обучен только
#                   на 09-18. Не подменять чекпойнтом сида 42 — это другой прогон.
#
# Запуск: bash research/strategies/results/SEQ6/diag.sh
set -u
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8
L=artifacts
DEPTHS="90 120 150 180 212 220 240 254 261"

for E in SEQ-D3A-BASE-S43 SEQ-D3A-S43; do
  T=V0918
  [ -f "$L/model_${E}-${T}.pt" ] || { echo "нет весов ${E}-${T}, пропуск"; continue; }
  echo "== depth ${E}-${T}"
  python -m src.seq depth --ckpt "${E}-${T}" --depths $DEPTHS \
      > "$L/SEQ6_depth_${E}-${T}.log" 2>&1
  tail -n 3 "$L/SEQ6_depth_${E}-${T}.log"
  echo "== availprobe ${E}-${T}"
  python -m src.seq availprobe --ckpt "${E}-${T}" \
      > "$L/SEQ6_availprobe_${E}.log" 2>&1
  tail -n 3 "$L/SEQ6_availprobe_${E}.log"
done
echo "== диагностика SEQ6 готова $(date '+%F %T')"
