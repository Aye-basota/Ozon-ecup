#!/usr/bin/env bash
# ETX2 — очередь локальной RTX 4060 Ti под `ETX-AVG3` (EXP-037).
#
# Здесь считаются ТОЛЬКО фолды сида 44 09-04/09-18/10-02: карта втрое медленнее
# A10 (2 901 против 8 611 примеров/с), поэтому ей отданы три самых дешёвых фолда,
# а всё остальное — на A10 (`run_a10.sh`). Пересечения по именам артефактов между
# машинами нет: локально пишутся `*-S44-V0904/0918/1002`, на A10 — всё прочее.
#
# Конфиг из `DEFAULT_CFG`, ровно как у `ETX-01-S42` (`exp_036`), отличие одно —
# `--seed 44`. `--compile` не используется: на Windows нет Triton.
set -u
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
L=artifacts

for V in ${1:-2025-09-04} ${2:-2025-09-18} ${3:-2025-10-02}; do
  P="ETX-01-S44-V$(echo "$V" | tr -d '-' | cut -c5-8)"
  if [ -s "$L/oof_${P}.npz" ]; then echo "== ПРОПУСК $P"; continue; fi
  echo "== СТАРТ $P $(date -u '+%F %T')"
  python -m src.etx fold --val "$V" --exp ETX-01-S44 --seed 44 --curve       > "$L/ETX2_${P}.log" 2>&1
  echo "== ГОТОВО $P rc=$? $(date -u '+%F %T')"
  tail -n 6 "$L/ETX2_${P}.log"
done
touch "$L/ETX2_LOCAL_ALLDONE"
echo "== ETX2 LOCAL ALL_DONE $(date -u '+%F %T')"
