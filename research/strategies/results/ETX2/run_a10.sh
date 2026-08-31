#!/usr/bin/env bash
# ETX2 — очередь A10 (`vm9161`) под `ETX-AVG3` (EXP-037).
#
# Конфиг ETX не меняется ни в одной оси, кроме `--seed`: d_model 128, blocks 5,
# heads 8, head_dim 16, ffn 384, dropout 0.10, n_tok 192, batch 512, chunk 128,
# lr 1.5e-3, wd 1e-2, epochs 4, warmup 500 — всё из `DEFAULT_CFG`, ровно как у
# `ETX-01-S42` (`exp_036`). Отличие среды: A10 + `--compile` (на Windows нет
# Triton, поэтому сид 42 считался eager). Замер шага: eager 5 708 примеров/с,
# compile 8 611 против 2 901 локально.
#
# Порядок неслучаен: сначала ДВЕ ТЕСТОВЫЕ модели, потому что решающий вопрос
# EXP-037 — снимает ли усреднение сидов аномалию `Var(z_ETX − z_SEQ)` на тестовом
# cutoff'е (3.22x, `exp_036`), — отвечается тестовой стороной, и ответ «нет»
# закрывает ветку раньше, чем досчитаются фолды.
#
# Тест ВСЕГДА при `--depth-clip 289` (`exp_027`, «Не повторять»).
set -u
cd /root/ecup
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
L=artifacts

run_test() {  # $1 = seed
  local S=$1 EXP="ETX-01-S$1"
  if [ -s "$L/ztest_${EXP}.npy" ]; then echo "== ПРОПУСК test $EXP"; return; fi
  echo "== СТАРТ test $EXP $(date -u '+%F %T')"
  python3 -m src.etx predict --exp "$EXP" --seed "$S" --depth-clip 289 --compile \
      > "$L/ETX2_test_${EXP}.log" 2>&1
  echo "== ГОТОВО test $EXP rc=$? $(date -u '+%F %T')"; tail -n 5 "$L/ETX2_test_${EXP}.log"
}

run_fold() {  # $1 = seed, $2 = val date
  local S=$1 V=$2 EXP="ETX-01-S$1" P
  P="ETX-01-S$1-V$(date -d "$2" +%m%d)"
  if [ -s "$L/oof_${P}.npz" ]; then echo "== ПРОПУСК fold $P"; return; fi
  echo "== СТАРТ fold $P $(date -u '+%F %T')"
  python3 -m src.etx fold --val "$V" --exp "$EXP" --seed "$S" --curve --compile \
      > "$L/ETX2_${P}.log" 2>&1
  echo "== ГОТОВО fold $P rc=$? $(date -u '+%F %T')"; tail -n 6 "$L/ETX2_${P}.log"
}

JOBS=${JOBS:-"T43 T44 F43:2025-10-16 F43:2025-10-02 F43:2025-09-18 F43:2025-09-04 F44:2025-10-16"}
for J in $JOBS; do
  case "$J" in
    T*)  run_test "${J#T}" ;;
    F*)  run_fold "$(echo "${J#F}" | cut -d: -f1)" "$(echo "$J" | cut -d: -f2)" ;;
  esac
done
touch "$L/ETX2_A10_ALLDONE"
echo "== ETX2 A10 ALL_DONE $(date -u '+%F %T')"
