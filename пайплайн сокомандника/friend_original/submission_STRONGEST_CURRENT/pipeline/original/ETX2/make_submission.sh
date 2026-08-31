#!/usr/bin/env bash
# ETX2 — сборка и проверка сильнейшего кандидата (`EXP-037`).
#
#   ALPHA=0.5 OUT=submission_STRONGEST_CURRENT.csv bash research/.../ETX2/make_submission.sh
#
# Рецепт: `CAP + UNC + DIST + слот SEQ`, слот = `ALPHA·ETX-AVG3 + (1−ALPHA)·SEQ-AVG3`.
# Веса смеси ФИКСИРОВАННЫЕ 0.10 / 0.20 / 0.25 / 0.45 — те же, что у `SEQ-AVG3-CLIP-MIX`
# (`exp_035`) и у кандидата `exp_036`; не оптимум поиска (`MIX-E11`: подбор ради
# 4-го знака дал локальные −0.00038 и +0.00023 на LB). `S1-CAP` = 0.10 сохранён.
#
# Доли внутри слота набираются ПОСИДОВО: `src.submit` усредняет В ЛОГ-ПРОСТРАНСТВЕ,
# поэтому три веса по `0.45·ALPHA/3` — это в точности `0.45·ALPHA ·` лог-среднее
# сидов, тот же объект, что `ETX-AVG3` в OOF.
#
# ТЕСТОВАЯ СТОРОНА ETX — только `*-DCW`: `--depth-clip 289` И статик query-токена,
# приведённый в обученный диапазон (глубина 289 вместо 365, день недели cutoff'а =
# четверг вместо пятницы). Обоснование и замеры на РАЗМЕЧЕННЫХ фолдах —
# `depth_fix.py`, `exp_037`. Сырые `ztest_ETX-01-S4?.npy` (статик 365) в сабмит
# не идут никогда: у них `Var(z−z_SEQ-AVG3)` на тесте 3.22x к OOF.
set -eu
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../../.."
ALPHA=${ALPHA:-0.5}
OUT=${OUT:-submission_STRONGEST_CURRENT.csv}
LEVEL=${LEVEL:-2.3293}

W_SEQ=$(python -c "print(round(0.45*(1-$ALPHA)/3, 6))")
W_ETX=$(python -c "print(round(0.45*$ALPHA/3, 6))")
Z="S1-CAP S1-UNC S1-DIST SEQ-01 SEQ-C289-S43 SEQ-C289-S44 ETX-01-S42-DCW ETX-01-S43-DCW ETX-01-S44-DCW"
W="0.10 0.20 0.25 $W_SEQ $W_SEQ $W_SEQ $W_ETX $W_ETX $W_ETX"

for n in $Z; do
  test -s "artifacts/ztest_${n}.npy" || { echo "НЕТ artifacts/ztest_${n}.npy"; exit 1; }
done
echo "слот SEQ: ALPHA=$ALPHA -> по $W_ETX на сид ETX, по $W_SEQ на сид TCN"

PYTHONPATH=. python research/strategies/results/ETX2/check_test_etx2.py
python -m src.submit --z $Z --weights $W --level "$LEVEL" --out "$OUT"
PYTHONPATH=. python research/strategies/results/MIX9/verify_submission.py \
    --csv "$OUT" --z $Z --weights $W --level "$LEVEL"
sha256sum "submissions/$OUT"
