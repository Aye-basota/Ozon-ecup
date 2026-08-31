#!/usr/bin/env bash
# ETX1 — сборка и проверка сабмита `ETX-SEQ-MIX` (`exp_036`).
#
# Кандидат: `CAP + UNC + DIST + слот SEQ`, где слот = 0.5·ETX + 0.5·SEQ-AVG3.
#   0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 ETX + 0.225 SEQ-AVG3 = 1.00
# Доля `SEQ-AVG3` 0.225 набирается тремя сидами по 0.075: `src.submit` усредняет
# В ЛОГ-ПРОСТРАНСТВЕ, поэтому три веса по 0.075 — это в точности `0.225 ·`
# лог-среднее сидов, тот же объект, что `SEQ-AVG3` в OOF.
#
# Веса слота — ФИКСИРОВАННЫЕ 0.10/0.20/0.25/0.45, не оптимум поиска. `MIX-E11`:
# подбор ради 4-го знака дал локальные −0.00038 и +0.00023 на LB. Честный LOFO
# этой конфигурации: **−0.00091 на 4/4** к отправленному `SEQ-01-MIX` (1.6501764),
# прежний рекорд слота −0.00061.
#
# ВСЕ тестовые прогнозы SEQ и ETX — при `--depth-clip 289` (`exp_027`; у ETX цена
# экстраполяции по глубине ещё выше, `exp_036`: обрезка до 180д стоит +0.01259
# против +0.00841 у TCN). `ztest_*-FULL.npy` в сабмит не идут никогда.
set -eu
export PYTHONIOENCODING=utf-8
OUT=${OUT:-submission_ETX_SEQ_mix.csv}
Z="S1-CAP S1-UNC S1-DIST SEQ-01 SEQ-C289-S43 SEQ-C289-S44 ETX-01-S42"
W="0.10 0.20 0.25 0.075 0.075 0.075 0.225"

for n in $Z; do
  test -s "artifacts/ztest_${n}.npy" || { echo "НЕТ artifacts/ztest_${n}.npy"; exit 1; }
done

# вменяемость тестовой модели ETX ДО сборки: обрезка применена, уровень на месте
PYTHONPATH=. python research/strategies/results/ETX1/check_test_etx.py

python -m src.submit --z $Z --weights $W --level 2.3293 --out "$OUT"
PYTHONPATH=. python research/strategies/results/MIX9/verify_submission.py \
    --csv "$OUT" --z $Z --weights $W --level 2.3293
