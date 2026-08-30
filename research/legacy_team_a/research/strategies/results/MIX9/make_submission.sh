#!/usr/bin/env bash
# MIX9 — сборка и проверка финального сабмита. Запускать ЛОКАЛЬНО после
# `push_pull.sh pull`, когда приехали ztest сидов 43/44.
#
# Кандидат: `CAP + E02 + DIST + SEQ-AVG3 @ clip289` = 0.10 / 0.20 / 0.25 / 0.45.
# Доля SEQ 0.45 набирается тремя сидами по 0.15: `src.submit` усредняет В
# ЛОГ-ПРОСТРАНСТВЕ, поэтому три веса по 0.15 — это в точности `0.45 ·` лог-среднее
# сидов, тот же объект, что `SEQ-AVG3` в OOF.
#
# Имена тестовых вариантов против имён OOF (`holiday_yoy.py` TEST_VARIANTS):
#   S1-E03a -> S1-CAP,  S1-E02 -> S1-UNC,  S1-DIST -> S1-DIST.
# `S1-E10`/`S1-NORM` в этот сабмит НЕ входит: `exp_027` §4 — он избыточен, без
# него тот же честный LOFO при вдвое более устойчивых весах.
set -eu
export PYTHONIOENCODING=utf-8
OUT=${OUT:-submission_SEQAVG3_clip289_mix.csv}
Z="S1-CAP S1-UNC S1-DIST SEQ-01 SEQ-C289-S43 SEQ-C289-S44"
W="0.10 0.20 0.25 0.15 0.15 0.15"

for n in $Z; do
  test -s "artifacts/ztest_${n}.npy" || { echo "НЕТ artifacts/ztest_${n}.npy"; exit 1; }
done

# вменяемость новых тестовых моделей ДО сборки: обрезка применена, сиды согласованы
PYTHONPATH=. python research/strategies/results/MIX9/check_test_seeds.py

python -m src.submit --z $Z --weights $W --level 2.3293 --out "$OUT"
PYTHONPATH=. python research/strategies/results/MIX9/verify_submission.py \
    --csv "$OUT" --z $Z --weights $W --level 2.3293
