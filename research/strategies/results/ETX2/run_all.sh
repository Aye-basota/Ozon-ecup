#!/usr/bin/env bash
# ETX2 — весь анализ EXP-037 одной командой, когда обучение закончено.
#
#   bash research/strategies/results/ETX2/run_all.sh
#
# Обучение (10 прогонов, две карты) — `run_a10.sh` и `run_local.sh`; сюда оно
# не входит. Здесь только то, что считается на готовых OOF и ztest.
set -eu
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
cd "$(dirname "$0")/../../../.."
R=research/strategies/results/ETX2

echo "########## 1. ETX-AVG3 и склейка сидов"
python $R/build_etx_avg.py
python $R/summary.py

echo; echo "########## 2. честный LOFO слота SEQ"
python $R/lofo2.py

echo; echo "########## 3. сегментный гейт ETX/TCN"
python $R/segblend.py

echo; echo "########## 4. режим на тестовом cutoff'е"
echo "-- как в exp_036 (сырой статик 365):"
ETX2_DC=0 python $R/regime.py 2>/dev/null | sed -n '1,12p' || true
echo "-- боевая политика (глубина 289 + dow четверг):"
ETX2_DC=DCW python $R/regime.py
