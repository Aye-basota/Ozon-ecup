#!/usr/bin/env bash
# Диагностика одного варианта SEQ-03A на фолде 10-16 — только inference.
#
#   availprobe  — одна точка: `avail` как в обучении -> `avail ≡ 1` (код exp_027);
#   availcurve  — вся кривая по ПОЛОЖЕНИЮ границы. Сетка сдвигов подобрана так,
#                 чтобы остаток нулей прошёл 76 / 63 / 50 / 38 / 25 / 12 / 0:
#                 76 — это край, реально прожитый самым глубоким обучающим
#                 cutoff'ом (2025-10-16), 0 — режим теста на полной глубине.
#                 Нужно видеть, деградация начинается сразу за train support или
#                 только у самого `avail ≡ 1`.
#
# Ждёт выхода процесса обучения: две панели по 2.9 ГБ в RAM не помещаются.
set -eu
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../../.."
TAG="SEQ-03A-${1:?нужен тег варианта}-S42"
P="${TAG}-V1016"

until [ -f "artifacts/oof_${P}.npz" ]; do sleep 20; done
while ps -W 2>/dev/null | grep -q "WindowsApps.python"; do sleep 10; done
sleep 5

echo "== $P обучен, диагностика $(date +%H:%M:%S)"
tail -6 "artifacts/SEQ4_${TAG}.log"
python -m src.seq availprobe --ckpt "$P" 2>&1 | tail -4
python -m src.seq availcurve --ckpt "$P" --shifts 0 13 26 38 51 64 76 2>&1 | tail -10
echo "== диагностика $P готова $(date +%H:%M:%S)"
