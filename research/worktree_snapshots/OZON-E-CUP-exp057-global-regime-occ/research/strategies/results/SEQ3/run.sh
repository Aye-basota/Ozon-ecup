#!/usr/bin/env bash
# exp_027 — диагностика провала SEQAVG3-MIX на LB. Полностью inference-only.
#
# Ничего не обучает: используются чекпойнты фолдов из exp_026, уже посчитанные
# тестовые прогнозы ztest_* и OOF. ~40 минут на RTX 4060 Ti.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHONPATH=.
export PYTHONIOENCODING=utf-8      # в отчётах есть Δ/α, а консоль Windows — cp1251

echo "### анти-лукап"
python -m pytest src/test_seq.py -q

echo "### этап 2: кросс-фолдовый стресс по глубине (ранняя модель -> поздняя панель)"
# 09-04 (обучено до 212) -> 10-16 (доступно 289) = +77, почти точная копия теста (+76)
GRID_0904="212 226 240 247 254 261 268 275 282 289"
for CK in SEQ-01C-S42-V0904 SEQ-01-S43-V0904 SEQ-01-S44-V0904; do
  python -m src.seq crossdepth --ckpt "$CK" --val 2025-10-16 --depths $GRID_0904
done
# 09-18 (226) -> 10-16 = +63 и 10-02 (240) -> 10-16 = +49: ось размера экстраполяции
GRID_LATE="226 240 247 254 261 268 275 282 289"
for CK in SEQ-01-S43-V0918 SEQ-01-S44-V0918 SEQ-01-S43-V1002 SEQ-01-S44-V1002; do
  python -m src.seq crossdepth --ckpt "$CK" --val 2025-10-16 --depths $GRID_LATE
done

echo "### канал avail: что стоит уход в непрожитый режим (данные не меняются)"
for CK in SEQ-01C-S42-V0904 SEQ-01-S43-V0904 SEQ-01-S44-V0904; do
  python -m src.seq availprobe --ckpt "$CK" --val 2025-10-16
done
python -m src.seq availprobe --ckpt SEQ-01-S43-V0904        # контроль на своём фолде
python -m src.seq availprobe --ckpt SEQ-01-S44-V1016

echo "### этап 1: разложение тестового прогноза по осям"
python research/strategies/results/SEQ3/decompose_test.py

echo "### этапы 2-3: кривая глубины и усадка alpha"
python research/strategies/results/SEQ3/stress.py

echo "### этап 4: честный LOFO семейств смеси"
python research/strategies/results/SEQ3/lofo.py
