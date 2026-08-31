#!/usr/bin/env bash
# ETX1 — гейт Sparse Event Transformer (EXP-036), фолд 2025-10-16, сид 42.
# Локальная RTX 4060 Ti (8 ГБ). torch.compile недоступен: на Windows нет Triton,
# поэтому eager и batch 512 — при 1024 пик VRAM 7.3 ГБ и шаг деградирует в 25 раз
# (1946 мс против 163 мс), это вытеснение в системную память, а не счёт.
set -e
cd "$(dirname "$0")/../../../.."
export PYTHONIOENCODING=utf-8 PYTHONPATH=.

# 0. событийная таблица (один раз, ~5 с из готовой панели SEQ)
python -m src.etx build

# 1. корректность представления и анти-лукап (стоп-условие)
python -m pytest src/test_etx.py -q
python -m src.etx smoke

# 2. стоимость шага
python -m src.etx bench --n-cutoffs 3 --iters 30 --batch 512 --chunk 128

# 3. проверка скорости обучения на train-лоссе (НЕ на гейтовом фолде):
#    3 cutoff'а, 1 эпоха, val-доля 5% — выбирается lr, дальше он фиксирован
for LR in 0.00075 0.0015 0.003; do
  python -m src.etx fold --val 2025-10-16 --exp ETX-PROBE-LR$LR \
    --n-cutoffs 3 --val-frac 0.05 --epochs 1 --warmup 200 \
    --batch 512 --chunk 128 --lr $LR --no-ckpt \
    > artifacts/ETX_probe_lr$LR.log 2>&1
done

# 4. боевой гейт: полный фолд, 24 обучающих cutoff'а, 4 эпохи (как у всех SEQ)
#    batch/chunk/lr уже в DEFAULT_CFG — переопределять нечего
python -m src.etx fold --val 2025-10-16 --exp ETX-01-S42 --curve \
  > artifacts/ETX1_ETX-01-S42-V1016.log 2>&1

# 5. что модель читает: глубина истории и лимит токенов, без переобучения
python -m src.etx depth --ckpt ETX-01-S42-V1016 \
  > artifacts/ETX1_depth_ETX-01-S42-V1016.log 2>&1

# 6. сводка гейта
python research/strategies/results/ETX1/diag.py --exps ETX-01-S42 --fold 2025-10-16
