#!/usr/bin/env bash
# MIX9 — поднятие арендованного A10 под тестовые модели SEQ-01 (сиды 43/44).
# Запускать НА СЕРВЕРЕ из /root/ecup после того, как туда уехали src/ и
# data/raw/train.parquet.
#
# Три ловушки Ubuntu 24.04 + драйвер 570 (замерены 2026-08-19, memory
# `ecup-seq-gpu-budget`) — порядок команд ниже их обходит:
#   1. `pip install torch==2.11.0` тянет колесо cu130, драйвер 570 = CUDA 12.8
#      его не запускает -> `torch.cuda.is_available() == False`;
#   2. системный `typing_extensions` 4.10 из Debian роняет `torch.compile`
#      на `TypeError: Too few arguments for ... CSE`;
#   3. PEP 668: каждой команде pip нужен `--break-system-packages`.
#
# Скрипт идемпотентен: если на карте уже стоит рабочий torch с CUDA (например,
# каталог остался от прошлой аренды), установка пропускается — колесо cu128
# весит 2.4 ГБ, и качать его повторно незачем.
set -eu
cd /root/ecup
export PYTHONIOENCODING=utf-8
P="pip3 install --break-system-packages"

if python3 -c 'import torch,polars,numpy; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "== окружение уже рабочее, установка пропущена"
else
  $P polars numpy pyarrow pandas
  $P --ignore-installed typing_extensions==4.15.0
  pip3 uninstall -y --break-system-packages torch 2>/dev/null || true
  $P --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
fi

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "CUDA не поднялась — см. ловушку 1"
PY

# плотная панель 250000 x 409 x 14 fp16 (~2.9 ГБ) — из сырья, ~20 с
python3 -m src.seq build

# профиль шага: проверяем, что --compile реально вдвое дешевле, до длинных прогонов
python3 -m src.seq bench --n-cutoffs 3 --iters 15 --compile
echo "== SETUP OK $(date -u '+%F %T')"
