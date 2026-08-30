#!/usr/bin/env bash
# SEQ-03A этап 2 — кросс-фолдовый стресс по глубине для победителя этапа 1.
#
# Вопрос этапа: аугментация сняла зависимость от `avail` — но не ценой ли
# способности пользоваться реальными дополнительными днями? Это две разные вещи,
# и разделяет их та же конструкция, что в exp_027: модель РАННЕГО фолда (09-04,
# обучена до глубины 212) применяется к ПОЗДНЕЙ панели (10-16, доступно 289),
# то есть экстраполяция +77 при тестовых +76. Лукапа нет: таргеты обучения
# кончаются 2025-09-04, таргет панели живёт в (2025-10-16, 2025-11-15].
#
# BASE считается заново в том же окружении (eager, локальная 4060 Ti): числа
# exp_027 сняты на A10 с --compile, и сравнивать пару нужно внутри одного прогона.
#
# Использование: bash stage2.sh <ТЕГ-ПОБЕДИТЕЛЯ>   (например: B)
set -eu
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../../.."
WIN=${1:?нужен тег победителя этапа 1, например B}

fold0904 () {                  # fold0904 <тег> <аргументы аугментации...>
  tag="SEQ-03A-$1-S42"; shift
  if [ -f "artifacts/oof_${tag}-V0904.npz" ]; then
    echo "== $tag V0904 уже посчитан, пропуск"; return
  fi
  echo "== $tag V0904 $(date +%H:%M:%S)"
  python -m src.seq fold --val 2025-09-04 --epochs 4 --seed 42 --exp "$tag" "$@" \
    > "artifacts/SEQ4_${tag}_V0904.log" 2>&1
  tail -3 "artifacts/SEQ4_${tag}_V0904.log"
}

aug_args () {                  # те же аргументы, что на этапе 1
  case "$1" in
    BASE) echo "--aug none" ;;
    B)    echo "--aug avail_bnd --aug-p 0.5 --aug-full 0.5" ;;
    C)    echo "--aug no_avail" ;;
    A50)  echo "--aug avail_drop --aug-p 0.5" ;;
    A25)  echo "--aug avail_drop --aug-p 0.25" ;;
    *) echo "неизвестный тег $1" >&2; exit 1 ;;
  esac
}

fold0904 BASE $(aug_args BASE)
fold0904 "$WIN" $(aug_args "$WIN")

# кривая глубины на поздней панели: сетка та же, что в exp_027
for T in BASE "$WIN"; do
  P="SEQ-03A-$T-S42-V0904"
  python -m src.seq crossdepth --ckpt "$P" --val 2025-10-16 \
    --depths 212 230 247 261 275 289 > "artifacts/SEQ4_xdepth_$T.log" 2>&1
  tail -4 "artifacts/SEQ4_xdepth_$T.log"
  # и availprobe раннего чекпойнта на поздней панели — конфигурация exp_027
  python -m src.seq availprobe --ckpt "$P" --val 2025-10-16 \
    > "artifacts/SEQ4_probe_${T}_V0904on1016.log" 2>&1
  tail -2 "artifacts/SEQ4_probe_${T}_V0904on1016.log"
done
echo "== этап 2 закончен $(date +%H:%M:%S)"
