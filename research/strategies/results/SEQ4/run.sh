#!/usr/bin/env bash
# SEQ-03A (exp_028) — устранение OOD по каналу `avail` в sequence encoder.
#
# Проверяется РОВНО одна гипотеза: если сделать энкодер устойчивым к положению
# границы `avail`, полные 365 дней на тесте перестают быть непрожитым режимом,
# и −0.0037 от дополнительных дней (exp_027, кросс-фолдовый стресс) можно забрать
# без +0.0038 штрафа за `avail ≡ 1`.
#
# Ёмкость, таргет, головы, пулинг, эпохи, сетка cutoff'ов и смесь НЕ трогаются.
set -eu
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../../.."

# 0. панель и анти-лукап (62 проверки: 37 из exp_027 + 25 на аугментацию)
python -m src.seq build
python -m pytest src/test_seq.py -q

# 0б. дымовые прогоны каждого режима: 3 cutoff'а, 1 эпоха, 5% валидации
python -m src.seq smoke --aug avail_bnd --aug-p 0.5 --aug-full 0.5
python -m src.seq smoke --aug avail_drop --aug-p 0.5
python -m src.seq smoke --aug no_avail

# 1. диагностический фолд 2025-10-16, сид 42. Варианты — ПО ОДНОМУ, с ручным
#    гейтом между ними: фолд стоит 86 минут локальной 4060 Ti. Порядок
#    BASE -> B -> C; A запускается только если B/C не дали ясного ответа.
#    После каждого сразу availprobe (код exp_027) и availcurve.
for V in BASE B C; do
  bash research/strategies/results/SEQ4/stage1.sh "$V"
  PYTHONPATH=. python research/strategies/results/SEQ4/diag.py \
    --exps SEQ-03A-BASE-S42 SEQ-03A-B-S42 SEQ-03A-C-S42
done

# 2. кросс-фолдовый стресс глубины для победителя (+77 при тестовых +76)
bash research/strategies/results/SEQ4/stage2.sh B
PYTHONPATH=. python research/strategies/results/SEQ4/crossdepth.py --variants BASE B

# 3. полный прогон победителя на четырёх фолдах, ОДИН сид
bash research/strategies/results/SEQ4/stage3.sh B
