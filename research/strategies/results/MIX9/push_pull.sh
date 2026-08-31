#!/usr/bin/env bash
# MIX9 — оркестровка арендованного A10 с локальной машины: залить, запустить, забрать.
#
#   HOST=root@185.182.108.52 bash research/strategies/results/MIX9/push_pull.sh push
#   HOST=... bash .../push_pull.sh run          # setup + обе тестовые модели, фоном
#   HOST=... bash .../push_pull.sh watch        # хвост лога очереди
#   HOST=... bash .../push_pull.sh pull         # ztest_/uid_/model_ обратно
#
# Везём только `src/` и `data/raw/train.parquet` (180 МБ): плотная панель
# собирается на месте за ~20 с, тащить `data/processed` (3.7 ГБ) не нужно
# (`research/compute_profile.md`).
set -eu
HOST=${HOST:?укажите HOST=root@<ip>}
R=/root/ecup
CMD=${1:?push | run | watch | pull}

case "$CMD" in
push)
  ssh "$HOST" "mkdir -p $R/src $R/data/raw $R/artifacts $R/data/processed"
  scp -q src/*.py "$HOST:$R/src/"
  scp -q research/strategies/results/MIX9/setup_server.sh \
         research/strategies/results/MIX9/run_test_models.sh "$HOST:$R/"
  ssh "$HOST" "test -s $R/data/raw/train.parquet" \
    && echo "train.parquet уже на месте" \
    || scp data/raw/train.parquet "$HOST:$R/data/raw/"
  ssh "$HOST" "ls -la $R; du -sh $R/data/raw"
  ;;
run)
  ssh "$HOST" "cd $R && setsid nohup bash -c 'bash setup_server.sh && bash run_test_models.sh' \
      > artifacts/MIX9_queue.log 2>&1 < /dev/null & echo запущено PID \$!"
  ;;
watch)
  ssh "$HOST" "tail -n 40 $R/artifacts/MIX9_queue.log"
  ;;
pull)
  for S in 43 44; do
    scp -q "$HOST:$R/artifacts/ztest_SEQ-C289-S$S.npy" artifacts/ || true
    # полная глубина нужна не сабмиту, а диагностике `check_test_seeds.py`:
    # ею проверяется, что `--depth-clip 289` действительно сработал
    scp -q "$HOST:$R/artifacts/ztest_SEQ-C289-S$S-FULL.npy" artifacts/ || true
    scp -q "$HOST:$R/artifacts/uid_SEQ-C289-S$S.npy" artifacts/ || true
    scp -q "$HOST:$R/artifacts/model_SEQ-C289-S$S-TEST.pt" artifacts/ || true
    scp -q "$HOST:$R/artifacts/MIX9_SEQ-C289-S$S.log" artifacts/ || true
  done
  scp -q "$HOST:$R/artifacts/MIX9_queue.log" artifacts/ || true
  ls -la artifacts/ztest_SEQ-C289-* artifacts/uid_SEQ-C289-*
  ;;
*) echo "неизвестная команда $CMD"; exit 2;;
esac
