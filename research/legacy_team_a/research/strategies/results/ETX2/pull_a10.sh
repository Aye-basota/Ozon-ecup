#!/usr/bin/env bash
# ETX2 — забрать с A10 всё, что уже посчитано (идемпотентно, отсутствующее молча пропускается).
#
#   HOST=root@185.182.108.52 bash research/strategies/results/ETX2/pull_a10.sh
#
# Локальная машина считает ТОЛЬКО `ETX-01-S44-V0904/0918/1002`, поэтому здесь нет
# ни одного файла, который пишется локально: пересечения записи между машинами нет
# по построению.
set -u
HOST=${HOST:-root@185.182.108.52}
R=/root/ecup/artifacts
cd "$(dirname "$0")/../../../.."

FILES=""
for S in 43 44; do
  FILES="$FILES ztest_ETX-01-S$S.npy ztest_ETX-01-S$S-FULL.npy uid_ETX-01-S$S.npy"
  FILES="$FILES uid_ETX-01-S$S-FULL.npy model_ETX-01-S$S-TEST.pt ETX2_test_ETX-01-S$S.log"
done
for T in V0904 V0918 V1002 V1016; do
  FILES="$FILES oof_ETX-01-S43-$T.npz curve_ETX-01-S43-$T.json ETX2_ETX-01-S43-$T.log"
done
FILES="$FILES oof_ETX-01-S44-V1016.npz curve_ETX-01-S44-V1016.json ETX2_ETX-01-S44-V1016.log"
FILES="$FILES ETX2_queue.log"

# один вызов ssh на список: экономит рукопожатия, их тут два десятка
HAVE=$(ssh "$HOST" "cd $R && ls $FILES 2>/dev/null")
[ -z "$HAVE" ] && { echo "на сервере пока ничего из списка"; exit 0; }
# shellcheck disable=SC2086
scp -q $(for f in $HAVE; do echo "$HOST:$R/$f"; done | tr '\n' ' ') artifacts/ 2>/dev/null
echo "забрано $(echo "$HAVE" | wc -w) файлов:"
echo "$HAVE" | tr ' ' '\n' | sed 's/^/  /'
