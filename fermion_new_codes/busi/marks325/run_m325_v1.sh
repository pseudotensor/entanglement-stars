#!/bin/bash
# run_m325_v1.sh (busi f52b1152, 2026-10-20; inbox #325): card-queue entry for one m325 continuation (own56, pa56 or fs56). Model:
# ${QI2_ROOT}/run_J1_W2_anc_v1.sh. Verifies the kit md5s, launches through m325_launch_v1.sh (which refuses if any output exists,
# and refuses fs56 until own56 ended rc 0 at 56), checks the pid is the kit's engine, blocks until the engine exits. Never kills.
# Usage: run_m325_v1.sh own56|pa56|fs56      (env as m325_launch_v1.sh: M325_KIT, M325_OUT, M325_PY, M325_GPU)
export NUMPY_MADVISE_HUGEPAGE=0
MODE=$1; K=${M325_KIT:-${QI2_ROOT}/kit_v1}; OUT=${M325_OUT:-${QI2_ROOT}/marks325}; now() { date -u +%FT%TZ; }
case "$MODE" in own56) EN=engine_collapse_v36d.py;; pa56|fs56) EN=engine_collapse_v36g.py;; *) echo "usage: $0 own56|pa56|fs56"; exit 1;; esac
echo "$(now) $(hostname) m325 $MODE start; checking kit md5s"
( cd $K && md5sum -c --quiet MD5SUMS ) || { echo "$(now) kit md5 check FAILED; refusing"; exit 1; }
echo "$(now) kit md5s OK"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
df -BG --output=avail $(dirname $OUT) | tail -1 | sed "s/^/$(now) free on the output disk: /"
bash $K/m325_launch_v1.sh $MODE || { echo "$(now) launcher refused"; exit 1; }
PF=$OUT/pids/$MODE.pid; P=$(cat $PF); tr "\0" " " < /proc/$P/cmdline | grep -q "$K/code/$EN" || { echo "$(now) pid $P is not the kit $EN engine"; exit 1; }
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "$(now) engine pid $P ended rc $(cat $PF.rc 2>/dev/null)"; exit $(cat $PF.rc 2>/dev/null || echo 1)
