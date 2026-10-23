#!/bin/bash
# run_J1_W2_v36v_v1.sh (busi 9ae0d5c4, 2026-10-19): card-queue entry for W2 under engine v36v (busi/notes/W2_pass12625_ruling_reviewa_v1.md sec. 3 "v36v launch":
# R2prime's arguments + dg_stall=8 dg_sub_counts=1,2,3,4,5,6,8 dg_sub_max=6 dg_sub_start=1 dg_prim_halve=on). Model: run_J1_W2_v36p_r2b_v1.sh.
# Usage: run_J1_W2_v36v_v1.sh [SLICE_T] [TAGX]   SLICE_T default 12.50 (the kit slice 4aa640af); else R2prime's latest slice clock (12.75, 12.875, 13.00 ...),
#   read from $ROOT/states/<TAGP>_r2b/ (read only). TAGX: an extra tag suffix for a second v36v launch.
# First verifies every kit file against ${QI2_ROOT}/MD5SUMS and MD5SUMS_v36v; refuses if the out dir already exists (launcher). Blocks until the engine
# exits. Never kills anything. Only ONE card may run this.
export NUMPY_MADVISE_HUGEPAGE=0
K=${QI2_ROOT}/w2r2; R=${QI2_ROOT}/ladders; now() { date -u +%FT%TZ; }
export SLICE_T=${1:-12.50} TAGX=${2:-}
echo "$(now) $(hostname) start (SLICE_T $SLICE_T, TAGX '$TAGX'); checking kit md5s"
( cd $K && md5sum -c --quiet MD5SUMS && md5sum -c --quiet MD5SUMS_v36v ) || { echo "$(now) kit md5 check FAILED; refusing"; exit 1; }
echo "$(now) kit md5s OK"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
bash $K/w2p_launch_busi.v36v.sh main || { echo "$(now) launcher refused"; exit 1; }
PF=$R/pids/W2_dgp_v36v$TAGX.pid; P=$(cat $PF); tr "\0" " " < /proc/$P/cmdline | grep -q ${QI2_ROOT}/engine_collapse_v36v.py || { echo "$(now) pid $P is not the kit v36v engine"; exit 1; }
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "$(now) engine pid $P ended rc $(cat $PF.rc 2>/dev/null)"; exit $(cat $PF.rc 2>/dev/null || echo 1)
