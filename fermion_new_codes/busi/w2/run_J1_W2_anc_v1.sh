#!/bin/bash
# run_J1_W2_anc_v1.sh (busi f52b1152, 2026-10-20): card-queue entry for the W2 no-code fallback (ruling sec. 4): the epoch 12.5 -> 14.5 under
# dg_assign=anchor (v36l's solver) on fixv4c. Model: run_J1_W2_v36v_v1.sh. Verifies the kit md5s, launches, blocks until the engine exits. Never kills.
export NUMPY_MADVISE_HUGEPAGE=0
K=${QI2_ROOT}/w2r2; R=${QI2_ROOT}/ladders; now() { date -u +%FT%TZ; }
echo "$(now) $(hostname) start; checking kit md5s"
( cd $K && md5sum -c --quiet MD5SUMS && md5sum -c --quiet MD5SUMS_anc ) || { echo "$(now) kit md5 check FAILED; refusing"; exit 1; }
echo "$(now) kit md5s OK"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
bash $K/w2p_launch_busi.f4c_anc.sh main || { echo "$(now) launcher refused"; exit 1; }
PF=$R/pids/W2_dgp_anc.pid; P=$(cat $PF); tr "\0" " " < /proc/$P/cmdline | grep -q ${QI2_ROOT}/engine_collapse_v36p_fixv4c.py || { echo "$(now) pid $P is not the kit fixv4c engine"; exit 1; }
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "$(now) engine pid $P ended rc $(cat $PF.rc 2>/dev/null)"; exit $(cat $PF.rc 2>/dev/null || echo 1)
