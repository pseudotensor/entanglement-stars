#!/bin/bash
# run_s8half_h100f_v1.sh (busi, inbox #461: h100-f preferred for C_half450). ON h100-f, as the user of the F256_s8 runs. Wraps the kit's
# run_s8half_card_v1.sh for h100-f: the T0004.50 slice already there (${QI2_ROOT}/states, md5-checked here, fa4e0ec3)
# is linked into ROOT/parent; the venv is h100-f's ${QI2_ROOT}/venv (the ended run's own); GPU 0; the shared lock
# ${QI2_ROOT}/busi_gpu.lock is held and the GPU must be empty (the B' benchmark/arms go first), as the F256_s8 restarts did; self-stop STOP_AT
# (default 2026-10-23T04:00:00Z: no reclaim on h100-f, so this is a wall cap only).
#   bash run_s8half_h100f_v1.sh check|go     (kit unpacked at ${QI2_ROOT}/s8half_kit_v1 by push_kit_s8half_v1.sh)
set -u
MODE=${1:?check|go}; ROOT=${QI2_ROOT}/busi_s8half; KIT=$ROOT/s8half_kit_v1
T8=lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb
SRC=${QI2_ROOT}/orbitals_${T8}_T0004.50.npz
mkdir -p $ROOT/parent; [ -e $ROOT/parent/orbitals_${T8}_T0004.50.npz ] || ln -s $SRC $ROOT/parent/
if [ ! -e $ROOT/parent/SLICE_OK ]; then m=$(md5sum < $SRC | cut -c1-32); [ $m = fa4e0ec3019e8400fd565672a1067205 ] || { echo "slice md5 $m FAIL"; exit 6; }; touch $ROOT/parent/SLICE_OK; fi
export GPU=0 ROOT V=${QI2_ROOT}/venv STOP_AT=${STOP_AT:-2026-10-23T04:00:00Z} THR=${THR:-14}
[ $MODE = go ] || exec bash $KIT/run_s8half_card_v1.sh check
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; echo "$(date -u +%FT%TZ) lock held" >> $ROOT/RUN_LOG.txt
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 15; done
bash $KIT/run_s8half_card_v1.sh go
while [ ! -e $ROOT/RC ]; do sleep 30; done; echo "$(date -u +%FT%TZ) lock released ($(cat $ROOT/RC))" >> $ROOT/RUN_LOG.txt
