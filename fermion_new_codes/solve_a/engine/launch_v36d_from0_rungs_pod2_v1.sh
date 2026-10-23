#!/bin/bash
# launch_v36d_from0_rungs_pod2_v1.sh (the busi lane as the solve builder; spec 55.1, item 5 of reorientation 4; inbox busi #138: pod 2 idle, "put
# another gated item on pod 2 meanwhile"): the averaging ladder's rungs sigma/a 4 and 8 on engine_collapse_v36d from the common C_0 at T = 0 to
# T/r_s 18.125, side by side on pod 2's empty H100 (55.1 placed them on pod 4 after P1_purity_c30_app8192, about 25 h away; turn 63b on this
# placement is pending and governs if it differs). 55.1's lines: the point run's command with --resume-slice/--resume-run removed (the engine's
# preparation at --prep-ell 4 is the common C_0), --T-over-rs 18.125, --slices-c128 1 to 18, no --slices-c64, --slice-dir states/<tag>,
# --out ., --gpu 0; the plain iteration (--mix plain, the engine's default) with no --nk flag; --pair-workers dropped because the engine
# refuses it without --nk-jvp exact; the rest as printed. Tags: the point run's with v36cc_v36d -> v36d_from0, ell2 -> ell4 / ell8,
# nk40r2b -> plain. code_v36d and venv_v36d copied from pod 4 (md5 of the code tree equal). Resumable: relaunch a rung from its newest slice
# with --resume-slice/--resume-run. Usage (pod 2): bash launch_v36d_from0_rungs_pod2_v1.sh [4|8|both]
W=${QI2_ROOT}/v36d_from0; mkdir -p $W/logs && cd $W || exit 1
PY=${QI2_ROOT}/python; E=${QI2_ROOT}/engine_collapse_v36d.py
BASE="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 18.125 --prep-ell 4 --tau 8 --gpu 0 --slices-c128 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 --mix plain --source-continuation two-stage --end-above 1e-3 --field-kernel-cache --v36-fixes --out ."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { local L=$1 TAG=c32_rs64_A_R2_tau8_ell$1_prep4_avg_plain_v36d_from0
  mkdir -p states/$TAG
  echo "launch rung ell $L tag $TAG $(date -u +%FT%TZ) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg)" >> status.txt
  setsid nohup nice -n 10 $PY $E $BASE --ell $L --slice-dir states/$TAG --tag $TAG > logs/$TAG.log 2>&1 < /dev/null &
  echo "rung ell $L pid $! $(date -u +%FT%TZ)" >> status.txt; echo $! > rung_ell$L.pid; }
case ${1:-both} in 4) run 4;; 8) run 8;; both) run 4; sleep 120; run 8;; esac
