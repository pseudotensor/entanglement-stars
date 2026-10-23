#!/bin/bash
# launch_v36d_from0_rung2_pod3_v1.sh (the busi lane as the solve builder; spec 55b (4); inbox busi #144 item 5 and #157): the averaging ladder's
# rung sigma/a 2 on engine_collapse_v36d from the common C_0 at T = 0 to T/r_s 40 in one run, slices every 1 r_s (1 to 40), on pod 3's idle H100.
# 55b (4) placed it on pod 5 after O5; #157 (pod 3 idle after C's r_s/a 64 member stopped): "put the next gated GPU item on pod 3 at once (the
# ladder members ...)". 55.1's arguments as launch_v36d_from0_rungs_pod2_v2.sh (rungs 4 and 8) with --ell 2, --T-over-rs 40 and --slices-c128
# 1 to 40. code_v36d (code-tree md5 74e9948d2d5eb7f3463a3df8f8cb769f, equal to pod 7's) and venv_v36d (numpy 2.4.6, scipy 1.17.1, torch
# 2.4.1+cu124; python 3.11.10 as pod 7's) copied from pod 7 to ${QI2_ROOT}/rung2_env. Pod 3's /workspace (MooseFS) is dead, so the run sits on the
# container disk (${QI2_ROOT}/v36d_from0; 40 GB): the box mirror (mirror_slices_v2.sh with no dataset prefix, inbox #154) copies each slice to /data
# and deletes the pod copy after an md5 match, keeping the newest 2; this script's disk guard pauses the rung (SIGSTOP) below 8 GB free and
# resumes it (SIGCONT) at 12 GB. 55b (4): the 17.125 to 18.125 compare against the point run runs on the box without stopping it. Resumable
# from its newest slice with --resume-slice/--resume-run (55.1's restart).
# Usage (pod 3): setsid nohup bash launch_v36d_from0_rung2_pod3_v1.sh > ${QI2_ROOT}/rung2_launch.out 2>&1 < /dev/null &
W=${QI2_ROOT}/v36d_from0; mkdir -p $W/logs && cd $W || exit 1
PY=${QI2_ROOT}/python; E=${QI2_ROOT}/engine_collapse_v36d.py
L=2; TAG=c32_rs64_A_R2_tau8_ell2_prep4_avg_plain_v36d_from0
BASE="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c128 $(seq -s, 1 40) --mix plain --field-kernel-cache --v36-fixes --out ."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$* $(date -u +%FT%TZ)" >> status.txt; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
[ -e collapse_${TAG}.json ] && { st "rung ell $L: refusing, collapse_${TAG}.json exists (resume from the newest slice instead)"; exit 1; }
mkdir -p states/$TAG
st "launch rung ell $L tag $TAG gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg) free_GB $(free_gb)"
nice -n 10 $PY $E $BASE --ell $L --slice-dir states/$TAG --tag $TAG > logs/$TAG.log 2>&1 < /dev/null &
P=$!; echo $P > rung_ell$L.pid; st "rung ell $L pid $P"
paused=0
while kill -0 $P 2>/dev/null; do   # the disk guard
  f=$(free_gb)
  if [ $paused = 0 ] && [ "$f" -lt 8 ]; then kill -STOP $P && paused=1 && st "guard: paused rung pid $P at $f GB free"; fi
  if [ $paused = 1 ] && [ "$f" -ge 12 ]; then kill -CONT $P && paused=0 && st "guard: resumed rung pid $P at $f GB free"; fi
  sleep 30
done
wait $P; st "rung ell $L exit $?"
