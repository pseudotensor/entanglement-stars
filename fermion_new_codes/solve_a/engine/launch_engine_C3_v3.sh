#!/bin/bash
# The collapse on GPU 1 (version 3 of the runner: engine_collapse_v4.py; the GPU-free test ignores the resident litellm services) (SPEC_final4b_engine.md secs. 12 and 13, job order of 13.6): first waits until J3's channel runner
# (launch_obs_J3_v1.sh) has exited and GPU 1 holds under 1 GB and no process; then runs the lines of a list file sequentially, one process
# at a time, re-checking GPU 1 before each line (else the line is skipped and recorded); a line runs only if the previous line exited 0
# (a failed benchmark stops the list); nice 10, 4 host threads, CUDA_VISIBLE_DEVICES=1.
# List lines: "<tag> <engine_collapse_v4.py args...>" (without --gpu, --out and --tag, which the runner adds: --gpu 0 --out C3 --tag <tag>).
# Usage (on the box, detached): nohup bash code/launch_engine_C3_v1.sh <list file> > C3/launcher_<name>.log 2>&1 < /dev/null &
# Builder analyst.
LIST=$1
cd ${QI2_ROOT}/engine
mkdir -p C3
PY=${QI2_ROOT}/python
STATUS=status.txt
gpu_free() {
  # version 3 of the runner: the resident litellm service processes (a container's proxy, 260 MiB each, there for days and beside J3) are not
  # jobs; every other compute process on GPU 1 is.
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==1{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')
  local nproc=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u {print $2}'); do
    ps -o args= -p $p | grep -q litellm || nproc=$((nproc + 1))
  done
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
j3_alive() {
  ps -eo args | awk '$1=="bash" && $2=="launch_obs_J3_v1.sh"' | grep -q .
}
echo "C3 waiter start $(date -u +%H:%M:%S) UTC (GPU 1 after J3's channel runner)" >> $STATUS
while j3_alive || ! gpu_free; do sleep 60; done
sleep 30
rc=0
while read -r tag args; do
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac
  if [ $rc -ne 0 ]; then echo "C3 $tag not started: the previous line exited $rc" >> $STATUS; continue; fi
  if ! gpu_free; then echo "C3 $tag skipped: GPU 1 not free at $(date -u +%H:%M:%S) UTC" >> $STATUS; rc=1; continue; fi
  echo "C3 $tag start $(date -u +%H:%M:%S) UTC (gpu 1)" >> $STATUS
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY code/engine_collapse_v4.py $args --gpu 0 --out C3 --tag $tag > C3/$tag.log 2>&1 < /dev/null
  rc=$?
  echo "C3 $tag exit $rc $(date -u +%H:%M:%S) UTC" >> $STATUS
done < "$LIST"
echo "C3 DONE $(basename $LIST) $(date -u +%H:%M:%S) UTC" >> $STATUS
