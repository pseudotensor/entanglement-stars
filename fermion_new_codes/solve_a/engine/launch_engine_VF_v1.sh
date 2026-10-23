#!/bin/bash
# J-VF, the vacuum-flow control (SPEC_final4b_engine.md sec. 13.2) on GPU 2 only: first waits until the box3d subagent's list runner
# (launch_obs_J45_v1.sh) has exited and GPU 2 holds under 1 GB and no process; then runs the lines of a list file sequentially, one process
# at a time, re-checking GPU 2 before each line (else the line is skipped and recorded); nice 10, 2 host threads, CUDA_VISIBLE_DEVICES=2.
# List lines: "<tag> <engine_vacflow_v1.py args...>" (without --gpu, --out and --tag, which the runner adds: --gpu 0 --out VF --tag <tag>).
# Usage (on the box, detached): nohup bash code/launch_engine_VF_v1.sh <list file> > VF/launcher_<name>.log 2>&1 < /dev/null &
# Builder analyst.
LIST=$1
cd ${QI2_ROOT}/engine
mkdir -p VF
PY=${QI2_ROOT}/python
STATUS=status.txt
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==2{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
  local nproc=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u' | wc -l)
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
j45_alive() {
  ps -eo args | awk '$1=="bash" && $2=="code/launch_obs_J45_v1.sh"' | grep -q .
}
echo "J-VF waiter start $(date -u +%H:%M:%S) UTC (GPU 2 after the J45 list runner)" >> $STATUS
while j45_alive || ! gpu_free; do sleep 60; done
sleep 30
while read -r tag args; do
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac
  if ! gpu_free; then echo "J-VF $tag skipped: GPU 2 not free at $(date -u +%H:%M:%S) UTC" >> $STATUS; continue; fi
  echo "J-VF $tag start $(date -u +%H:%M:%S) UTC (gpu 2)" >> $STATUS
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/engine_vacflow_v1.py $args --gpu 0 --out VF --tag $tag > VF/$tag.log 2>&1 < /dev/null
  echo "J-VF $tag exit $? $(date -u +%H:%M:%S) UTC" >> $STATUS
done < "$LIST"
echo "J-VF DONE $(basename $LIST) $(date -u +%H:%M:%S) UTC" >> $STATUS
