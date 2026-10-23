#!/bin/bash
# GPU 2's queue after the J-VF re-plan (SPEC_final4b_engine.md secs. 15.2 and 15.4): waits until the J-VF runner v2 has gone and GPU 2 holds
# under 1 GB and no job process (the resident litellm services are not jobs), then runs the list's lines in order, one process at a time,
# re-checking the GPU before each line. List lines: "<tag> <script relative to engine/code> <args...>" (the args include --out; the runner adds
# --gpu 0 and --tag <tag>). nice 10, 2 host threads (4 for the collapse), CUDA_VISIBLE_DEVICES=2.
# Usage (on the box, detached, from engine): nohup bash code/launch_engine_gpu2_v1.sh code/list_gpu2_v1.txt > gpu2_launcher_v1.log 2>&1 < /dev/null &
# Builder analyst.
LIST=$1
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
STATUS=status.txt
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==2{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
  local nproc=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u {print $2}'); do
    ps -o args= -p $p | grep -q litellm || nproc=$((nproc + 1))
  done
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
vf2_alive() { ps -eo args | awk '$1=="bash" && $2=="code/launch_engine_VF_v2.sh"' | grep -q .; }
echo "GPU2 queue waiter start $(date -u +%H:%M:%S) UTC (after the J-VF cap at kappa 192)" >> $STATUS
while vf2_alive || ! gpu_free; do sleep 30; done
sleep 20
while read -r tag script args; do
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac
  if ! gpu_free; then echo "GPU2 $tag skipped: GPU 2 not free at $(date -u +%H:%M:%S) UTC" >> $STATUS; continue; fi
  th=2; case "$script" in *collapse*) th=4;; esac
  echo "GPU2 $tag start $(date -u +%H:%M:%S) UTC ($script, gpu 2)" >> $STATUS
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th nice -n 10 $PY code/$script $args --gpu 0 --tag $tag > logs_gpu2_$tag.log 2>&1 < /dev/null
  echo "GPU2 $tag exit $? $(date -u +%H:%M:%S) UTC" >> $STATUS
done < "$LIST"
echo "GPU2 DONE $(basename $LIST) $(date -u +%H:%M:%S) UTC" >> $STATUS
