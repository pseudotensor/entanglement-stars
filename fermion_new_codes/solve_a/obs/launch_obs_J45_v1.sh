#!/bin/bash
# J4 smoke and J5 (SPEC_final4b_observables.md secs. 1.2, 4.2, 4.4) on GPU 2 only: runs the lines of a list file sequentially, one process
# at a time; before each line it checks that GPU 2 holds less than 1 GB and no process (else the line is skipped and recorded); nice 10,
# 2 host threads, CUDA_VISIBLE_DEVICES=2.  List lines: "<tag> <script> <args...>" (the script path relative to J45/code).
# Usage (on the box, detached): nohup bash launch_obs_J45_v1.sh <list file> > launcher_<name>.log 2>&1 < /dev/null &
# Builder analyst.
LIST=$1
cd ${QI2_ROOT}/J45
PY=${QI2_ROOT}/python
STATUS=status.txt
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==2{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
  local nproc=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u' | wc -l)
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
while read -r tag script args; do
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac
  if ! gpu_free; then echo "J45 $tag skipped: GPU 2 not free at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS; continue; fi
  echo "J45 $tag start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/$script $args > out/$tag.log 2>&1 < /dev/null
  echo "J45 $tag exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS
done < "$LIST"
echo "J45 DONE $(basename $LIST) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS
