#!/bin/bash
# A GPU queue runner, version 4 of launch_engine_gpu_v3.sh: a line runs only if the previous line exited 0 (a failed benchmark stops the list). (lead turn 24, engine 16.5): waits until each PID in AFTER (comma list, or "none") has exited, the MARKER file
# exists (or "none") and GPU <G> holds under 1 GB and no job process (the resident litellm services are not jobs); then runs the list's lines
# in order, one process at a time, re-checking the GPU before each line; a line whose script is missing is skipped and recorded.
# List lines: "<tag> <script relative to engine/code> <args...>" (the args include --out; the runner adds --gpu 0 and --tag <tag>; for
# obs scripts give the path relative to engine/code, e.g. ../../obs/code/x.py, with their own output args). nice 10, 2 host threads (4 for the
# collapse and star engines), CUDA_VISIBLE_DEVICES=<G>.
# Usage (on the box, detached, from engine): nohup bash code/launch_engine_gpu_v4.sh <G> <list> <AFTER> <MARKER> > gpu<G>_launcher_<name>.log 2>&1 < /dev/null &
# Builder analyst.
G=$1; LIST=$2; AFTER=$3; MARKER=$4
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
STATUS=status.txt
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$G '$1==g{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$G '$1==g{print $2}')
  local nproc=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u {print $2}'); do
    ps -o args= -p $p | grep -q litellm || nproc=$((nproc + 1))
  done
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
pids_alive() {
  [ "$AFTER" = "none" ] && return 1
  for p in ${AFTER//,/ }; do kill -0 $p 2>/dev/null && return 0; done
  return 1
}
marker_missing() { [ "$MARKER" != "none" ] && [ ! -f "$MARKER" ]; }
echo "GPU$G queue $(basename $LIST) waiter start $(date -u +%H:%M:%S) UTC (after PIDs $AFTER, marker $MARKER)" >> $STATUS
while pids_alive || marker_missing || ! gpu_free; do sleep 30; done
sleep 20
rc=0
while read -r tag script args; do
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac
  if [ $rc -ne 0 ]; then echo "GPU$G $tag not started: the previous line exited $rc" >> $STATUS; continue; fi
  if [ ! -f "code/$script" ]; then echo "GPU$G $tag skipped: code/$script missing $(date -u +%H:%M:%S) UTC" >> $STATUS; continue; fi
  if ! gpu_free; then echo "GPU$G $tag skipped: GPU $G not free at $(date -u +%H:%M:%S) UTC" >> $STATUS; continue; fi
  th=2; case "$script" in *collapse*|*star*) th=4;; esac
  echo "GPU$G $tag start $(date -u +%H:%M:%S) UTC ($script, gpu $G)" >> $STATUS
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th nice -n 10 $PY code/$script $args --gpu 0 --tag $tag > logs_gpu${G}_$tag.log 2>&1 < /dev/null
  rc=$?
  echo "GPU$G $tag exit $rc $(date -u +%H:%M:%S) UTC" >> $STATUS
done < "$LIST"
echo "GPU$G DONE $(basename $LIST) $(date -u +%H:%M:%S) UTC" >> $STATUS
