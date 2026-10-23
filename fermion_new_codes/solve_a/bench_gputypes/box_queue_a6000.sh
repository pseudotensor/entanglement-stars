#!/bin/bash
# box_queue_a6000.sh (solve bench_gputypes v1): runs the kits on the box's A6000s, one at a time, each only on a GPU that holds no compute process
# at its start (bench_exec refuses otherwise). Waits (polling every 60 s, up to WAIT_H hours) for such a GPU before each kit. NT 20 (= the PERF H1
# on-arm's count on R2; the box has 72 cores, no quota). Usage: nohup bash box_queue_a6000.sh [kits...] > queue.log 2>&1 &
B=${QI2_ROOT}/bench_gputypes; NT=${NT:-20}; WAIT_H=${WAIT_H:-24}
KITS=${*:-b_page_v6q_k19 a_rung2_v36q c_twofield_v9c2}
free_gpu() {
  local busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read i u m; do
    echo "$busy" | grep -q "$u" || { [ "$m" -lt 1024 ] && { echo $i; break; }; }
  done
}
for k in $KITS; do
  t0=$(date +%s)
  while :; do
    G=$(free_gpu)
    if [ -n "$G" ]; then sleep 20; [ "$(free_gpu)" = "$G" ] && break; echo "$(date -u +%FT%TZ) GPU $G taken during the settle; waiting"; fi
    [ $(( $(date +%s) - t0 )) -gt $((WAIT_H * 3600)) ] && { echo "$(date -u +%FT%TZ) $k: no free GPU in $WAIT_H h; stop"; exit 1; }
    sleep 60
  done
  echo "$(date -u +%FT%TZ) $k start on GPU $G NT $NT"
  BENCH_TAG=box bash $B/kits/$k/run.sh $G $NT; echo "$(date -u +%FT%TZ) $k end rc $?"
done
echo "$(date -u +%FT%TZ) queue done"
