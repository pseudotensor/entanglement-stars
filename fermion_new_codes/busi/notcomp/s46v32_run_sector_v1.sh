#!/bin/bash
# busi #472: the sector self-test of the state-leakage bound on the leak pilot's three slices; waits for the s46v32 queue to end
# ("queue done") and for GPU 1 to carry no compute process other than the litellm proxies; cuda:1 only, nice 10, 8 threads.
PY=${QI2_ROOT}/python
B=${QI2_ROOT}/s46v32
OPS=${QI2_ROOT}/ops
UUID1=GPU-d53d5251-00ff-d3f0-1183-3dbec64d0ec3
cd $B/code
gpu1_free() {
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u=$UUID1 '$2==u{print $1}'); do
    ps -p $p -o args= | grep -q litellm || return 1
  done
  return 0
}
until grep -q "queue done" $B/logs/run_all.status; do sleep 120; done
for s in "prod 15" "prod 35" "s8 14"; do
  set -- $s
  tag=$(printf "%s_T%07.2f" $1 $2)
  until gpu1_free; do sleep 300; done
  echo "$(date -u +%FT%TZ) start sector $1 $2 load $(cut -d' ' -f1 /proc/loadavg)" >> $B/logs/run_sector.status
  nice -n 10 $PY -B s46v32_sector_selftest_v1.py --slice $1 $2 --ops $OPS/${tag}_ops.npz --n 12,14,16 --device cuda:1 --threads 8 > $B/logs/sector_${1}_T$2.log 2>&1
  echo "$(date -u +%FT%TZ) end sector rc=$? $1 $2" >> $B/logs/run_sector.status
done
echo "$(date -u +%FT%TZ) sector done" >> $B/logs/run_sector.status
