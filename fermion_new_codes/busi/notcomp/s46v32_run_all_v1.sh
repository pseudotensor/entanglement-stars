#!/bin/bash
# busi #472: the state-leakage bound on all 18 Hartree slices, box GPU 1 (cuda:1) only, nice 10, 8 threads; each job waits until
# GPU 1 carries no compute process other than the litellm proxies (lower priority than the running B' and foundations work).
# Measured: s46v32_stateleak_v1 335 s on prod 15; s46v31_mass2save_v1 1296 to 1733 s per slice. 15 slices need their operators first:
# about 15 x (1500 + 340) s = 7.7 h, plus 2 x 340 s.
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
waitgpu() { until gpu1_free; do echo "$(date -u +%FT%TZ) GPU 1 busy; waiting" >> $B/logs/run_all.status; sleep 300; done; }
for s in "prod 35" "s8 14" "prod 12" "prod 16" "prod 20" "prod 24" "prod 25" "prod 28" "prod 30" "prod 40" "s8 12" "s8 13" "s8 16" "s8 20" "s8 25" "s8 30" "s8 40"; do
  set -- $s
  tag=$(printf "%s_T%07.2f" $1 $2)
  if [ ! -f $OPS/${tag}_ops.npz ]; then
    waitgpu
    echo "$(date -u +%FT%TZ) start ops $1 $2" >> $B/logs/run_all.status
    nice -n 10 $PY -B s46v31_mass2save_v1.py --slice $1 $2 --device cuda:1 --threads 8 --out $OPS > $B/logs/mass2save_${1}_T$2.log 2>&1
    echo "$(date -u +%FT%TZ) end ops rc=$? $1 $2" >> $B/logs/run_all.status
  fi
  waitgpu
  echo "$(date -u +%FT%TZ) start stateleak $1 $2" >> $B/logs/run_all.status
  nice -n 10 $PY -B s46v32_stateleak_v1.py --slice $1 $2 --ops $OPS/${tag}_ops.npz --device cuda:1 --threads 8 > $B/logs/stateleak_${1}_T$2.log 2>&1
  echo "$(date -u +%FT%TZ) end stateleak rc=$? $1 $2" >> $B/logs/run_all.status
done
echo "$(date -u +%FT%TZ) queue done" >> $B/logs/run_all.status
