#!/bin/bash
# stop_ladder_noslices_v1.sh (builder analyst): the ladder's sigma/a 8 and 2 members were launched without orbital slices, which task 6's
# interior slices on the ladder's records (the order of work's item 9) need; both are about 25 minutes in and are relaunched with slices.
# Kills each PID only when its command line still matches. Run on pod 1 from ${QI2_ROOT}/engine.
cd ${QI2_ROOT}/engine || exit 1
stop_one() {
  local pid=$1 want=$2 args
  args=$(ps -o args= -p "$pid" 2>/dev/null)
  if [ -z "$args" ]; then echo "pid $pid gone"; return; fi
  case "$args" in *"$want"*) kill "$pid" && echo "killed $pid ($want)";; *) echo "pid $pid kept: $args";; esac
}
stop_one 20327 "engine_collapse_v27.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --out C3lad27 --gpu 0 --ell 8"
stop_one 20613 "engine_collapse_v27.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --out C3lad27 --gpu 0 --ell 2"
grep -qxF lad27_sigma4 queue_lane4.txt.started || echo lad27_sigma4 >> queue_lane4.txt.started
echo "ladder members sigma/a 8 and 2 stopped for a relaunch with orbital slices $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
