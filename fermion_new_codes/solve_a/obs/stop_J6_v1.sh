#!/bin/bash
# Stops J6 (flow_chart_v4.py cut, nk 24, 1024 sites, ramp 2560 at dt 0.5; launched by launch_obs_J6_v1.sh at 08:35 UTC): measured cost 15.3 hours
# per channel on one thread (eigh 8.7 s + 2.1 s per step, 5120 steps) and 1446 channels on 6 workers, i.e. about 150 days; nothing written after
# 7.8 hours. Exact PIDs checked against their arguments: the launcher, the pool's parent, its six workers. Builder analyst.
chk() { local pid=$1 pat=$2; local a; a=$(ps -o args= -p $pid 2>/dev/null); if [[ "$a" == *"$pat"* ]]; then kill $pid && echo "stopped $pid: ${a:0:90}"; else echo "skip $pid: '${a:0:60}'"; fi; }
chk 1005147 "launch_obs_J6_v1.sh"
for pid in 1005161 1005219 1005220 1005221 1005222 1005223 1005224; do chk $pid "flow_chart_v4.py cut --inv-kappa 64 --delta 0.4 --nsites 1024"; done
