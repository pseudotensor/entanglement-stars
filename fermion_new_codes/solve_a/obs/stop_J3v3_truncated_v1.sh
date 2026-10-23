#!/bin/bash
# Stops the J3' launcher (launch_obs_J3v3_primary_v2.sh) and its obs_raymap_solved_v3 child, started on the v6 primary's truncated history
# (the primary stopped at T/rs 0.25 on a solver rejection). Exact PIDs, checked against their arguments first. Builder analyst.
for pid in 1328289 2214269; do
  a=$(ps -o args= -p $pid 2>/dev/null)
  case "$a" in
    *launch_obs_J3v3_primary_v2.sh*|*obs_raymap_solved_v3.py*collapse_primary_rs64_A_R2_tau8_v6*) kill $pid && echo "stopped $pid: ${a:0:80}";;
    *) echo "skip $pid: ${a:0:80}";;
  esac
done
