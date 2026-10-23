#!/bin/bash
# Version 3 of wait_then_J45final_ell16_v1.sh (engine 17.6's GPU 2 order after the r_s = 32 set on engine_collapse_v10): waits until the engine's
# GPU 2 runner reports "GPU2 DONE list_gpu2_v7.txt", then (1) the post list engine/code/list_gpu2_post_v1.txt if it exists by then (C5.3' and
# J6's GPU route, observables 13.4), (2) the subagent's list_final_v1.txt (obs/J45, launch_obs_J45_v1.sh), (3) the ell = 16 control on
# engine_collapse_v10 (list_gpu2_ell16_v2.txt). Usage (on the box, detached, from obs/J45): nohup bash wait_then_J45final_ell16_v3.sh > launcher_final_v3.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/J45 || exit 1
echo "J45 final waiter (v3) start $(date -u +%Y-%m-%dT%H:%M:%SZ) (after GPU2 DONE list_gpu2_v7.txt)" >> status.txt
until grep -q "^GPU2 DONE list_gpu2_v7.txt" ${QI2_ROOT}/status.txt; do sleep 60; done
sleep 30
if [ -f ${QI2_ROOT}/list_gpu2_post_v1.txt ]; then
  (cd ${QI2_ROOT}/engine && bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_post_v1.txt none none)
else
  echo "J45 final waiter: no list_gpu2_post_v1.txt at $(date -u +%Y-%m-%dT%H:%M:%SZ); C5.3' and J6 not run here" >> status.txt
fi
bash code/launch_obs_J45_v1.sh list_final_v1.txt
cd ${QI2_ROOT}/engine || exit 1
bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_ell16_v2.txt none none
