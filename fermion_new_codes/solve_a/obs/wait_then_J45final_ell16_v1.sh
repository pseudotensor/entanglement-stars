#!/bin/bash
# Engine 16.5's GPU 2 order after the r_s = 32 set: waits until the engine's GPU 2 runner reports "GPU2 DONE list_gpu2_v5.txt", then runs the
# subagent's list_final_v1.txt (obs/J45, launch_obs_J45_v1.sh: the validation, smoke, precision control, J4 at L = 48, the vacuum-flow box,
# P36, the escape and detector boxes, the timings), then the ell = 16 collapse control at r_s 64 on engine_collapse_v9 (list_gpu2_ell16_v1.txt
# through launch_engine_gpu_v4.sh on GPU 2). Usage (on the box, detached, from obs/J45): nohup bash wait_then_J45final_ell16_v1.sh > launcher_final_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/J45 || exit 1
echo "J45 final waiter start $(date -u +%Y-%m-%dT%H:%M:%SZ) (after GPU2 DONE list_gpu2_v5.txt)" >> status.txt
until grep -q "^GPU2 DONE list_gpu2_v5.txt" ${QI2_ROOT}/status.txt; do sleep 60; done
sleep 30
bash code/launch_obs_J45_v1.sh list_final_v1.txt
cd ${QI2_ROOT}/engine || exit 1
bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_ell16_v1.txt none none
