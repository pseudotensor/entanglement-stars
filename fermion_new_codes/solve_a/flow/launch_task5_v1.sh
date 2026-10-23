#!/bin/bash
# Task 5 (SPEC_task5_black_object.md) on one GPU, sequential: E1 (3 masses), E4 (N 4096, m 0 and 0.03), E1's R_max control, E4's N 2048
# control, E3 (CPU), E2 (2 shards in sequence unless --e2-gpu given). One process at a time; nice 10; 1 thread each.  Builder analyst.
GPU=${1:-0}
cd ${QI2_ROOT}/flow
mkdir -p task5
PY=${QI2_ROOT}/python
mine_on_gpu() { local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}'); for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u{print $2}'); do ps -o user= -p $p 2>/dev/null | grep -qx jon && echo $p; done; }
if [ -n "$(mine_on_gpu)" ]; then echo "GPU $GPU busy with this lane's process: $(mine_on_gpu)"; exit 1; fi
run() { local name=$1; shift; echo "task5 $name start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> task5/status.txt; nice -n 10 $PY code/flow_chart_v5.py "$@" > task5/$name.log 2>&1; echo "task5 $name exit $? $(date -u +%H:%M:%S)" >> task5/status.txt; }
run e1_m0 e1 --masses 0 --gpu $GPU --output task5/e1_m0.json
run e1_m0.03 e1 --masses 0.03 --gpu $GPU --output task5/e1_m0.03.json
run e1_m0.1 e1 --masses 0.1 --gpu $GPU --output task5/e1_m0.1.json
run e4_N4096 e4 --masses 0 0.03 --nsites 4096 --gpu $GPU --output task5/e4_N4096.json
run e1_rmax e1 --masses 0 --kappas -1 --nsites 4096 --rmax-control --gpu $GPU --output task5/e1_rmax_control.json
run e4_N2048 e4 --masses 0 --nsites 2048 --gpu $GPU --output task5/e4_N2048_control.json
run e4_robs12 e4 --masses 0 --kappas -1 --nsites 4096 --robs-over-rs 12 --gpu $GPU --output task5/e4_robs12_control.json
run e3 e3 --output task5/e3.json
run e2_s0 e2 --rs 32 --Rm 8 --masses 0 0.1 --shard 0 --shards 2 --gpu $GPU --output task5/e2_shard0.json
run e2_s1 e2 --rs 32 --Rm 8 --masses 0 0.1 --shard 1 --shards 2 --gpu $GPU --output task5/e2_shard1.json
run e2_rmax e2 --rs 32 --Rm 8 --masses 0 --omega-rs 4 --e2-rmax-control --gpu $GPU --output task5/e2_rmax_control.json
echo "DONE" >> task5/status.txt
