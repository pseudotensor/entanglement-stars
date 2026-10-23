#!/bin/bash
# Task 5 version 6 reruns (flow_chart_v6.py: the root fix, the centre margin, the symbol series): E1 (3 masses) and E4 (N 4096, m 0 and 0.03) on one GPU. Builder analyst.
# control, E3 (CPU), E2 (2 shards in sequence unless --e2-gpu given). One process at a time; nice 10; 1 thread each.  Builder analyst.
GPU=${1:-0}
cd ${QI2_ROOT}/flow
mkdir -p task5
PY=${QI2_ROOT}/python
mine_on_gpu() { local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}'); for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u{print $2}'); do ps -o user= -p $p 2>/dev/null | grep -qx jon && echo $p; done; }
until grep -q "^DONE$" task5/status.txt 2>/dev/null; do sleep 30; done   # queued behind launch_task5_v1.sh (E2) on the same GPU
if [ -n "$(mine_on_gpu)" ]; then echo "GPU $GPU busy with this lane's process: $(mine_on_gpu)"; exit 1; fi
run() { local name=$1; shift; echo "task5 $name start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> task5/status_v6.txt; nice -n 10 $PY code/flow_chart_v6.py "$@" > task5/$name.log 2>&1; echo "task5 $name exit $? $(date -u +%H:%M:%S)" >> task5/status_v6.txt; }
run v6_e1_m0 e1 --masses 0 --gpu $GPU --output task5/v6_e1_m0.json
run v6_e1_m0.03 e1 --masses 0.03 --gpu $GPU --output task5/v6_e1_m0.03.json
run v6_e1_m0.1 e1 --masses 0.1 --gpu $GPU --output task5/v6_e1_m0.1.json
run v6_e4_N4096 e4 --masses 0 0.03 --nsites 4096 --gpu $GPU --output task5/v6_e4_N4096.json
echo "DONE v6" >> task5/status_v6.txt
