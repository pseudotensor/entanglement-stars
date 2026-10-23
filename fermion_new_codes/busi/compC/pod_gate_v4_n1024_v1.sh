#!/bin/bash
# pod_gate_v4_n1024_v1.sh (busi lane): the chain-length test after GATE V4 FAILED at 2048 sites (stage A, the uncoupled quench, unconverged at
# T = 0 with kept 1.88e-04; the judgment's reading "A unconverged: the chain length"). G2 and G3 of gate_v4_s2_v1.py at --nsites 1024, the
# production's chain: does the uncoupled quench converge there, and does the coupled start finish? Diagnostic only; 3 records each.
B=${QI2_ROOT}/busi_compC; O=$B/results/gateV4_n1024; mkdir -p $O
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "gateV4_n1024 $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }
st "launch load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
cd $B/code && nice -n 10 python gate_v4_s2_v1.py --engine-dir $B/code --out $O --gpu 0 --records 3 --nsites 1024 --only g2,g3 > $O/gate_v4_n1024.log 2>&1
st "exit $?: $(grep -E 'GATE V4' $O/gate_v4_n1024.log | tail -1)"
