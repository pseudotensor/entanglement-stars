#!/bin/bash
# pod_gate_v6_v1.sh (busi lane): the gates of engine_twofield_v6 on pod 3's H100 (C_s2_start_judgment_v3.md): G1 bitwise
# off at sigma/a 4 (v5 against v6), G2 the uncoupled limit at sigma/a 2, G3 the coupling homotopy at T = 0 and the covariance homotopy after, the soft gain g printed; 3 records each, 2048 sites, sequential.
B=${QI2_ROOT}/busi_compC; O=$B/results/gateV6; mkdir -p $O
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "gateV6 $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }
st "launch load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
cd $B/code && nice -n 10 python gate_v6_s2_v1.py --engine-dir $B/code --out $O --gpu 0 --records 3 --nsites 2048 > $O/gate_v6.log 2>&1
st "exit $?: $(grep -E 'PASS|FAIL' $O/gate_v6.log | tail -1)"
