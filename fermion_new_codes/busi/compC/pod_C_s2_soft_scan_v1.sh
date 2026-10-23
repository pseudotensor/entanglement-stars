#!/bin/bash
# pod_C_s2_soft_scan_v1.sh (busi lane): the soft-direction scan at the sigma/a 2 member's T = 0 floor iterates (C_s2_fold_reading_v1.md, section 1) on
# pod 3's H100: s = 0 (the uncoupled map) and s = 1 (the coupled map), the floors from gate v6's runs in results/gateV6 (re-run by the engine when absent),
# v from soft_gain_v6, 15 complement-corrected line points per s, a sign change polished and given to the engine's own loop; under about 1 hour, under 30 GiB.
B=${QI2_ROOT}/busi_compC; O=$B/results/soft_scan_v1; mkdir -p $O
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "softscan $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }
st "launch load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
cd $B/code && nice -n 10 python C_s2_soft_scan_v1.py --engine-dir $B/code --gate-dir $B/results/gateV6 --out $O --gpu 0 > $O/soft_scan_v1.log 2>&1
st "exit $?: $(grep -E 'VERDICT s = |SOFT SCAN DONE' $O/soft_scan_v1.log | tail -3 | tr '\n' ' ')"
