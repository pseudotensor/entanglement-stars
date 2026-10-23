#!/bin/bash
# GATECOMP step (iii): the cold dust comparator (solve's dust_comparator_v1d.py b0112336, run read-only from its own dir; control mode = L2 0,
# released from coordinate rest, averaged alike) on ISO_128_s4's own T = 0 profile at sigma/a 4. Box CPU, nice 10, 8 threads. Output small.
set -e
B=${QI2_ROOT}/dustcomp_iso128_s4
CODE=${QI2_ROOT}/dust_comparator_v1d.py
SL=${QI2_ROOT}/orbitals_lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n_T0000.00.npz
REC=${QI2_ROOT}/collapse_lad_ISO_128_s4_rd_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_rd2.json
[ -e $B ] && { echo "refusing: $B exists"; exit 1; }
mkdir -p $B
md5sum $CODE $SL > $B/inputs_md5.txt
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
nohup nice -n 10 ${QI2_ROOT}/python $CODE --slice $SL --records $REC --sigma 4 --modes control --rs 128 --R0-head 512 \
  --T-end 13.0 --dt-rec 8 --budget-seconds 30000 --out $B/out > $B/run.out 2>&1 < /dev/null &
echo "pid $!"
