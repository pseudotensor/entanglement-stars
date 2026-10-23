#!/bin/bash
# Strand S diagnostic D-a1 (SPEC sec. 9.1; strandS_dos_v2.py): the sea's density of states at fixed boundary charge on the static chart's
# locked background L, sizes E0 = 8 and 16 (r_s0 = 16, 32; r_h/h about 15 and 30) at h = 1 on the CPU (one thread each, in parallel), the
# three markers (N_s = 1/2, 4h/r_h, h T_H), the massless packet and the light rectangle's masses at aM = 0.1, R_max = 20 r_s0; E0 = 32 follows
# on a GPU.  Gated on the lane's busy processes (at most 15).  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p Da1
export STRAND7_CODE=${QI2_ROOT}/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for E in 8 16; do
  ( until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
    date -u "+Da1 E$E start %H:%M:%S UTC (cpu)" >> Da1/status.txt
    nice -n 10 ${QI2_ROOT}/python code/strandS_dos_v2.py --E0 $E --Rmax-over-rs 20 --device cpu --out-json Da1/dos_E$E.json > Da1/run_E$E.log 2>&1
    echo "Da1 E$E exit $? $(date -u +%H:%M:%S)" >> Da1/status.txt ) &
  sleep 40
done
wait
echo DONE >> Da1/status.txt
