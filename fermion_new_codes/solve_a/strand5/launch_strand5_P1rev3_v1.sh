#!/bin/bash
# Strand 5 production P1 revision 3 (JUDGMENT_strand5_production.md sec. 5.4 item 7, the lead's BUILDER item): the directional derivative of the
# exact gradient along the lowest full-node direction at P1's three trio members (z_c = 0.831894, 0.833894, 0.835894; P1run1's p1new family and
# its stage-2 version 10 record) against omega_1, as sec. 10(r)(a) (strand5_J4_softdir_v2.py --members 0,1,2), 2 threads; then (minutes) the
# record read of the (0.01, 0.1) twin and the minimum of V_eff over the zone per family (strand5_P1rev3_reads_v1.py).  Starts when the lane runs
# at most 14 busy processes.  Builder analyst.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=P1rev3
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 14 ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
KP=$D/J4kernels_run1
P1024="1:$D/J4kernels_run2/packets_C_m0.01_L1024.json,-1:$KP/packets_C_m0.10049876_L256.json,-1:$KP/packets_C_m0.10049876_L256.json,1:$KP/packets_C_m0.14177447_L256.json"
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 $PY code/strand5_J4_softdir_v2.py --family P1run1/lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json --packets $P1024 --stage2 P1run1/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json --members 0,1,2 --out $O > $O/softdir.log 2>&1
echo "softdir exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
