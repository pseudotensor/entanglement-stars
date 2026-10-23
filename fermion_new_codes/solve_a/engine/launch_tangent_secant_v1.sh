#!/bin/bash
# engine_tangent_secant_v1 (engine spec 36.4 (1)) once the two variance-tangent members have ended: waits for both members' exit lines in
# the GPU queue's status.txt (launch_engine_gpu_v4.sh writes "GPU<G> <tag> exit <rc> <time> UTC"), then runs the table on the box CPU (nice 10,
# 2 threads) into TAN27/tangent_secant_v1.md; the lane copies it to results/engine/LAD27/. Builder analyst.
# Usage (on the box, detached): setsid nohup bash TAN27/launch_tangent_secant_v1.sh > /dev/null 2>&1 < /dev/null &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
T0=c27_rs64_A_R2_tau8_ell3p6_prep4_avg; T1=c27_rs64_A_R2_tau8_ell4p4_prep4_avg
echo "tangent_secant_v1 waiter start $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
until grep -q "GPU0 $T0 exit" status.txt && grep -q "GPU1 $T1 exit" status.txt; do sleep 60; done
echo "tangent_secant_v1 start $(date -u +%H:%M:%S) UTC ($(grep "GPU0 $T0 exit" status.txt); $(grep "GPU1 $T1 exit" status.txt))" >> TAN27/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_tangent_secant_v1.py \
  --member 3.6=C3tan27/collapse_$T0.json --member 4.4=C3tan27/collapse_$T1.json \
  --member 4=H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg.json --member 8=H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json \
  --fields4 H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg_fields.npz --md TAN27/tangent_secant_v1.md > TAN27/log_tangent_secant_v1.log 2>&1
echo "tangent_secant_v1 exit $? $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
