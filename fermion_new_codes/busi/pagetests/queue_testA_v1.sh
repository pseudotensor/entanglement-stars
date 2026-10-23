#!/bin/bash
# Test A production queue on the box GPU 1 (inbox #428): the held control and the four recession rates, kappa 1 then 2.
# Both bands on the two check runs (kappa 1 held and L_eff 10); positive packets elsewhere (the massless chain's exact pairing).
set -u
cd ${QI2_ROOT}/code
PY=${QI2_ROOT}/python
PREP=${QI2_ROOT}/testA_prep_P64_s4_B2n4096_T100.json
OUT=${QI2_ROOT}/testA
LOG=${QI2_ROOT}/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
BOTH="-6,-3,-2,-1,-0.5,0.5,1,2,3,6"
POS="0.5,1,2,3,6"
run() { # kappa Leff omegas tag
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  echo "$(date -u +%FT%TZ) start $4 (GPU 1 memory used before: $used MiB)" >> $LOG/queue_testA_v1.log
  nice -n 10 $PY pt_testA_run_v1.py --prep $PREP --kappa $1 --Leff $2 --omegas=$3 --out $OUT --tag $4 --device cuda:1 > $LOG/$4.log 2>&1
  echo "$(date -u +%FT%TZ) end $4 rc $?" >> $LOG/queue_testA_v1.log
}
run 1 0 $BOTH A_k1_L0
run 1 10 $BOTH A_k1_L10
run 1 1 $POS A_k1_L1
run 1 3 $POS A_k1_L3
run 1 30 $POS A_k1_L30
run 2 0 $POS A_k2_L0
run 2 10 $POS A_k2_L10
run 2 1 $POS A_k2_L1
run 2 3 $POS A_k2_L3
run 2 30 $POS A_k2_L30
echo "$(date -u +%FT%TZ) queue done" >> $LOG/queue_testA_v1.log
