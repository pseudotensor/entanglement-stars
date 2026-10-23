#!/bin/bash
# J4 version 2, stages D (R 56) and E (R 62) computations on the pod (builder), the version-2 driver as in RUN_ORDER_v3
# section 4. Launched while the stage-C gate judgment (G4 firing) is with reviewer A: the records are the window balls at W = 2
# above R 40, a subset of any wider window, and do not depend on the fit; D then E (the GPU lock serializes them).
cd ${QI2_ROOT}/j4
J4=${QI2_ROOT}/j4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 J4_GPU_LOCK=${QI2_ROOT}/j4_gpu.lock
for R in 56 62; do
  [ "$R" = 56 ] && S=D || S=E
  python ${QI2_ROOT}/j4_production_v2.py --radii $R --outdir $J4/prod_v2 \
    --v1-dirs $J4/prod --wide-upto 40 --box-ladder-upto 48 --redo-v1 \
    --workers 12 --mem-budget-gb 230 --mem-limit-gb 16 --method auto --torch-from 40 \
    > $J4/stage$S.out 2>&1
  echo "stage$S production exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $J4/v2_stageDE_status.txt
done
