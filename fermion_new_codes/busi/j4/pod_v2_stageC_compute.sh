#!/bin/bash
# J4 version 2, stage C's computations on the pod (builder): the new balls of R 40 and 48 (the window's digital balls at
# boxes 4R, 6R, 8R as j4_production_v2.py plans them). The fit is not run (the version-2 fit's defects and B4's design are
# with reviewer A); the entropies do not depend on it.
cd ${QI2_ROOT}/j4
PY=python; J4=${QI2_ROOT}/j4; PROD1=${QI2_ROOT}/prod
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 J4_GPU_LOCK=${QI2_ROOT}/j4_gpu.lock
$PY ${QI2_ROOT}/j4_production_v2.py --radii 40 48 --outdir $J4/prod_v2 --v1-dirs $PROD1 --workers 20 --method auto --mem-budget-gb 200 > $J4/prod_v2_stageC.log 2>&1
echo "stageC production exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $J4/v2_stageC_status.txt
