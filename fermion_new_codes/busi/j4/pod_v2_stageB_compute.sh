#!/bin/bash
# J4 version 2, stage B's computations on the pod (builder): the torch gate at R 40 and the new balls of R 12 to 34.
# The fit is not run here (j4_fit_v2's self-test defects are with reviewer A); the entropies do not depend on it.
cd ${QI2_ROOT}/j4
PY=python; CODE=${QI2_ROOT}/j4; J4=${QI2_ROOT}/j4; PROD1=${QI2_ROOT}/prod
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 J4_GPU_LOCK=${QI2_ROOT}/j4_gpu.lock
mkdir -p $J4/prod_v2/kcache $J4/checks_v2
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $J4/v2_stageB_status.txt; }
N40=$($PY -c "import j4window_v2 as jw; print(jw.ball_at_radius(40.0625))")
st "N40 $N40"
$PY $CODE/j4_reduced_v2.py --n2max $N40 --L 160 --window-centre 40 --method torch --kernel-cache $J4/prod_v2/kcache --out $J4/checks_v2/R40_torch_L160.json > $J4/checks_v2/R40_torch_L160.log 2>&1
st "torch gate exit $?"
$PY - <<'PYEOF' >> $J4/v2_stageB_status.txt
import json
S = json.load(open('${QI2_ROOT}/R40_torch_L160.json'))['S']
ref = 18553.885891843
print('torch gate S %.9f against the version-1 gram %.9f: relative %.2e -> %s' % (S, ref, abs(S - ref) / ref, 'PASS' if abs(S - ref) / ref < 1e-9 else 'FAIL'))
PYEOF
$PY $CODE/j4_production_v2.py --radii 12 16 20 24 29 34 --outdir $J4/prod_v2 --v1-dirs $PROD1 --workers 20 --method auto --mem-budget-gb 230 > $J4/prod_v2_stageB.log 2>&1
st "stageB production exit $?"
