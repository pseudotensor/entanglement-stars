#!/bin/bash
# J4 ladder on the lane's H100 pod (builder analyst), following RUN_ORDER.md stages 2 to 7 back to back so the pod
# never idles: stage 2 (the 3 controls as records), stage 3 (pilot R 12-24, all boxes, CPU gram), stage 4 (R 29.0625,
# L 116 by gram, svd and the builder's torch route; the ladder stops unless all three agree to 1e-9 relative),
# stage 5 (R 29, 34, 40 at 4R and 6R, CPU gram), stage 6 (R 48, 56, 62 at 4R, torch route with the GPU lock),
# stage 7 (the fit). The pod's cgroup gives 22 CPUs and 251 GB; workers 20 on the CPU stages, 6 on stage 6.
# Usage on the pod: cd ${QI2_ROOT}/j4 && setsid nohup bash ${QI2_ROOT}/pod_ladder_v1.sh > ladder_v1.log 2>&1 < /dev/null &
set -u
cd ${QI2_ROOT}/j4
PY=python
CODE=${QI2_ROOT}/j4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 J4_GPU_LOCK=${QI2_ROOT}/j4_gpu.lock
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a ladder_status_v1.txt; }
mkdir -p controls fits checks_gram checks_svd checks_torch

st "stage2 start"
for c in "4 64" "6 128" "6.5 80"; do set -- $c
  $PY $CODE/j4_dense.py --Reff $1 --L $2 --threads 1 --out controls/dense_R$1_L$2.json > controls/dense_R$1_L$2.log 2>&1
  $PY $CODE/j4_reduced.py --Reff $1 --L $2 --threads 1 --check-dense --check-dense4 --out controls/reduced_R$1_L$2.json > controls/reduced_R$1_L$2.log 2>&1
done
st "stage2 exit"

st "stage3 start"
$PY $CODE/j4_production.py --radii 12 16 20 24 --workers 20 --outdir prod --python $PY > prod_stage3.log 2>&1
st "stage3 exit $?"
$PY $CODE/j4_fit.py --indir prod --allow-partial --out fits/J4_fit_pilot.json > fits/J4_fit_pilot.log 2>&1
st "pilot fit exit $?"

st "stage4 start"
$PY $CODE/j4_production.py --radii 29 --offsets 1 --boxes 4 --method gram --workers 1 --outdir checks_gram --python $PY > checks_gram.log 2>&1
$PY $CODE/j4_production.py --radii 29 --offsets 1 --boxes 4 --method svd --workers 1 --outdir checks_svd --python $PY > checks_svd.log 2>&1
$PY $CODE/j4_production.py --radii 29 --offsets 1 --boxes 4 --method torch --workers 1 --outdir checks_torch --python $PY > checks_torch.log 2>&1
$PY - <<'EOF'
import glob, json, sys
S = {}
for m in ('gram', 'svd', 'torch'):
    f = glob.glob('checks_%s/R29_o+1_L116.json' % m)
    S[m] = json.load(open(f[0]))['S'] if f else float('nan')
rel = max(abs(S[a] - S[b]) / abs(S['gram']) for a in S for b in S)
print('stage4 S', S, 'max relative difference %.3e' % rel)
open('stage4_gate.txt', 'w').write('%s %.3e\n' % ('PASS' if rel < 1e-9 else 'FAIL', rel))
sys.exit(0 if rel < 1e-9 else 1)
EOF
gate=$?
st "stage4 gate $(cat stage4_gate.txt)"
if [ $gate -ne 0 ]; then st "STOP: the spectral routes disagree at R 29"; exit 1; fi

st "stage5 start"
$PY $CODE/j4_production.py --radii 29 34 40 --workers 20 --mem-budget-gb 200 --outdir prod --python $PY > prod_stage5.log 2>&1
st "stage5 exit $?"
$PY $CODE/j4_fit.py --indir prod --allow-partial --out fits/J4_fit_stage5.json > fits/J4_fit_stage5.log 2>&1
st "stage5 fit exit $?"

st "stage6 start"
$PY $CODE/j4_production.py --radii 48 56 62 --workers 6 --mem-budget-gb 200 --mem-limit-gb 16 --method torch --outdir prod --python $PY > prod_stage6.log 2>&1
st "stage6 exit $?"

st "stage7 start"
$PY $CODE/j4_fit.py --indir prod --out fits/J4_fit_v1.json > fits/J4_fit_v1.log 2>&1
st "stage7 exit $?"
st "all done"
