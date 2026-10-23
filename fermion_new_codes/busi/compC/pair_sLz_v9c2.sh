#!/bin/bash
# REF (v9c1) and ON (v9c2, all options) side by side, 4 threads each, sector lagged lz, no slice: the same box load for an indicative wall (busi PERF C-A1..C-A5)
ST=${QI2_ROOT}/v9c2; cd $ST
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 \"\" --slices-light-c128 \"\" --slices-c128 \"\" --reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps \"\""
mkdir -p g0runs/pair/REF g0runs/pair/ON
eval nice -n 10 $PY engine_twofield_v9c1.py $E2E --out g0runs/pair/REF --tag e2e_v9cL > g0runs/pair/REF/run.log 2>&1 &
eval nice -n 10 $PY engine_twofield_v9c2.py $E2E --out g0runs/pair/ON --tag e2e_v9cL --perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on > g0runs/pair/ON/run.log 2>&1 &
wait
$PY cmp_v9c2_runs.py g0runs/pair/REF/e2e_v9cL g0runs/pair/ON/e2e_v9cL --b-log g0runs/pair/ON/run.log
for arm in REF ON; do $PY -c "
import json; J=json.load(open(\"g0runs/pair/$arm/collapse_e2e_v9cL.json\")); R=[r for r in J[\"records\"] if r.get(\"E\") is not None]
s=[r.get(\"seconds\",0) for r in R]; e=[(r.get(\"timing\") or {}).get(\"eigh_s\",0) for r in R]; d=[(r.get(\"timing\") or {}).get(\"densities_s\",0) for r in R]
print(\"$arm record seconds\", [round(x,1) for x in s], \"sum %.1f; heavy eigh_s sum %.1f; densities_s sum %.2f\" % (sum(s), sum(e), sum(d)))"; done
echo "== pair done"
