#!/bin/bash
# Indicative side-by-side timing (busi PERF C-A7), 4 threads each, nice 10, sector lagged lz, no slice, the same box load: v9c2 with every v9c2
# option (ON2) against v9c3 with those and --perf-solve-memo on (ON3); and v9c2 at the defaults (D2) against v9c3 with the memo alone (M3).
ST=${QI2_ROOT}/v9c3; cd $ST
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
E2E="--m 0.2 --g 0.25 --rs 32.0 --R0-over-rs 4 --reading A --nsites 1024 --prep-ell 4 --ell 4 --kappa-sea 10 --tau 2.0 --T-over-rs 0.1875 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --slices-c64 \"\" --slices-light-c128 \"\" --slices-c128 \"\" --reference-solve lagged --cut-transport sector --anchor-passage lz --m1-simpson-steps \"\""
ALLV9C2="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
W=g0runs/pair; rm -rf $W; mkdir -p $W/ON2 $W/ON3 $W/D2 $W/M3
eval nice -n 10 $PY engine_twofield_v9c2.py $E2E --out $W/ON2 --tag e2e_v9cL $ALLV9C2 > $W/ON2/run.log 2>&1 &
eval nice -n 10 $PY engine_twofield_v9c3.py $E2E --out $W/ON3 --tag e2e_v9cL $ALLV9C2 --perf-solve-memo on > $W/ON3/run.log 2>&1 &
eval nice -n 10 $PY engine_twofield_v9c2.py $E2E --out $W/D2 --tag e2e_v9cL > $W/D2/run.log 2>&1 &
eval nice -n 10 $PY engine_twofield_v9c3.py $E2E --out $W/M3 --tag e2e_v9cL --perf-solve-memo on > $W/M3/run.log 2>&1 &
wait
$PY cmp_v9c3_runs.py $W/ON2/e2e_v9cL $W/ON3/e2e_v9cL --b-log $W/ON3/run.log
$PY cmp_v9c3_runs.py $W/D2/e2e_v9cL $W/M3/e2e_v9cL --b-log $W/M3/run.log
for arm in ON2 ON3 D2 M3; do $PY -c "
import json; J=json.load(open(\"$W/$arm/collapse_e2e_v9cL.json\")); R=[r for r in J[\"records\"] if r.get(\"E\") is not None]
s=[r.get(\"seconds\",0) for r in R]; t=[(r.get(\"timing\") or {}) for r in R]
print(\"$arm record seconds\", [round(x,1) for x in s], \"sum %.1f; solve_s sum %.2f (solve_n %s); heavy eigh_s sum %.1f\" % (sum(s), sum(x.get(\"solve_s\",0) for x in t), [x.get(\"solve_n\") for x in t], sum(x.get(\"eigh_s\",0) for x in t)))"; done
find $W -name "*.npz" -delete
echo "== pair done"
