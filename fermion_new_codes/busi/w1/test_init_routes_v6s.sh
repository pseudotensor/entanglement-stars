#!/bin/bash
# test_init_routes_v6s.sh (busi lane, #309 item 3): the host initial vacuum by the subset route (page_interior_init_v6s.py, zheevr range V) and the
# all-levels route (page_interior_init_mrrr_v6s.py, zheevr range A = MRRR) at nsites $1 on B2's T_0 fields, timed, then the two caches compared.
NS=$1; TH=${2:-8}; W=${QI2_ROOT}/init_route_test; S=${QI2_ROOT}/code_d13; PY=${QI2_ROOT}/python
B2=${QI2_ROOT}/B2_history; mkdir -p $W/cache; cd $W
ARGS="--record $B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json --fields $B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz --fields-key applied --nsites $NS --kappas 1-1 --interp linear --interp-sub 8 --cheb-tol 1e-16 --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --T-end-rs 500 --hold-fields-from 100 --lean-interior --device cpu --threads $TH --out runs/init_$NS"
export CUDA_VISIBLE_DEVICES= PYTHONPATH=${QI2_ROOT}/code_v6:$S PAGE_CODE_V6=${QI2_ROOT}/code_v6 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH
for R in subset mrrr; do
  if [ $R = subset ]; then E=$S/page_interior_init_v6s.py; else E=$S/page_interior_init_mrrr_v6s.py; fi
  t0=$(date +%s)
  /usr/bin/time -v nice -n 10 $PY $E $ARGS --init-cache $W/cache/n${NS}_$R.pt --tag t$NS$R > $W/init_${NS}_$R.log 2>&1
  echo "$R n $NS exit $? wall $(( $(date +%s) - t0 )) s maxrss_kB $(grep 'Maximum resident' $W/init_${NS}_$R.log | awk '{print $NF}')" >> $W/routes.txt
done
nice -n 10 $PY $S/cmp_init_cache_v6s.py $W/cache/n${NS}_subset.pt $W/cache/n${NS}_mrrr.pt > $W/cmp_$NS.txt 2>&1
echo "cmp n $NS exit $? $(tail -1 $W/cmp_$NS.txt)" >> $W/routes.txt
echo "DONE n $NS" >> $W/routes.txt
