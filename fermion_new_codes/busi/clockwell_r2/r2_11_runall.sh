#!/bin/bash
# Second reader driver: run r2_10_cohort.py on all 16 even cohorts (8 filtered + 8 on-site). nice 10, 4 BLAS threads.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="nice -n 10 ${QI2_ROOT}/python -I"
C=${QI2_ROOT}/code; O=${QI2_ROOT}/clockwell_r2; B=${QI2_ROOT}/solve_b_bulk
for n in 2 3; do for g in 0 0p5 1 2; do
  c=even_n${n}_g${g}_v1
  ro=$B/MOCK_filtered_comparison_readouts_v1/$c
  [ -d $ro ] || ro=${QI2_ROOT}/$c
  echo "== filtered $c ($ro)"; $PY $C/r2_10_cohort.py $B/MOCK_filtered_physical_v1/$c filtered $ro $O/r2_filtered_$c.json
  echo "== onsite $c"; $PY $C/r2_10_cohort.py $B/MOCK_unitary_lead_run_v1/$c onsite $B/MOCK_comparison_readouts_v1/$c $O/r2_onsite_$c.json
done; done
