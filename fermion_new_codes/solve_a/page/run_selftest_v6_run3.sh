#!/bin/bash
# the self-test v6 on the box CPU (builder analyst; engine spec 37.1 (3)): every v5 test in v5 mode (with T-V5REPRO against the stored v5
# run 2) and in v6 mode, T5, T7a, T7b and the dense oracle T8 at eps_rank 0 and 1e-9; nice 10, 16 threads
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
date -u +"start %H:%M:%S UTC" > selftest_v6_run3.marker
nice -n 10 ${QI2_ROOT}/python page_recursion_v6.py --selftest --device cpu --threads 16 --engine-path ${QI2_ROOT}/code --obs-path ${QI2_ROOT}/code --out runs/selftest_v6_run3 --v5-selftest-results ${QI2_ROOT}/selftest_results.json > selftest_v6_run3.log 2>&1
echo "exit $? $(date -u +%H:%M:%S) UTC" >> selftest_v6_run3.marker
