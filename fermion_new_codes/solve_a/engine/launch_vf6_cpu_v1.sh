#!/bin/bash
# The two engine_vacflow_v6 controls of engine 16.6 on the CPU now (GPU 2 is busy with the J-VF vf3 runs until about 20:40 UTC): the R2 column
# with the degeneracy-aware continuation (r_s 16, exact, massless, to kappa 16) and the full chain current at t_r = 2 r_s; 8 threads each, nice
# 10, in parallel; outputs VF/vacflow_<tag>.json and logs VF/<tag>_cpu.log. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
run() { local tag=$1; shift; echo "vf6 cpu $tag start $(date -u +%H:%M:%S) UTC (8 threads)" >> status.txt
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY code/engine_vacflow_v6.py "$@" --out VF --tag $tag > VF/${tag}_cpu.log 2>&1 < /dev/null
  echo "vf6 cpu $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt; }
run vf6_rs16_exact_m0_R2aware --rs 16 --profile exact --m 0 --kmax 16 --r2 &
run vf6_rs16_exact_m0_fullcurrent --rs 16 --profile exact --m 0 --kmax 16 --r2 --chain-current full --tr-list 2 &
wait
