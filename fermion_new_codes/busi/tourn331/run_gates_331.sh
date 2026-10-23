#!/bin/bash
# #331 gates on the box CPU (busi builder): v36z selftest part 3 (whole engines), then the solve lane's selftest_srf_v1b with ENGINE_NEW = v36y
B=${QI2_ROOT}/tourn331; PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
cd $B/code && nice -n 10 $PY engine_collapse_v36z_selftest.py --code $B/code --cli --whole --out $B/selftest/whole --threads 8 > $B/selftest/selftest_v36z_whole.log 2>&1; echo "rc $?" >> $B/selftest/selftest_v36z_whole.log
cp -n $B/src/selftest_srf_v1b.py $B/code/
cd $B/code && ENGINE_OLD=$B/code/engine_collapse_v36q.py ENGINE_NEW=$B/code/engine_collapse_v36y.py SRF_OUT=$B/selftest/srf_v36y nice -n 10 $PY selftest_srf_v1b.py > $B/selftest/selftest_srf_v1b_v36y.log 2>&1; echo "rc $?" >> $B/selftest/selftest_srf_v1b_v36y.log
touch $B/selftest/GATES_DONE
