#!/bin/bash
# #331 gates v1b (busi builder): the v36z selftest rerun after two harness fixes (the log's command line; the srf snapshot's pinned flag), then
# the solve lane's selftest_srf_v1b on the ORIGINAL v36q_srf (77857541) to see whether its behaviour failure on v36y is the test's own
B=${QI2_ROOT}/tourn331; PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
cp $B/build/engine_collapse_v36z_selftest.py $B/code/engine_collapse_v36z_selftest.py
cp -n $B/src/engine_collapse_v36q_srf.py $B/code/
cd $B/code && nice -n 10 $PY engine_collapse_v36z_selftest.py --code $B/code --cli --whole --out $B/selftest/whole_b --threads 8 > $B/selftest/selftest_v36z_whole_b.log 2>&1; echo "rc $?" >> $B/selftest/selftest_v36z_whole_b.log
cd $B/code && ENGINE_OLD=$B/code/engine_collapse_v36q.py ENGINE_NEW=$B/code/engine_collapse_v36q_srf.py SRF_OUT=$B/selftest/srf_v36q_srf nice -n 10 $PY selftest_srf_v1b.py > $B/selftest/selftest_srf_v1b_v36q_srf.log 2>&1; echo "rc $?" >> $B/selftest/selftest_srf_v1b_v36q_srf.log
touch $B/selftest/GATES_B_DONE
