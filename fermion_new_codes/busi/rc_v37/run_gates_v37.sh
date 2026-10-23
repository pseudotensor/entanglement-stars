#!/bin/bash
# rc_ruling_v1 6.1 gates (busi builder): engine_collapse_v37_selftest.py --cli --whole on the box CPU (nice 10, 8 threads)
B=${QI2_ROOT}/rc_v37; PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
cd $B/code && nice -n 10 $PY engine_collapse_v37_selftest.py --code $B/code --cli --whole --out $B/selftest/whole2 --threads 8 > $B/selftest/selftest_v37_whole2.log 2>&1; echo "rc $?" >> $B/selftest/selftest_v37_whole2.log
touch $B/selftest/GATES_WHOLE2_DONE
