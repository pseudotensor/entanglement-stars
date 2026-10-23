#!/bin/bash
# rc_ruling_v1 6.1 ("the self-test's D, UD and L lines are bitwise with the option off"; busi builder): engine_collapse_v36n_selftest.py (90501b60)
# in an aliased copy of the ISO code tree: engine_collapse_v36j.py = v36nq (8549c542, the host), engine_collapse_v36n.py = v37n (v36nq + --confirm-test,
# default off). Cases J, N, D, L, U, UD on the box CPU (nice 10, 8 threads): G-A N = J, G-A' D = J, G-A'' L = J, G-A* UD = U.
D=${QI2_ROOT}/v36n_selftest; PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
md5sum $D/code/engine_collapse_v36j.py $D/code/engine_collapse_v36n.py $D/code/engine_collapse_v36n_selftest.py > $D/inputs_md5.txt
cd $D/code && nice -n 10 $PY engine_collapse_v36n_selftest.py --code-dir $D/code --out $D/out --cases J,N,D,L,U,UD --threads 8 > $D/selftest_v37n_cpu.log 2>&1; echo "rc $?" >> $D/selftest_v37n_cpu.log
touch $D/DONE
