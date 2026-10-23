#!/bin/bash
# bench53_pod1_gpu_v2.sh (the lane's analyst builder; engine spec 40.4 (2) (c), (d); an interim massive record while the ruled one waits): the
# ruled massive record (the production's T/r_s 28 complex128 slice) needs a restart path from the production's v19-family records that version
# 34's resume lacks; so that pod 1's H100 does not idle, the massive configuration on version 32's solver settings is benchmarked from its own
# initial data instead: the sigma/a 2 ladder member's exact arguments (engine_collapse_v32 on pod 2, untouched) run to T/r_s 0.5 (records 0 to
# 0.5) by engine_collapse_v35 --nk-jvp exact and by engine_collapse_v34 (bitwise v32 with its options off), one at a time on the H100, 4 CPU
# threads; the per-record evaluations, eigendecompositions and walls and the two runs' differences per record are read afterwards (a record
# beside the ruled benchmark, not in place of it). Run directory ${QI2_ROOT}/bench53_gpu2.
D=${QI2_ROOT}/bench53_gpu2; mkdir -p $D/jvp $D/base && cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; PY=python; S=$D/status.txt
ARGS="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 0.5 --prep-ell 4 --tau 8 --gpu 0 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3"
echo "massive jvp start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
$PY $C/engine_collapse_v35.py $ARGS --nk-jvp exact --out jvp --tag m_jvp > jvp/m_jvp.log 2>&1
echo "massive jvp exit $? $(date -u '+%H:%M:%S UTC')" >> $S
echo "massive base start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
$PY $C/engine_collapse_v34.py $ARGS --out base --tag m_base > base/m_base.log 2>&1
echo "massive base exit $? $(date -u '+%H:%M:%S UTC')" >> $S
