#!/bin/bash
# launch_s11_cpu_tests_v1.sh (builder analyst; S11 of the quasi-frozen ruling, brief lead/BRIEF_D_S11_builder.md: "test each new piece on
# the box CPU before any GPU launch; time one record first"). Box CPU, nice 10, 16 threads, run from ${QI2_ROOT}/engine:
# (1) the gate: engine_collapse_v32 and engine_collapse_v33 (every version 33 option off) on one small massive configuration, 3 records
#     each, compared by compare_v33_gate_v1 (every json number and fields array bitwise; wall-clock keys and script/args excluded);
# (2) S11 at r_s/a 16, G/a^2 1/4, m 0, ell 2, tau 2, 1024 sites (64 r_s), 10 channels, v32's solve settings of the sigma/a 2 member
#     (Newton-Krylov, 40 Krylov vectors, 2 restarts, halving line search, the two-stage source continuation, end above 1e-3), the field
#     module v17: the ball's preparation, the T = 0 record and one step (T/r_s 0.125), the record's seconds timed;
# (3) the mass-label characteristic predictor on the T = 0 profile of (2), with its unit tests.
set -u
cd ${QI2_ROOT}/engine || exit 2
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=${QI2_ROOT}/python
mkdir -p S11test GATE33
ST=S11test/status_s11_cpu_tests_v1.txt
echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $ST
COMMON="--m 0.2 --g 0.25 --rs 8 --R0-over-rs 4 --reading A --nsites 128 --ell 2 --kappa-sea 3 --tau 2 --T-over-rs 0.5 --reference R2 --site-fields averaged --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3"
nice -n 10 $PY code/engine_collapse_v32.py $COMMON --out GATE33 --tag gate_v32 > GATE33/gate_v32.log 2>&1
echo "gate v32 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
nice -n 10 $PY code/engine_collapse_v33.py $COMMON --out GATE33 --tag gate_v33 > GATE33/gate_v33.log 2>&1
echo "gate v33 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
nice -n 10 $PY code/compare_v33_gate_v1.py GATE33/gate_v32 GATE33/gate_v33 > GATE33/gate_compare_v1.txt 2>&1
echo "gate compare exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC'): $(tail -n 1 GATE33/gate_compare_v1.txt)" >> $ST
S11="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1"
nice -n 10 $PY code/engine_collapse_v33.py $S11 --T-over-rs 0.125 --out S11test --tag s11_cpu_n1024_v1 > S11test/s11_cpu_n1024_v1.log 2>&1
echo "s11 n1024 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
nice -n 10 $PY code/engine_masslabel_char_v1.py --run S11test/s11_cpu_n1024_v1 --nu 256 --dt 0.0625 --t-end-rs 5 --out-every-rs 0.125 --tau-engine 2 --unit-tests --out-json S11test/char_s11_cpu_n1024_v1.json > S11test/char_s11_cpu_n1024_v1.log 2>&1
echo "char exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
echo "done $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
