#!/bin/bash
# bench53_pod1_cpu_v2.sh (the lane's analyst builder; engine spec 40.4 (2) (c)): the fixed-record benchmark of bench53_pod1_cpu_v1.sh extended over
# G2's four records (S11's T/r_s 1.125 to 1.5 from the stored 1.0 slice, across the first trapping zero at 1.375), on pod 1's CPU (4 threads per
# run, nice 10), version 34's restart and version 35 --nk-jvp exact side by side (G1v35 passed in v1's run), each compared with the stored LZ
# records and with each other by compare_s11_segment_v2 --mode segment (the floor). Run directory ${QI2_ROOT}/bench53 (base4/, jvp4/).
D=${QI2_ROOT}/bench53; cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; PY=python; S=$D/status.txt
TAG=s11_ball_rs16_ell2_tau2_v1
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1"
RES="--T-over-rs 40 --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.5"
mkdir -p base4 jvp4
echo "bench4 start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
( nice -n 10 $PY $C/engine_collapse_v34.py $COMMON $RES --out base4 --tag bench4_base > base4/bench4_base.log 2>&1; echo "bench4 base exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P1=$!
( nice -n 10 $PY $C/engine_collapse_v35.py $COMMON $RES --nk-jvp exact --out jvp4 --tag bench4_jvp > jvp4/bench4_jvp.log 2>&1; echo "bench4 jvp exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P2=$!
wait $P2
$PY $C/compare_s11_segment_v2.py --mode segment --run jvp4/bench4_jvp --ref S11cert/snap/$TAG --md bench4_jvp_vs_LZ.md > /dev/null 2>&1
echo "bench4 jvp compared $(date -u '+%H:%M:%S UTC')" >> $S
wait $P1
$PY $C/compare_s11_segment_v2.py --mode segment --run base4/bench4_base --ref S11cert/snap/$TAG --md bench4_base_vs_LZ.md > /dev/null 2>&1
$PY $C/compare_s11_segment_v2.py --mode segment --run jvp4/bench4_jvp --ref base4/bench4_base --md bench4_jvp_vs_base.md > /dev/null 2>&1
echo "bench4 compared $(date -u '+%H:%M:%S UTC')" >> $S
