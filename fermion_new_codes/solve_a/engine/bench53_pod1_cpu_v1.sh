#!/bin/bash
# bench53_pod1_cpu_v1.sh (the lane's analyst builder; engine spec 40.4 (2) (b), (c), the lead's instruction: #53 step (b) on a free CPU lane):
# on pod 1's CPU (no GPU; nice 10; 4 threads per run), in ${QI2_ROOT}/bench53:
# (G1v35) the bitwise gate: engine_collapse_v34 and engine_collapse_v35 with every option off (v35's --nk-jvp absent), S11's arguments, 3 records
#   (--T-over-rs 0.25), side by side, compared by compare_v33_gate_v1.py (every json number but the wall-clock keys and script/args, every
#   fields array, bitwise);
# then, only if G1v35 passes, (c) the fixed record: S11's T/r_s 1.125 from the stored T/r_s 1.0 complex128 slice (G2's record), version 34's
#   restart (the baseline on this device) and version 35 with --nk-jvp exact, side by side; each compared by compare_s11_segment_v2.py --mode
#   segment against the stored LZ record (the floor E/E0 1.5e-07, roots 1.7e-06, min(N - |v|) 2.3e-08) and the two against each other; the
#   evaluations, eigendecompositions and wall per record printed by bench53_compare_v1.py.
# The inputs: the S11 slice and snapshot under ${QI2_ROOT}/s11cert_full (md5-checked copies of the box's); the modules from ${QI2_ROOT}/code.
# Usage (pod 1): setsid nohup bash ${QI2_ROOT}/bench53_pod1_cpu_v1.sh &
D=${QI2_ROOT}/bench53; mkdir -p $D && cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; PY=python; S=$D/status.txt
TAG=s11_ball_rs16_ell2_tau2_v1; SRC=${QI2_ROOT}/s11cert_full
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1"
mkdir -p g1_v34 g1_v35 base jvp states/$TAG S11cert/snap
ln -sf $SRC/states/$TAG/orbitals_${TAG}_T0001.00.npz states/$TAG/
for f in $SRC/S11cert/snap/*; do ln -sf $f S11cert/snap/; done
echo "G1v35 start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
( nice -n 10 $PY $C/engine_collapse_v34.py $COMMON --T-over-rs 0.25 --out g1_v34 --tag g1_v34 > g1_v34/g1_v34.log 2>&1; echo "G1 v34 exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P1=$!
( nice -n 10 $PY $C/engine_collapse_v35.py $COMMON --T-over-rs 0.25 --out g1_v35 --tag g1_v35 > g1_v35/g1_v35.log 2>&1; echo "G1 v35 exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P2=$!
wait $P1 $P2
$PY $C/compare_v33_gate_v1.py g1_v34/g1_v34 g1_v35/g1_v35 > gate1_v35_vs_v34.txt 2>&1
echo "G1v35 compared $(date -u '+%H:%M:%S UTC'): $(tail -1 gate1_v35_vs_v34.txt)" >> $S
grep -q "GATE PASS" gate1_v35_vs_v34.txt || { echo "the benchmark not started: G1v35 did not pass" >> $S; exit 3; }
RES="--T-over-rs 40 --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.125"
echo "bench start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
( nice -n 10 $PY $C/engine_collapse_v34.py $COMMON $RES --out base --tag bench_base > base/bench_base.log 2>&1; echo "bench base exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P1=$!
( nice -n 10 $PY $C/engine_collapse_v35.py $COMMON $RES --nk-jvp exact --out jvp --tag bench_jvp > jvp/bench_jvp.log 2>&1; echo "bench jvp exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P2=$!
wait $P1 $P2
$PY $C/compare_s11_segment_v2.py --mode segment --run base/bench_base --ref S11cert/snap/$TAG --md bench_base_vs_LZ.md > /dev/null 2>&1
$PY $C/compare_s11_segment_v2.py --mode segment --run jvp/bench_jvp --ref S11cert/snap/$TAG --md bench_jvp_vs_LZ.md > /dev/null 2>&1
$PY $C/compare_s11_segment_v2.py --mode segment --run jvp/bench_jvp --ref base/bench_base --md bench_jvp_vs_base.md > /dev/null 2>&1
echo "bench compared $(date -u '+%H:%M:%S UTC')" >> $S
