#!/bin/bash
# launch_s11_cert_gates_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 39.4 (2) and the lane's approval of 01:16 UTC):
# the gates of engine_collapse_v34 on box GPU 0 before the certified S11 segment, started when the slices job (engine_single_object_slices_v2)
# has written its last status line, with S11's own launch arguments (code/launch_s11_ball_v1.sh, md5 e9aaa8a34ca216b28601364c2ec06131):
# (G1) the bitwise gate: engine_collapse_v33 and engine_collapse_v34 with every version 34 option off, the same arguments, 3 records
#      (--T-over-rs 0.25, no slices), side by side on GPU 0 (1 CPU thread each), compared by compare_v33_gate_v1.py (every json number but the
#      wall-clock keys and script/args, every fields array, bitwise);
# (G2) the resume gate: engine_collapse_v34 restarted from the stored complex128 slice at T/r_s 1.0 of the S11 run (its files snapshotted
#      md5-checked in S11cert/snap/ at 01:53:37 UTC) on version 3's continuation (no certification), the 4 records T/r_s 1.125 to 1.5
#      (--segment-end-T-over-rs 1.5; 2 CPU threads), compared with the stored records by compare_s11_segment_v1.py --mode gate.
# The certified segment itself runs only after the lane has the gates' results and the launch line (launch_s11_cert_segment_v1.sh).
# Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_s11_cert_gates_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
S=S11cert/status.txt
mkdir -p S11cert/g1_v33 S11cert/g1_v34 S11cert/g2 S11cert/g2_slices
until grep -q "slices_v2_s8 exit" S4one/v2/status.txt 2>/dev/null; do sleep 30; done
TAG=s11_ball_rs16_ell2_tau2_v1
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0"
echo "G1 start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
( export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; nice -n 10 $PY code/engine_collapse_v33.py $COMMON --T-over-rs 0.25 --out S11cert/g1_v33 --tag g1_v33 > S11cert/g1_v33/g1_v33.log 2>&1; echo "G1 v33 exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P1=$!
( export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; nice -n 10 $PY code/engine_collapse_v34.py $COMMON --T-over-rs 0.25 --out S11cert/g1_v34 --tag g1_v34 > S11cert/g1_v34/g1_v34.log 2>&1; echo "G1 v34 exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P2=$!
echo "G2 start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; nice -n 10 $PY code/engine_collapse_v34.py $COMMON --T-over-rs 40 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir S11cert/g2_slices \
    --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.5 --out S11cert/g2 --tag g2_resume > S11cert/g2/g2_resume.log 2>&1; echo "G2 exit $? $(date -u '+%H:%M:%S UTC')" >> $S ) &
P3=$!
wait $P1 $P2
$PY code/compare_v33_gate_v1.py S11cert/g1_v33/g1_v33 S11cert/g1_v34/g1_v34 > S11cert/gate1_v34_vs_v33.txt 2>&1
echo "G1 compared $(date -u '+%H:%M:%S UTC'): $(tail -1 S11cert/gate1_v34_vs_v33.txt)" >> $S
wait $P3
$PY code/compare_s11_segment_v1.py --mode gate --run S11cert/g2/g2_resume --ref S11cert/snap/$TAG --out-json S11cert/gate2_resume.json > S11cert/gate2_resume.txt 2>&1
echo "G2 compared $(date -u '+%H:%M:%S UTC'): $(grep -o 'GATE [A-Z]*' S11cert/gate2_resume.txt | tail -1)" >> $S
echo "gates done $(date -u '+%H:%M:%S UTC')" >> $S
