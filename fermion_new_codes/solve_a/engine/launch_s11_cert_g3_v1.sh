#!/bin/bash
# launch_s11_cert_g3_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 39.4 (2); the lane's condition of 01:16 UTC: "the
# cheaper route ... allowed only if its oracle reproduces the full route exactly on a test record, both cost figures printed"): (G3) the test
# record: engine_collapse_v34 restarted from the S11 run's T/r_s 1.0 complex128 slice with the certified continuation on the eigenvalue route
# (--weyl-certified eigvals) and the full-route oracle at the accepted fields (--weyl-oracle), one record (T/r_s 1.125), on box GPU 0 (nice 10,
# 4 CPU threads), started only after G1 and G2 have printed GATE PASS (S11cert/status.txt); the record's certificate and oracle printed by
# compare_s11_segment_v1.py --mode segment against the stored LZ record. Output S11cert/g3/.
# Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_s11_cert_g3_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
S=S11cert/status.txt
until grep -q "gates done" $S 2>/dev/null; do sleep 30; done
if ! grep -q "G1 compared.*GATE PASS" $S || ! grep -q "G2 compared.*GATE PASS" $S; then echo "G3 not started: G1 or G2 did not pass $(date -u '+%H:%M:%S UTC')" >> $S; exit 3; fi
TAG=s11_ball_rs16_ell2_tau2_v1
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0"
mkdir -p S11cert/g3 S11cert/g3_slices
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "G3 start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
nice -n 10 $PY code/engine_collapse_v34.py $COMMON --T-over-rs 40 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir S11cert/g3_slices \
    --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.125 \
    --weyl-certified eigvals --weyl-oracle --out S11cert/g3 --tag g3_cert_eigvals > S11cert/g3/g3_cert_eigvals.log 2>&1
echo "G3 exit $? $(date -u '+%H:%M:%S UTC')" >> $S
$PY code/compare_s11_segment_v1.py --mode segment --run S11cert/g3/g3_cert_eigvals --ref S11cert/snap/$TAG --md S11cert/g3_test_record.md --out-json S11cert/g3_test_record.json > S11cert/g3_test_record.txt 2>&1
echo "G3 compared $(date -u '+%H:%M:%S UTC')" >> $S
