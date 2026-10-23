#!/bin/bash
# launch_s11_cert_segment_jvp_v1.sh (the lane's analyst builder; engine spec 41.1 (2)): the combined construction, engine_collapse_v35
# --nk-jvp exact --weyl-certified eigvals --weyl-oracle, over the same 4 records (T/r_s 1.125 to 1.5) from the S11 run's stored T/r_s 1.0
# complex128 slice, on box GPU 0 with S11's arguments and 4 CPU threads (launch_s11_cert_segment_v1.sh's line with v35 and the option); the gate:
# within the floor of the stored LZ records (compare_s11_segment_v2 --mode segment) and of the certified segment (--mode routes against
# S11cert/seg), masks equal, every certificate passing; its wall per record is the certified pace the owner's r_s/a 32 decision reads.
# Usage (from ${QI2_ROOT}/engine): setsid nohup bash code/launch_s11_cert_segment_jvp_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
S=S11cert/status.txt
TAG=s11_ball_rs16_ell2_tau2_v1
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0"
mkdir -p S11cert/segjvp S11cert/segjvp_slices
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "segment jvp start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $S
nice -n 10 $PY code/engine_collapse_v35.py $COMMON --T-over-rs 40 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir S11cert/segjvp_slices \
    --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.5 \
    --weyl-certified eigvals --weyl-oracle --nk-jvp exact --out S11cert/segjvp --tag s11cert_segjvp_v1 > S11cert/segjvp/s11cert_segjvp_v1.log 2>&1
echo "segment jvp exit $? $(date -u '+%H:%M:%S UTC')" >> $S
$PY code/compare_s11_segment_v2.py --mode segment --run S11cert/segjvp/s11cert_segjvp_v1 --ref S11cert/snap/$TAG --md S11cert/segjvp_vs_LZ_v2.md --out-json S11cert/segjvp_vs_LZ_v2.json > /dev/null 2>&1
$PY code/compare_s11_segment_v2.py --mode routes --run S11cert/segjvp/s11cert_segjvp_v1 --ref S11cert/seg/s11cert_seg_v1 --md S11cert/segjvp_vs_seg_v2.md --out-json S11cert/segjvp_vs_seg_v2.json > /dev/null 2>&1
echo "segment jvp compared $(date -u '+%H:%M:%S UTC')" >> $S
