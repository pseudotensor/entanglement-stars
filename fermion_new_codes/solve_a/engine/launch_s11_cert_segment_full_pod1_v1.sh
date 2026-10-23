#!/bin/bash
# launch_s11_cert_segment_full_pod1_v1.sh (the lane's analyst builder; engine spec 39.4 (2) as ruled: the certified segment with ceil(2 ||dh||/gap)
# sub-steps per channel and version 3's maximum-overlap assignment at every sub-step, the full route): launch_s11_cert_segment_v1.sh with
# --weyl-certified full (no oracle: the full route is the oracle's own route) on pod 1's H100 (inbox #54 and the lead: the free H100 takes the next
# critical GPU item), beside the eigenvalue-route segment on box GPU 0, so that the ruled construction does not wait on the eigenvalue route's
# oracle. The inputs copied md5-checked from the box: the S11 run's T/r_s 1.0 complex128 slice and the run's files snapshotted at 01:53:37 UTC
# (S11cert/snap/); the engine's modules from ${QI2_ROOT}/code (md5-equal to the box's). Run directory ${QI2_ROOT}/s11cert_full (container disk);
# 8 CPU threads. Then compare_s11_segment_v1.py --mode segment against the stored LZ records. Usage (pod 1): setsid nohup bash THIS &
D=${QI2_ROOT}/s11cert_full
cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=python
S=$D/status.txt
TAG=s11_ball_rs16_ell2_tau2_v1
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0"
mkdir -p seg_full seg_full_slices
echo "segment (full route) start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $S
$PY ${QI2_ROOT}/engine_collapse_v34.py $COMMON --T-over-rs 40 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir seg_full_slices \
    --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.5 \
    --weyl-certified full --out seg_full --tag s11cert_seg_full_v1 > seg_full/s11cert_seg_full_v1.log 2>&1
echo "segment (full route) exit $? $(date -u '+%H:%M:%S UTC')" >> $S
$PY ${QI2_ROOT}/compare_s11_segment_v1.py --mode segment --run seg_full/s11cert_seg_full_v1 --ref S11cert/snap/$TAG --md segment_full_vs_LZ_v1.md --out-json segment_full_vs_LZ_v1.json > segment_full_vs_LZ_v1.txt 2>&1
echo "segment (full route) compared $(date -u '+%H:%M:%S UTC')" >> $S
