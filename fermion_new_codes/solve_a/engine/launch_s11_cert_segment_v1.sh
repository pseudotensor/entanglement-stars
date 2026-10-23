#!/bin/bash
# launch_s11_cert_segment_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 39.4 (2), verbatim: "a certified segment of 2
# records across the formation, from the stored complex128 slice nearest before T/r_s 1.375 (the builder names it), with ceil(2 ||dh||/gap)
# sub-steps per channel so that the Weyl certificate passes on every channel (a channel that still fails marked), on box GPU 0 ...; compared with
# the LZ run at the same clocks in E/E0, the roots, min(N - |v|) and the masks"; the lane's approval of 01:16 UTC): the slice is the S11 run's
# T/r_s 1.0 complex128 slice (states/s11_ball_rs16_ell2_tau2_v1/, the nearest stored before 1.375; S11 stores complex128 slices at T/r_s 0, 1,
# 2, 3, ...), so the segment runs the 4 records T/r_s 1.125, 1.25, 1.375 and 1.5 across the formation (the first trapped record 1.375):
# engine_collapse_v34 with S11's own arguments, --weyl-certified eigvals (the eigenvalue route, allowed by G3's oracle), --weyl-oracle (the
# full route and version 3's one-step continuation at every record's accepted fields), --segment-end-T-over-rs 1.5; box GPU 0, nice 10, 4 CPU
# threads; then compare_s11_segment_v1.py --mode segment against the stored LZ records (the lead's criterion printed). Output S11cert/seg/,
# S11cert/segment_vs_LZ_v1.md and .json. Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_s11_cert_segment_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
S=S11cert/status.txt
TAG=s11_ball_rs16_ell2_tau2_v1
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0"
mkdir -p S11cert/seg S11cert/seg_slices
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "segment start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> $S
nice -n 10 $PY code/engine_collapse_v34.py $COMMON --T-over-rs 40 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir S11cert/seg_slices \
    --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.5 \
    --weyl-certified eigvals --weyl-oracle --out S11cert/seg --tag s11cert_seg_v1 > S11cert/seg/s11cert_seg_v1.log 2>&1
echo "segment exit $? $(date -u '+%H:%M:%S UTC')" >> $S
$PY code/compare_s11_segment_v1.py --mode segment --run S11cert/seg/s11cert_seg_v1 --ref S11cert/snap/$TAG --md S11cert/segment_vs_LZ_v1.md --out-json S11cert/segment_vs_LZ_v1.json > S11cert/segment_vs_LZ_v1.txt 2>&1
echo "segment compared $(date -u '+%H:%M:%S UTC')" >> $S
