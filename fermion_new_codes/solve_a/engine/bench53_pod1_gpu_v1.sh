#!/bin/bash
# bench53_pod1_gpu_v1.sh (the lane's analyst builder; engine spec 40.4 (2) (d): "the gain in wall is read on the H100 only"): the #53 timing
# record on pod 1's H100, one run at a time (the device not shared): S11's records 1.125 to 1.5 from the stored T/r_s 1.0 slice, first
# engine_collapse_v35 --nk-jvp exact, then engine_collapse_v34's restart (the baseline), each --gpu 0 with 4 CPU threads (S11's own setting);
# each compared with the stored LZ records and with each other by compare_s11_segment_v2 (the floor). Run directory ${QI2_ROOT}/bench53_gpu.
D=${QI2_ROOT}/bench53_gpu; mkdir -p $D && cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; PY=python; S=$D/status.txt
TAG=s11_ball_rs16_ell2_tau2_v1; SRC=${QI2_ROOT}/s11cert_full
mkdir -p states/$TAG S11cert/snap jvp base
ln -sf $SRC/states/$TAG/orbitals_${TAG}_T0001.00.npz states/$TAG/
for f in $SRC/S11cert/snap/*; do ln -sf $f S11cert/snap/; done
COMMON="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0"
RES="--T-over-rs 40 --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG --segment-end-T-over-rs 1.5"
echo "gpu jvp start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
$PY $C/engine_collapse_v35.py $COMMON $RES --nk-jvp exact --out jvp --tag gpu_jvp > jvp/gpu_jvp.log 2>&1
echo "gpu jvp exit $? $(date -u '+%H:%M:%S UTC')" >> $S
$PY $C/compare_s11_segment_v2.py --mode segment --run jvp/gpu_jvp --ref S11cert/snap/$TAG --md gpu_jvp_vs_LZ.md > /dev/null 2>&1
echo "gpu base start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
$PY $C/engine_collapse_v34.py $COMMON $RES --out base --tag gpu_base > base/gpu_base.log 2>&1
echo "gpu base exit $? $(date -u '+%H:%M:%S UTC')" >> $S
$PY $C/compare_s11_segment_v2.py --mode segment --run base/gpu_base --ref S11cert/snap/$TAG --md gpu_base_vs_LZ.md > /dev/null 2>&1
$PY $C/compare_s11_segment_v2.py --mode segment --run jvp/gpu_jvp --ref base/gpu_base --md gpu_jvp_vs_base.md > /dev/null 2>&1
echo "gpu compared $(date -u '+%H:%M:%S UTC')" >> $S
