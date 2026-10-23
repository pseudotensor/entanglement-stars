#!/bin/bash
# run_probe_iso_v1.sh NAME ENGINE(v36nq|rd13|rd13m|rd13x) ARM(dense|banded|banded_st|stream) MAXWALL [extra engine args]
# (busi #384) one memlaw probe at 4096 sites, K 10: ISO_128_s4 nkqh2 resumed from its T0005.00 slice (the reopen2 stage's hard links,
# read only) with the stored run's arguments (as the RD-1 builder's replay runs them), one record (segment end 5.0625), no slice written.
set -u
NAME=$1; ENG=$2; ARM=$3; MAXW=$4; shift 4
ROOT=${QI2_ROOT}/memlaw
PY=${QI2_ROOT}/python
GPU=${GPU:-1}
[ "$GPU" = 0 ] && { echo "never cuda:0" >&2; exit 2; }
case $ENG in v36nq) EF=$ROOT/code/engine_collapse_v36nq.py;; rd13) EF=$ROOT/code/engine_collapse_v36nq_rd13.py;; rd13m) EF=$ROOT/rd13m/code/engine_collapse_v36nq_rd13m.py;; rd13x) EF=$ROOT/rd13m/code/engine_collapse_v36nq_rd13.py;; *) echo bad engine; exit 2;; esac
case $ARM in dense) RD="";; stream) RD="--eigh banded --nk-cache-host on";; banded) RD="--eigh banded";; banded_st) RD="--eigh banded --jvp sternheimer";; *) echo bad arm; exit 2;; esac
IN=${QI2_ROOT}/inputs
OLD=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
W=$ROOT/runs/$NAME; mkdir -p $W/run
ARGS="--m 0.2 --g 0.25 --rs 128.0 --R0-over-rs 4.0 --reading A --nsites 4096 --ell 4 --prep-ell 4 --field-rtol 1e-12 --field-atol 1e-16 --kappa-sea 10 \
--tau 8.0 --T-over-rs 40.0 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 --out $W/run --tag memlaw_$NAME \
--site-fields averaged --stop-E 1e-07 --stop-fields 1e-06 --maxit 20 --mix newton --nk-maxiter 30 --nk-inner 15 --mix-alpha 0.5 --delta-rtol-every 10 \
--commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --nk-line-search halving --nk-halvings 6 --nk-krylov 40 \
--nk-restarts 2 --end-above 0.001 --ball-R0-over-rs 4.0 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --ball-k0-max 0.5 \
--ball-E-tol 1e-10 --ball-passes 200 --ball-tol 1e-10 --s11-collars 8,16 --s11-stop 1 --weyl-refine-max 10 --field-kernel-cache --v36-fixes \
--nk-jvp exact --pair-workers 0 --nk-set resolve --dust-set isotropic --dust-multiplicity 1 --iter-cache drop --prep-memory lean \
--nk-host-release on --nk-plain-fallback on --resume-slice $IN/states_h2/orbitals_${OLD}_T0005.00.npz --resume-run $IN/run_h2/$OLD \
--segment-end-T-over-rs 5.0625 $RD $*"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THR:-8} MKL_NUM_THREADS=${THR:-8} OPENBLAS_NUM_THREADS=${THR:-8} NUMBA_NUM_THREADS=${THR:-8} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date -u +%FT%TZ) start $NAME $EF $(md5sum < $EF | cut -c1-8) GPU $GPU args: $ARGS" > $W/log.txt
cd $W && nice -n 10 $PY -u $ROOT/probe/memlaw_probe_v1.py --engine $EF --out $W/probe.json --max-wall $MAXW -- $ARGS >> $W/log.txt 2>&1
echo "$(date -u +%FT%TZ) rc $?" >> $W/log.txt
