#!/bin/bash
# run_probe_v1.sh NAME ENGINE(v36nq|rd13|rd13m|rd13x) NSITES K ARM(dense|banded|banded_st|stream) MAXWALL [extra engine args...]
# (busi #384) one memlaw probe on the box: the lad256 family's arguments scaled to NSITES (rs = NSITES/32, ell = tau = rs/16, prep-ell 4),
# fresh from T = 0, the Newton block of the reopen2 plan (pair workers 0: the field solves are host work, not device memory), K channels.
# MIX=plain swaps the Newton block for the plan's plain line. GPU: CUDA_VISIBLE_DEVICES=$GPU (default 1; never 0), THR threads (8), nice 10. Output: runs/NAME/{probe.json,log.txt,run/}.
set -u
NAME=$1; ENG=$2; NS=$3; K=$4; ARM=$5; MAXW=$6; shift 6
ROOT=${QI2_ROOT}/memlaw
PY=${QI2_ROOT}/python
GPU=${GPU:-1}
[ "$GPU" = 0 ] && { echo "never cuda:0" >&2; exit 2; }
case $ENG in v36nq) EF=$ROOT/code/engine_collapse_v36nq.py;; rd13) EF=$ROOT/code/engine_collapse_v36nq_rd13.py;; rd13m) EF=$ROOT/rd13m/code/engine_collapse_v36nq_rd13m.py;; rd13x) EF=$ROOT/rd13m/code/engine_collapse_v36nq_rd13.py;; *) echo bad engine; exit 2;; esac
case $ARM in dense) RD="";; stream) RD="--eigh banded --nk-cache-host on";; banded) RD="--eigh banded";; banded_st) RD="--eigh banded --jvp sternheimer";; *) echo bad arm; exit 2;; esac
[ "$ENG" = v36nq ] && [ -n "$RD" ] && { echo "the parent has no rd13 options" >&2; exit 2; }
RS=$((NS / 32)); ELL=$((RS / 16)); [ $ELL -lt 1 ] && ELL=1
W=$ROOT/runs/$NAME; mkdir -p $W/run
COMMON="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea $K --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes"
NK="--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --pair-workers 0"
[ "${MIX:-newton}" = plain ] && NK="--mix plain --mix-alpha 0.5 --iter-cache drop --prep-memory lean --pair-workers 0"   # the reopen2 plan's MIX_LAD256=plain line
ARGS="$COMMON --rs $RS --nsites $NS --ell $ELL --prep-ell 4 --tau $ELL $NK --out $W/run --tag memlaw_$NAME $RD $*"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THR:-8} MKL_NUM_THREADS=${THR:-8} OPENBLAS_NUM_THREADS=${THR:-8} NUMBA_NUM_THREADS=${THR:-8} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date -u +%FT%TZ) start $NAME $EF $(md5sum < $EF | cut -c1-8) GPU $GPU args: $ARGS" > $W/log.txt
cd $W && nice -n 10 $PY -u $ROOT/probe/memlaw_probe_v1.py --engine $EF --out $W/probe.json --max-wall $MAXW -- $ARGS >> $W/log.txt 2>&1
echo "$(date -u +%FT%TZ) rc $?" >> $W/log.txt
