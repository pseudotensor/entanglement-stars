#!/bin/bash
# pod_diag_C_s2_v1.sh (busi lane): short diagnostics of C's sigma/a 2 start failure on pod 3's H100, while the reviewer A judgment
# (reviewa_C_s2_fix_20261017_221803) is pending; inbox #109 (the card is not to idle). Each run is pod_launch_C_v5.sh's s2 argument set with
# --nk-jvp exact (version 4's launch), no slices, to T/r_s 0.125 (records 0, 0.0625, 0.125), into its own out dir and tag; nothing touches
# results/C32. The diagnostics, one hypothesis each:
#   krylov80  --nk-krylov 80 --nk-restarts 3           the Krylov budget (production: 40 vectors, 0 restarts)
#   v34eig    --light-transport v34-eigvals             the light reference's continuation (re-selection inside the Newton loop)
#   off       --light-coupling off                      the uncoupled control with this exact argument set (gate C3 ran it against v35)
# krylov80 and v34eig run together; off runs after krylov80 ends. Diagnostic only: no record of these runs is C's history.
B=${QI2_ROOT}/busi_compC; stamp=$(date -u +%Y%m%dT%H%M%SZ); D=$B/diag/$stamp; mkdir -p $D
cd $B || exit 1
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
base() {  # $1 nk-krylov $2 nk-restarts $3 light-coupling $4 light-transport
echo "--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 0.125 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov $1 --nk-restarts $2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling $3 --light-transport $4 --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --ell 2"
}
st() { echo "diag_s2 $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }
run() {  # $1 name, rest: base args
  local n=$1; shift; mkdir -p $D/$n/states
  st "$n launch load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) out $D/$n"
  nice -n 10 python $B/code/engine_twofield_v3.py $(base "$@") --slices-c128 "" --slices-light-c128 "" --slice-dir $D/$n/states --gpu 0 --out $D/$n --tag C32_s2_diag_$n > $D/$n/run.log 2>&1
  st "$n exit $?"
}
run krylov80 80 3 on banded &
P1=$!
sleep 90
run v34eig 40 0 on v34-eigvals &
P2=$!
wait $P1
run off 40 0 off banded
wait $P2
st "all done $D"
