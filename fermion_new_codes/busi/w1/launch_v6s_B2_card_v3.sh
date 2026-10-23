#!/bin/bash
# launch_v6s_B2_card_v3.sh (busi f52b1152, inbox #309; v3 = v2 (1f694466) with two hold-mode guards: at 32768 sites it REFUSES unless the initial-vacuum
# cache $INIT exists (and, when W1_INIT_MD5 is set, carries that md5), because without it the engine's host subset eigensolve takes about 12 h or more
# (measured 655 s at 8192 sites, O(n m^2)); below 32768 sites without a cache it passes --init-eig-device device (minutes on the card). kz unchanged.)
# v2's header: busi 9ae0d5c4, inbox #309, the Page-curve verification plan; v2 = v1 with page_interior_v6s 15e08e28 (the lean in-place propagator) and hold mode's --impurity-every-rs 50: the solve lane's two v6s jobs on B2's history, in
# launch_v6q_B2_card_v1.sh's (677fd00a) form: the production's arguments, nice 10, NUMPY_MADVISE_HUGEPAGE=0, exits with the engine's rc.
#   MODE kz   (D1, A2's conformal control): kappa 1's chain with the mass term kappa/R off (--kappa-zero), B2's fields, 8192 sites, to 100 r_s,
#             the production's cuts 2,16,24,30,48 r_s and readouts; tag v6s_B2_kz (outputs runs/v6s_B2_kz/v6s_B2_kz_k1_*).
#   MODE hold (B2, the held-record drain): kappa 1, ${W1_NSITES:-32768} sites, the fields held from the 100 r_s record to 500 r_s, the lean interior
#             mode, interior readouts every 10 r_s (cuts 2,16,24,30,48 r_s); tag v6s_B2hold (runs/v6s_B2hold/v6s_B2hold_k1_{summary.json,interior.md});
#             the initial vacuum from $W1_INIT (the box's cache, its h_+ md5 checked; computed on the host when absent, hours at 32768 sites).
# Both resume (--resume) from their saved state after a stop. The code dir holds code_v6/ and code_d13/; the run dir gets runs/, logs/, w1/.
# Usage: W1_PY=... [W1_MAX_WALL_HOURS=h] [W1_INIT=path] [W1_CODE=dir] [W1_RUNDIR=dir] [W1_B2=dir] [W1_NSITES=n] [W1_INIT_MD5=md5] bash launch_v6s_B2_card_v3.sh <card> kz|hold
export NUMPY_MADVISE_HUGEPAGE=0; CARD=$1; MODE=$2
CODE=${W1_CODE:-${QI2_ROOT}/page}; RUNDIR=${W1_RUNDIR:-${QI2_ROOT}/page}; D=${W1_B2:-${QI2_ROOT}/B2}; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
cd $RUNDIR || exit 2
export PYTHONPATH=$CODE/code_v6:$CODE/code_d13 PAGE_CODE_V6=$CODE/code_v6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MD5_V6S=611d190f5047d0a65dfc1c72866cc69e; MD5_IR=15e08e2807a3bce6e565306de023e80f
m1=$(md5sum $CODE/code_d13/page_history_v6s.py | cut -d' ' -f1); m2=$(md5sum $CODE/code_d13/page_interior_v6s.py | cut -d' ' -f1)
if [ "$m1" != "$MD5_V6S" ] || [ "$m2" != "$MD5_IR" ]; then echo "v6s_B2_$MODE REFUSED on $CARD $(date -u +%FT%TZ): md5 v6s $m1 (want $MD5_V6S), interior $m2 (want $MD5_IR)" >> v6o_status.txt; exit 1; fi
COMMON="--record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --kappas 1-1 --kappa-hard-max 1 --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --device cuda:0 --resume"
case $MODE in
  kz)   TAG=v6s_B2_kz; MAXH=${W1_MAX_WALL_HOURS:-8}
        ARGS="$COMMON --nsites 8192 --kappa-zero --sector-check-every-rs 10 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs 100 --save-every 4 --threads 4 --columns-at-rs 12,20,40 --columns-every-rs 0" ;;
  hold) NS=${W1_NSITES:-32768}; TAG=v6s_B2hold; [ "$NS" != 32768 ] && TAG=v6s_B2hold_n$NS; MAXH=${W1_MAX_WALL_HOURS:-30}
        INIT=${W1_INIT:-$RUNDIR/w1/B2_n${NS}_k1_T0_Phiplus.pt}
        ARGS="$COMMON --nsites $NS --T-end-rs 500 --hold-fields-from 100 --hold-step-rs 1 --lean-interior --interior-every-rs 10 --impurity-every-rs 50 --save-every 1 --init-cache $INIT --threads 8"
        if [ ! -s "$INIT" ] && [ ! -e runs/$TAG/${TAG}_k1_istate.pt ]; then
          if [ "$NS" = 32768 ]; then echo "$TAG REFUSED on $CARD $(date -u +%FT%TZ): no initial-vacuum cache $INIT (copy the box's; the host route takes 12 h or more)" >> v6o_status.txt; exit 1; fi
          ARGS="$ARGS --init-eig-device device"
        fi
        if [ -s "$INIT" ] && [ -n "$W1_INIT_MD5" ]; then mi=$(md5sum "$INIT" | cut -d' ' -f1)
          if [ "$mi" != "$W1_INIT_MD5" ]; then echo "$TAG REFUSED on $CARD $(date -u +%FT%TZ): cache $INIT md5 $mi (want $W1_INIT_MD5)" >> v6o_status.txt; exit 1; fi; fi ;;
  *) echo "MODE kz or hold"; exit 2 ;;
esac
mkdir -p runs/$TAG logs w1
echo "$TAG launch on $CARD $(date -u +%FT%TZ) cap_h $MAXH gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')" >> v6o_status.txt
nice -n 10 ${W1_PY:-python} $CODE/code_d13/page_history_v6s.py $ARGS --out runs/$TAG --tag $TAG --max-wall-hours $MAXH >> logs/$TAG.log 2>&1 &
echo $! > w1/$TAG.pid; wait $!; rc=$?
echo "$TAG exit $rc $(date -u +%FT%TZ)" >> v6o_status.txt
exit $rc
