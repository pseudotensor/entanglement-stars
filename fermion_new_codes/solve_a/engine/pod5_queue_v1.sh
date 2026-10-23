#!/bin/bash
# pod5_queue_v1.sh (the lane's analyst builder; engine spec 42.2 (4)): pod 5's ordered list after S11's restart (PID given): (1) S11's recursion
# on this GPU (launch_s11_page_v2's collectors cut8 and cut16, page_history_v6i, the history the restart's json and fields npz, which carry
# the records from T = 0; both on cuda:0 at once, 1024 sites); (2) the preparation scan of 42.1 (2): engine_collapse_v35 with the cost run's
# arguments (r_s/a 32, ell 2a, tau 2, 2048 sites, --nk-jvp exact) and --ball-orbitals 1:6, 1:12, 1:24 to T = 0 only, one after another.
# The rest of 42.2 (4)'s order (item 3's variant runs on v6j, the ell 2a member under 42.1 (4)'s pace condition, C, the absorber gate) is
# launched by the lane when each is ready. Usage (pod 5): setsid nohup bash pod5_queue_v1.sh <S11 PID> &
set -u
SPID=${1:?S11 PID}
W=/workspace; LOG=$W/pod5_queue_v1.log
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "queue v1 start; waiting for S11 PID $SPID ($(tr '\0' ' ' < /proc/$SPID/cmdline | cut -c1-80))"
while [ -e /proc/$SPID ]; do sleep 60; done
log "S11 ended: $(tail -c 300 $W/s11v35/s11_v35from20.log | tr '\n' ' ')"
H=$W/s11v35; TAG=s11_ball_rs16_ell2_tau2_v35from20
cd $W/page || exit 2; mkdir -p runs/s11 logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for NAME in cut8 cut16; do
  case "$NAME" in cut8) CUT="--R-cut-rs 8 --R-collar-rs 5.333333333333333";; cut16) CUT="--R-cut-rs 16 --R-collar-rs 10.666666666666666";; esac
  ( log "start s11pagei_$NAME"
    nice -n 10 python code_v6/page_history_v6i.py --record $H/collapse_$TAG.json --fields $H/collapse_${TAG}_fields.npz --fields-key applied --nsites 1024 --channels 1,2 $CUT --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --join-rs 32,64 --max-wall-hours 20 --device cuda:0 --threads 8 --out runs/s11 --tag s11pagei_$NAME > logs/s11pagei_$NAME.log 2>&1
    log "exit s11pagei_$NAME $?" ) &
done
wait
cd $W/engine || exit 2; mkdir -p prepscan
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ARGS="--m 0 --g 0.25 --rs 32 --nsites 2048 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact"
for B in 6 12 24; do
  log "start prepscan b$B"
  ( cd prepscan && nice -n 10 python $W/engine/code/engine_collapse_v35.py $ARGS --ball-orbitals 1:$B --T-over-rs 0 --out . --tag prepscan_rs32_ell2_b$B > prepscan_b$B.log 2>&1 )
  log "exit prepscan b$B $?"
done
log "queue v1 done"
