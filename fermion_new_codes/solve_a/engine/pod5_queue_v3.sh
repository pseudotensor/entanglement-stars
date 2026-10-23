#!/bin/bash
# pod5_queue_v3.sh (the lane's analyst builder; engine spec 43.2 (2), 43.1 (4)): pod 5's order under turn 51. Now, beside S11: the preparation
# scan on engine_collapse_v36 (--field-kernel-cache --v36-fixes --nk-jvp exact; r_s/a 32, ell 2a, tau 2, 2048 sites; --ball-orbitals 1:6, 1:12,
# 1:24; to T = 0 only, the call tested on v35). When S11 (PID given) ends: its recursion cut8 then cut16 (page_history_v6i on the restart's
# records); the preparation chosen by 42.1 (2)'s rule (the smallest set whose bare-energy fraction lies within a factor 2 of S11's 0.720019
# with the discarded norm below 9.39e-02; else the set nearest the rule, labelled) and the ell 2a member launched on v36 with it (T/r_s 40,
# complex128 slices at every integer T/r_s), its running mean wall per record printed every 4 r_s by pace_monitor_v1.py, a mean above 135 s
# flagged; then item 3's two variants (page_history_v6j, A's s4 settings: --collector-select gen, then --cubic-window 32,40). The held-field
# reruns on v6j with the signed collar split are inserted by the lane when v6j (v) is built. Usage (pod 5): setsid nohup bash pod5_queue_v3.sh <S11 PID> &
set -u
SPID=${1:?S11 PID}; W=/workspace; LOG=$W/pod5_queue_v3.log; C=$W/engine/code
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
COST="--m 0 --g 0.25 --rs 32 --nsites 2048 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact --field-kernel-cache --v36-fixes"
log "queue v3 start; S11 PID $SPID ($(tr '\0' ' ' < /proc/$SPID/cmdline | cut -c1-60))"
mkdir -p $W/engine/prepscan36; cd $W/engine/prepscan36
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for B in 6 12 24; do
  log "start prepscan36 b$B"; nice -n 10 python $C/engine_collapse_v36.py $COST --ball-orbitals 1:$B --T-over-rs 0 --out . --tag prepscan36_b$B > b$B.log 2>&1; log "exit prepscan36 b$B $?"
done
CH=$(python - <<'PY'
import re, math
best = None; near = None
for B in (6, 12, 24):
    t = open("b%d.log" % B).read()
    m = re.search(r"the bare energy \(weighted\) [-0-9.e+]+ = ([-0-9.e+]+) E;.*?discarded norm max ([-0-9.e+]+)", t)
    if not m:
        continue
    f, d = float(m.group(1)), float(m.group(2))
    ok = 0.720019 / 2 <= f <= 2 * 0.720019 and d < 9.39e-2
    if ok and best is None:
        best = B
    dist = abs(math.log(max(f, 1e-300) / 0.720019))
    if near is None or dist < near[1]:
        near = (B, dist)
print(best if best else ("%d nearest" % near[0] if near else "none"))
PY
)
log "the preparation chosen: $CH"
while [ -e /proc/$SPID ]; do sleep 60; done
log "S11 ended"
H=$W/s11v35; TAG=s11_ball_rs16_ell2_tau2_v35from20
cd $W/page || exit 2; mkdir -p runs/s11 logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for NAME in cut8 cut16; do
  case "$NAME" in cut8) CUT="--R-cut-rs 8 --R-collar-rs 5.333333333333333";; cut16) CUT="--R-cut-rs 16 --R-collar-rs 10.666666666666666";; esac
  log "start s11pagei_$NAME"; nice -n 10 python code_v6/page_history_v6i.py --record $H/collapse_$TAG.json --fields $H/collapse_${TAG}_fields.npz --fields-key applied --nsites 1024 --channels 1,2 $CUT --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --join-rs 32,64 --max-wall-hours 20 --device cuda:0 --threads 8 --out runs/s11 --tag s11pagei_$NAME > logs/s11pagei_$NAME.log 2>&1; log "exit s11pagei_$NAME $?"
done
B=${CH%% *}
if [ "$B" != "none" ]; then
  mkdir -p $W/engine/pair_ell2a; cd $W/engine/pair_ell2a
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  SL=$(seq -s, 1 40)
  log "start the ell 2a member (preparation 1:$B, $CH)"
  ( nice -n 10 python $C/engine_collapse_v36.py $COST --ball-orbitals 1:$B --T-over-rs 40 --slices-c128 $SL --slice-dir states --out . --tag pair_rs32_ell2a_v36 > pair_rs32_ell2a_v36.log 2>&1; log "exit the ell 2a member $?" ) &
  sleep 60; nohup python $C/pace_monitor_v1.py collapse_pair_rs32_ell2a_v36.json 32 4 135 >> $W/engine/pair_ell2a/pace.log 2>&1 &
else
  log "no preparation read from the scan: the member waits for the lane"
fi
cd $W/page; mkdir -p runs/v6j
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
BASE="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0"
log "start v6jA_s4gen"; nice -n 10 python code_v6/page_history_v6j.py $BASE --collector-select gen --out runs/v6j/v6jA_s4gen --tag v6jA_s4gen --resume > logs/v6jA_s4gen.log 2>&1; log "exit v6jA_s4gen $?"
log "start v6jA_s4cub"; nice -n 10 python code_v6/page_history_v6j.py $BASE --cubic-window 32,40 --out runs/v6j/v6jA_s4cub --tag v6jA_s4cub --resume > logs/v6jA_s4cub.log 2>&1; log "exit v6jA_s4cub $?"
log "queue v3 done"
