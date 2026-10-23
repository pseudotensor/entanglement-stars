#!/bin/bash
# launch_s11_page_v1.sh (builder analyst; S11 of the quasi-frozen ruling: "the Page recursion of S1 running alongside with its collectors
# at 8 and 16 r_s on a chain to 64 r_s"): page_history_v6c.py (S1's code, unchanged) on the S11 history (engine_collapse_v33's record json
# and fields npz, the applied fields), the chain the engine's own 1024 sites (64 r_s at r_s/a 16: nothing appended), channels 1, 2 of the
# massless operator (the same species as the ball), S1's settings with the collectors scaled: R_cut 8 r_s with the collar 16/3 r_s (the
# primary) and R_cut 16 r_s with the collar 32/3 r_s (the far collector), S1's collar-to-cut ratio 2/3; collection every 2 r_s; the
# transfer ledger at the S_esc peak; the join check over 32 to 64 r_s (the engine's own exterior against the vacuum exterior at E_k).
# Usage (from ${QI2_ROOT}/page on the box): bash ../engine/code/launch_s11_page_v1.sh <cut8|cut16> [device: cpu | cuda:N]
# [T-end in r_s, default the history's last record]. Runs on the engine's records as they stand (re-run with --resume as more land).
set -u
NAME=${1:?cut8 or cut16}
DEV=${2:-cpu}
TEND=${3:-}
cd ${QI2_ROOT}/page || exit 2
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=${QI2_ROOT}/python
TAG=s11_ball_rs16_ell2_tau2_v1
H=${QI2_ROOT}/S11ball
case "$NAME" in
  cut8) CUT="--R-cut-rs 8 --R-collar-rs 5.333333333333333";;
  cut16) CUT="--R-cut-rs 16 --R-collar-rs 10.666666666666666";;
  *) echo "unknown collector $NAME" >&2; exit 2;;
esac
mkdir -p runs/s11 logs
EXTRA=""
[ -n "$TEND" ] && EXTRA="--T-end-rs $TEND"
echo "start $NAME $(date -u '+%Y-%m-%d %H:%M:%S UTC') device $DEV pid $$" >> runs/s11/status_s11page.txt
nice -n 10 $PY v6/page_history_v6c.py --record $H/collapse_$TAG.json --fields $H/collapse_${TAG}_fields.npz --fields-key applied --nsites 1024 --channels 1,2 $CUT --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --join-rs 32,64 --max-wall-hours 20 --device $DEV --threads 16 --out runs/s11 --tag s11page_$NAME $EXTRA > logs/s11page_$NAME.log 2>&1
echo "exit $NAME $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> runs/s11/status_s11page.txt
