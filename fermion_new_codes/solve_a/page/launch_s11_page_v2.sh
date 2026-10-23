#!/bin/bash
# launch_s11_page_v2.sh (the Page builder, analyst, 2026-10-17): launch_s11_page_v1.sh (the D/S11 builder's, md5 349cb439b477e9bcd777f45c6dcc6d63)
# on page_history_v6i.py in place of page_history_v6c.py, every setting and the history unchanged; the tags s11pagei_<collector>. Why: v6c
# raises KeyError at the W_R accumulation when a peak restart (--transfer-at-peak) leaves one channel with no partner mode (v6cA_s4far on
# 2026-10-16 21:20 UTC; fixed in v6d), and its partners carry the sea's and the archive's Gram deviations over f(1 - f) (v6h, v6i); the
# state, the collector and every row are v6c's (v6d to v6i change the ledger, the traces' summation order and add readouts). The gate: the
# history oracle v6i (box, page/v6/runs/history_oracle_v6i_run1, pass). Usage (from ${QI2_ROOT}/page on the box):
# bash v6/launch_s11_page_v2.sh <cut8|cut16> [device: cpu | cuda:N] [T-end in r_s]
# v1's text: launch_s11_page_v1.sh (builder analyst; S11 of the quasi-frozen ruling: "the Page recursion of S1 running alongside with its collectors
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
$PY -c "import json, sys; d = json.load(open('v6/runs/history_oracle_v6i_run1/history_oracle_v6i_results.json')); sys.exit(0 if d.get('pass') and d.get('code') == 'page_history_oracle_v6i.py' else 3)" || { echo "the history oracle v6i has not passed: no launch"; exit 3; }
EXTRA=""
[ -n "$TEND" ] && EXTRA="--T-end-rs $TEND"
echo "start s11pagei_$NAME $(date -u '+%Y-%m-%d %H:%M:%S UTC') device $DEV pid $$ (page_history_v6i.py)" >> runs/s11/status_s11page.txt
nice -n 10 $PY v6/page_history_v6i.py --record $H/collapse_$TAG.json --fields $H/collapse_${TAG}_fields.npz --fields-key applied --nsites 1024 --channels 1,2 $CUT --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --join-rs 32,64 --max-wall-hours 20 --device $DEV --threads 16 --out runs/s11 --tag s11pagei_$NAME $EXTRA > logs/s11pagei_$NAME.log 2>&1
echo "exit s11pagei_$NAME $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> runs/s11/status_s11page.txt
