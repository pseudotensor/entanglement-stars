#!/bin/bash
# box_gen_queue_v1.sh (solve builder 6, analyst; engine spec 47.1 (2) (v), (vi); 47.7 (2); 47.6 (1)): the corrected selector's runs on the box
# A6000s, one per card, page_history_v6l.py on A's production line (s4's settings, the record H100b/C3prod19 ..._r11, 0 to 40), each with
# --collector-select gen, --save-every 1, its output and states on ${QI2_ROOT}/runs<tag> (rule 9), its own guard
# (the card's free memory below 8192 MiB stops that python by its PID, re-queued once with --resume):
#   GPU 1: v6lA_s4gensplit (47.1 (2) (v): --collector-admit split);
#   GPU 0: v6lA_s4loc3gen (47.1 (2) (vi): eps_loc 1e-3), started only after the GPU 1 run has written 8 checkpoint rows and its device peak
#          plus 8 GiB fits the 48 GiB card;
#   GPU 2: v6lA_s4k3gen (47.1 (2) (vi): the kappa 3 set, channels 1,2,3), started when B's rerun (Brerun98) has ended and the card is free.
# Not here: the far collector at its matched arrival clock and the regression under gen (with the lead). Usage (box): setsid nohup bash box_gen_queue_v1.sh &
set -u
V6=${QI2_ROOT}/v6; E=${QI2_ROOT}/engine; O=${QI2_ROOT}/page_v6l; LOG=$O/box_gen_queue_v1.log
PY=${QI2_ROOT}/python
mkdir -p $O/runs $O/logs
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
BASE="--record $E/H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $E/H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied --nsites 8192 --R-collar-rs 16 --R-cut-rs 24 --collect-every-rs 2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 1 --max-wall-hours 30 --threads 8 --collector-select gen --resume"
free_of() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d ' '; }
run_one() {  # $1 gpu, $2 tag, rest extra args; guard; one --resume retry after a guard stop or a nonzero exit
  local g=$1 tag=$2; shift 2; local tries=0
  while [ $tries -lt 2 ]; do
    tries=$((tries + 1))
    ( cd $V6; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      exec nice -n 10 $PY page_history_v6l.py $BASE --device cuda:$g --out $O/runs/$tag --tag $tag "$@" >> $O/logs/$tag.log 2>&1 ) &
    local P=$! G=0
    log "start $tag on GPU $g (try $tries) pid $P; free $(free_of $g) MiB"
    while kill -0 $P 2>/dev/null; do
      sleep 30; f=$(free_of $g)
      if [ -n "$f" ] && [ "$f" -lt 8192 ]; then kill $P; G=1; log "GUARD: GPU $g free $f MiB below 8192: $tag stopped by PID $P"; break; fi
    done
    wait $P; rc=$?; log "exit $tag $rc (guard $G)"
    [ $G -eq 0 ] && [ $rc -eq 0 ] && return 0
  done
  log "$tag: left for the lead after 2 tries"
}
rows_of() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get('rows',[]); print(len(r), max([(x.get('device_peak_GiB') or {}).get('reserved', 0) for x in r if isinstance(x.get('device_peak_GiB'), dict)] or [0]))" "$1" 2>/dev/null || echo "0 0"; }
log "queue start"
run_one 1 v6lA_s4gensplit --channels 1,2 --eps-loc 1e-2 --collector-admit split &
Q1=$!
# GPU 0 waits for the GPU 1 run's 8 rows and its peak
while true; do
  read n pk <<< "$(rows_of $O/runs/v6lA_s4gensplit/v6lA_s4gensplit_summary.json)"
  if [ "${n:-0}" -ge 8 ]; then
    if $PY -c "import sys; sys.exit(0 if float('$pk') + 8.0 <= 48.0 else 1)"; then log "GPU 1 run: $n rows, device peak $pk GiB: GPU 0 run starts"; break
    else log "GPU 1 run: device peak $pk GiB + 8 above 48: GPU 0 run not started (with the lead)"; pk=none; break; fi
  fi
  kill -0 $Q1 2>/dev/null || { log "GPU 1 run ended before 8 rows"; break; }
  sleep 120
done
[ "$pk" != none ] && { run_one 0 v6lA_s4loc3gen --channels 1,2 --eps-loc 1e-3 & }
# GPU 2 after B's rerun
while ps -eo args | grep -q "[r]un_collapse_v19_from_slice_v1.py.*ext100_r98"; do sleep 120; done
log "Brerun98 ended; GPU 2 free $(free_of 2) MiB"
[ "$pk" != none ] && run_one 2 v6lA_s4k3gen --channels 1,2,3 --eps-loc 1e-2 &
wait
log "queue done"
