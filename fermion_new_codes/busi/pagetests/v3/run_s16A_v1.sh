#!/bin/bash
# run_s16A_v1.sh (busi; B' s = 16 STAGE A only, ReviewA_bpchk2_v1 sec. 5 and ReviewB_bpchk2_v1 sec. 3B). ONE free H100, alone.
# START ONLY AFTER busi POSTS THE BENCHMARK PASS. Runs, in order, with the benchmarked run_bprime_v4.py unchanged:
#   1  F1 d0, all 256 columns, to 1e2 L, dt 4 a          (prep F1_s16_d0,         port=sink2L onset 2 L, eta_max 1/a)
#   2a eta_max x 2, to 1e2 L, dt 4 a                     (prep F1_s16_d0_eta2_bb, eta_max 2/a; the base input basis)
#   2c dt = 2 a control, to 1e2 L                        (prep F1_s16_d0)
#   2b onset 2.25 L, half-side 2.75 L, to 1e2 L, dt 4 a  (prep F1_s16_d0_onset225_bb; the base basis)
# In-run reader: supply_filled (hypothesis tag 3); the other hypotheses are read on the box from the saved G and R (zero card cost).
# Every output is a sink2L quantity under a declared covariance (stageA_manifest_v1.json in the kit). Then STOP: busi reads stage A;
# stage B and C are separate lines. Per-run cap: CAP_H hours (default 8); stage cap: STAGE_CAP_H (default 28).
# Usage: PY=${QI2_ROOT}/python DEV=cuda:0 BATCH=128 bash run_s16A_v1.sh   (DRY=1 DEV=cpu: the CPU dry run)
set -u
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
PY=${PY:-python}; DEV=${DEV:-cuda:0}; OUT=${OUT:-$KIT/out}; BATCH=${BATCH:-128}
CAP_H=${CAP_H:-8}; STAGE_CAP_H=${STAGE_CAP_H:-28}; DRY=${DRY:-0}
IDX=${DEV#cuda:}
mkdir -p "$OUT"; LOG="$OUT/run_s16A_v1.log"
echo "start $(date -u +%FT%TZ) kit $KIT dev $DEV batch $BATCH cap ${CAP_H} h per run, stage ${STAGE_CAP_H} h" | tee -a "$LOG"
TMAX=100; EXTRA=""
if [ "$DRY" = "1" ]; then TMAX=1; EXTRA="--max-columns 4"; BATCH=2; CAP_H=1; echo "DRY run (CPU, 4 columns, T = 1 L)" | tee -a "$LOG"; fi
if [ "$DRY" != "1" ]; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$IDX" | tr -d ' ')
  if [ -z "$USED" ] || [ "$USED" -gt 2000 ]; then echo "STOP: card $IDX not free (used ${USED} MiB)" | tee -a "$LOG"; exit 2; fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv -i "$IDX" | tee -a "$LOG"
fi
cd "$KIT" && md5sum -c MD5SUMS >> "$LOG" 2>&1 || { echo "STOP: kit md5 check FAILED" | tee -a "$LOG"; exit 3; }
cd "$KIT/v3"
T_START=$(date +%s)
run() { # tag prep dt
  local tag=$1 prep=$2 dt=$3
  local el=$(( $(date +%s) - T_START ))
  if [ $el -gt $(( STAGE_CAP_H * 3600 )) ]; then echo "STOP: stage cap reached before $tag" | tee -a "$LOG"; return 9; fi
  echo "$(date -u +%FT%TZ) begin $tag" | tee -a "$LOG"
  timeout $(( CAP_H * 3600 )) $PY run_bprime_v4.py --phase run --prep ../prep/${prep}_prep.npz --device $DEV --card --T-max-over-L $TMAX \
     --dt $dt --col-batch $BATCH --cin supply_filled --threads 4 --out "$OUT" --tag $tag $EXTRA >> "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $tag rc $rc" | tee -a "$LOG"
  if [ $rc -eq 124 ]; then echo "STOP RULE: $tag hit its ${CAP_H} h cap; the remaining runs are not started" | tee -a "$LOG"; return 8; fi
  if [ $rc -ne 0 ]; then echo "STOP RULE: $tag failed (rc $rc); the remaining runs are not started" | tee -a "$LOG"; return 7; fi
  return 0
}
run s16A_1_F1_d0_T100 F1_s16_d0 4 && run s16A_2a_F1_eta2_T100 F1_s16_d0_eta2_bb 4 && run s16A_2c_F1_dt2_T100 F1_s16_d0 2 && \
  run s16A_2b_F1_onset225_T100 F1_s16_d0_onset225_bb 4
echo "rc chain $?" | tee -a "$LOG"
cp "$KIT/prep/stageA_manifest_v1.json" "$OUT/" 2>/dev/null
cd "$OUT" && md5sum $(ls | grep -v MD5SUMS) > MD5SUMS
echo "done $(date -u +%FT%TZ): STAGE A ends here; do not start stage B" | tee -a "$LOG"
