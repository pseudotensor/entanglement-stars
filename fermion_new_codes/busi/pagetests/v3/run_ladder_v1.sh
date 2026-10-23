#!/bin/bash
# run_ladder_v1.sh (busi; the B' dt-ladder diagnostic, NOT s = 16 production; inbox #461 approval). ONE free H100, about 0.3 card-hours.
# The orthonormal F1 s = 16 prep (prep/F1_s16_d0_prep.npz from prep_v5), 32 columns (16 supply + 16 exit), clocks 6.93 L and 10 L,
# dt = 4, 2, 1, 0.5 a, run_bprime_v5.py with checkpoints every 300 s (re-running resumes); then ladder_read_v1.py.
# Split per inbox #462: DTS="4 2" on one GPU, DTS=1 and DTS=0.5 on two others, each with its own OUT; busi reads them together.
# Usage: CUDA_VISIBLE_DEVICES=<g> DTS="<dts>" PY=<venv python> OUT=<dir> bash run_ladder_v1.sh      (DRY=1 DEV=cpu: CPU dry run, 2 columns, T 0.5 L)
set -u
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
PY=${PY:-python}; DEV=${DEV:-cuda:0}; OUT=${OUT:-$KIT/out_ladder}; DRY=${DRY:-0}
NC=32; TMAX=10; DTS=${DTS:-"4 2 1 0.5"}
if [ "$DRY" = "1" ]; then NC=2; TMAX=0.5; echo "DRY"; fi
mkdir -p "$OUT"; LOG="$OUT/run_ladder_v1.log"
echo "start $(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOG"
cd "$KIT" && md5sum -c --quiet MD5SUMS >> "$LOG" 2>&1 || { echo "STOP: kit md5 check FAILED" | tee -a "$LOG"; exit 3; }
cd "$KIT/v3"
for D in $DTS; do
  $PY run_bprime_v5.py --phase run --prep ../prep/F1_s16_d0_prep.npz --device $DEV --card --T-max-over-L $TMAX --extra-T-over-L $TMAX \
     --max-columns $NC --col-batch $NC --dt $D --threads 4 --ckpt-every-s 300 --resume --out "$OUT" --tag ladder_dt$D >> "$OUT/ladder_dt$D.log" 2>&1
  echo "$(date -u +%FT%TZ) dt $D rc $?" | tee -a "$LOG"
done
if [ -f "$OUT/ladder_dt4.npz" ] && [ -f "$OUT/ladder_dt2.npz" ] && [ -f "$OUT/ladder_dt1.npz" ] && [ -f "$OUT/ladder_dt0.5.npz" ]; then $PY ladder_read_v1.py --dir "$OUT" --dts 4,2,1,0.5 | tee -a "$LOG"; fi
rm -rf "$OUT"/ladder_dt*_ckpt
cd "$OUT" && md5sum $(ls | grep -v MD5SUMS) > MD5SUMS
echo "done $(date -u +%FT%TZ)" | tee -a "$LOG"
