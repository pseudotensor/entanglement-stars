#!/bin/bash
# run_bench_v1.sh (busi; the B' short H100 benchmark of PAGE_Bprime_design_v2.md sec. 6). ONE free H100, alone; about 1 card-hour.
# No production arm: three short propagations (T = 10 L) of the s = 16 F1 window (all 256 columns, in batches of 128, and of 64 for the
# cross-batch check; a single batch of 256 would need about 100 GB) plus an apply benchmark.
# Usage (kit root = this script's directory):  PY=/path/to/python DEV=cuda:0 bash run_bench_v1.sh   (DRY=1 DEV=cpu: the CPU dry run)
# Needs python with numpy, scipy, torch (CUDA). Output: $OUT (default <kit>/out), with MD5SUMS; then pull it back.
set -u
KIT=${KIT:-$(cd "$(dirname "$0")" && pwd)}
PY=${PY:-python}
DEV=${DEV:-cuda:0}
OUT=${OUT:-$KIT/out}
IDX=${DEV#cuda:}
mkdir -p "$OUT"
LOG="$OUT/run_bench_v1.log"
echo "start $(date -u +%FT%TZ) kit $KIT dev $DEV" | tee -a "$LOG"
DRY=${DRY:-0}
EXTRA=""; TMAX=10; BA=128; BB=64; BATCHES=1,8,32,64,128
if [ "$DRY" = "1" ]; then EXTRA="--max-columns 4"; TMAX=1; BA=4; BB=2; BATCHES=1,2; echo "DRY run (CPU, 4 columns, T = 1 L)" | tee -a "$LOG"; fi
if [ "$DRY" != "1" ]; then
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$IDX" | tr -d ' ')
if [ -z "$USED" ] || [ "$USED" -gt 2000 ]; then echo "card $IDX not free (used ${USED} MiB): stop" | tee -a "$LOG"; exit 2; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv -i "$IDX" | tee -a "$LOG"
fi
cd "$KIT" && md5sum -c MD5SUMS >> "$LOG" 2>&1 || { echo "kit md5 check FAILED" | tee -a "$LOG"; exit 3; }
cd "$KIT/v3"
P=../prep/F1_s16_d0_prep.npz
$PY bench_cube_apply_v1.py --prep $P --device $DEV --batches $BATCHES --out "$OUT" >> "$LOG" 2>&1 || echo "apply bench failed" | tee -a "$LOG"
COMMON="--phase run --prep $P --device $DEV --card --T-max-over-L $TMAX --extra-T-over-L $TMAX --threads 4 --out $OUT $EXTRA"
$PY run_bprime_v4.py $COMMON --dt 4 --col-batch $BA --tag bench_F1_s16_dt4_T10 >> "$LOG" 2>&1 || echo "dt4 failed" | tee -a "$LOG"
$PY run_bprime_v4.py $COMMON --dt 2 --col-batch $BA --tag bench_F1_s16_dt2_T10 >> "$LOG" 2>&1 || echo "dt2 failed" | tee -a "$LOG"
$PY run_bprime_v4.py $COMMON --dt 4 --col-batch $BB --tag bench_F1_s16_dt4_T10_bB >> "$LOG" 2>&1 || echo "batched failed" | tee -a "$LOG"
$PY bench_compare_v1.py --out "$OUT" --prep-jsons ../prep/*_prep.json >> "$LOG" 2>&1 || echo "compare failed" | tee -a "$LOG"
cd "$OUT" && md5sum $(ls | grep -v MD5SUMS) > MD5SUMS
echo "done $(date -u +%FT%TZ)" | tee -a "$LOG"
