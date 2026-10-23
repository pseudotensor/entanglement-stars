#!/bin/bash
# PAGE v3.1 stage 2 as released by reviewer A pst1 (inbox #442): the massless k = 1 probe, start (b) (f_ws(h(0)) ladder), held at record 100,
# trace reads only (no rows on this path). Waits for GPU 2 (released by NOTCOMP) or GPU 1, whichever is free first (memory < 2000 MiB and
# utilization < 10 percent at two checks 60 s apart; nvidia-smi logged at start). Never cuda:0. Then the held Killing-vacuum baseline (item 4):
# a short run to u = 1000 r_s and its held-step halving.
set -u
cd ${QI2_ROOT}/v3
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/stage2
LOG=${QI2_ROOT}/logs
mkdir -p $OUT
free_gpu() {
  for g in 2 1; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    if [ "$m" -lt 2000 ] && [ "$u" -lt 10 ]; then echo $g; return; fi
  done
}
echo "$(date -u +%FT%TZ) v2 waiting for a free GPU 2 or 1" >> $LOG/queue_stage2_v1.log
while true; do
  g=$(free_gpu)
  if [ -n "$g" ]; then sleep 60; g2=$(free_gpu); if [ "$g2" = "$g" ]; then break; fi; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) start on GPU $g: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')" >> $LOG/queue_stage2_v1.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() { # tag args...
  t=$1; shift
  nice -n 10 $PY pt3_stage2_held_audit.py --device cuda:$g --out $OUT --tag $t "$@" > $LOG/$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t rc $?" >> $LOG/queue_stage2_v1.log
  nice -n 10 $PY pt3_stage2_read.py --stem $OUT/$t > $LOG/${t}_read.log 2>&1
  echo "$(date -u +%FT%TZ) $t read rc $?" >> $LOG/queue_stage2_v1.log
}
run S2_k1_rec100_held
run S2_vac_baseline_u1000 --baseline-vacuum --u-max-rs 1000
run S2_vac_baseline_u1000_half --baseline-vacuum --u-max-rs 1000 --held-chunk 2048
echo "$(date -u +%FT%TZ) stage 2 queue done" >> $LOG/queue_stage2_v1.log
