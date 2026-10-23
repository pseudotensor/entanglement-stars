#!/bin/bash
# PAGE v3.1 stage 2 (inbox #438 (c) cleared): the native held joint-output audit, k = 1, record 100. Waits until GPU 1 or GPU 2 is free
# (memory < 2000 MiB and utilization < 10 percent at two checks 60 s apart), then runs there. Never cuda:0.
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
echo "$(date -u +%FT%TZ) waiting for a free GPU 1 or 2" >> $LOG/queue_stage2_v1.log
while true; do
  g=$(free_gpu)
  if [ -n "$g" ]; then sleep 60; g2=$(free_gpu); if [ "$g2" = "$g" ]; then break; fi; fi
  sleep 120
done
echo "$(date -u +%FT%TZ) start on GPU $g ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $g))" >> $LOG/queue_stage2_v1.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 $PY pt3_stage2_held_audit.py --device cuda:$g --out $OUT --tag S2_k1_rec100_held > $LOG/S2_k1_rec100_held.log 2>&1
echo "$(date -u +%FT%TZ) audit rc $?" >> $LOG/queue_stage2_v1.log
nice -n 10 $PY pt3_stage2_read.py --stem $OUT/S2_k1_rec100_held > $LOG/S2_k1_rec100_held_read.log 2>&1
echo "$(date -u +%FT%TZ) read rc $?" >> $LOG/queue_stage2_v1.log
