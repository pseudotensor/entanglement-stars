#!/bin/bash
# Box-side batch: wait for strand 3's run to finish, then launch strand 0 v3 (4 threads), the strand 2a gate (<= 3 threads),
# strand 2c (2 GPU processes x 2 threads, only if GPUs 0 and 2 are free at launch) and the 2b Jacobian diagnostic (1 thread).
LOG=${QI2_ROOT}/batch_after_s3_v1.log
until grep -q DONE ${QI2_ROOT}/status.txt; do sleep 20; done
date -u '+s3 done seen %Y-%m-%d %H:%M:%S UTC' > $LOG
(cd ${QI2_ROOT}/s0_source_allocation && cp code/launch_strand0_full_v3.sh . && setsid nohup bash launch_strand0_full_v3.sh > /dev/null 2>&1 < /dev/null &)
(cd ${QI2_ROOT}/s2a_gate && cp code/launch_strand2a_v1.sh . && setsid nohup bash launch_strand2a_v1.sh > /dev/null 2>&1 < /dev/null &)
(cd ${QI2_ROOT}/s2b_row && cp code/launch_strand2b_debug1.sh . && setsid nohup bash launch_strand2b_debug1.sh > /dev/null 2>&1 < /dev/null &)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $LOG
u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
if [ "$u0" -lt 2000 ] && [ "$u2" -lt 2000 ]; then
  (cd ${QI2_ROOT}/s2c_mismatch && cp code/launch_strand2c_v1.sh . && setsid nohup bash launch_strand2c_v1.sh > /dev/null 2>&1 < /dev/null &)
  echo "2c launched on GPUs 0 and 2" >> $LOG
else
  echo "2c NOT launched: GPU 0 used $u0 MiB, GPU 2 used $u2 MiB" >> $LOG
fi
echo "batch launched" >> $LOG
