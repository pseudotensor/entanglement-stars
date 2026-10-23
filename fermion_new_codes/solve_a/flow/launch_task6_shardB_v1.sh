#!/bin/bash
# Task 6 shard B: waits until the sphere control on GPU 1 or D-a1's E32 on GPU 2 has exited, then runs launch_task6_v2.sh on that GPU with
# task6_list_B_v1.txt (the m = 0.03 channels and the j = 1/2 controls).  Builder analyst.
cd ${QI2_ROOT}/flow
while true; do
  if grep -q "^S3 cap1024 (2,1,0) exit" ../sS_entropy/S3v3ctl/status.txt 2>/dev/null; then G=1; break; fi
  if grep -q "^Da1 E32 exit" ../sS_entropy/Da1v3/status.txt 2>/dev/null; then G=2; break; fi
  sleep 30
done
echo "shard B on GPU $G from $(date -u +%H:%M:%S)"
exec bash launch_task6_v2.sh $G task6_list_B_v1.txt - -
