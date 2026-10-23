#!/bin/bash
# Task 6 shard B (version 3 of the wrapper): waits until any of GPU 0 (shard A's list done), GPU 1 (the sphere's cap-1024 control exited) or
# GPU 2 (D-a1's E32 exited) is free of this lane's work, then runs launch_task6_v3.sh (flow_page_v2.py) there with task6_list_B_v1.txt.
# Builder analyst.
cd ${QI2_ROOT}/flow
while true; do
  if grep -q "^DONE task6_list_A_v1.txt" task6/status_v2.txt 2>/dev/null; then G=0; break; fi
  if grep -q "^S3 cap1024 (2,1,0) exit" ../sS_entropy/S3v3ctl/status.txt 2>/dev/null; then G=1; break; fi
  if grep -q "^Da1 E32 exit" ../sS_entropy/Da1v3/status.txt 2>/dev/null; then G=2; break; fi
  sleep 30
done
echo "shard B on GPU $G from $(date -u +%H:%M:%S)"
exec bash launch_task6_v3.sh $G task6_list_B_v1.txt - -
